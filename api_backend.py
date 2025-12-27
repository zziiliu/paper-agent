import os
import json
import logging
import time
import re
from datetime import datetime
from collections import Counter, deque
from functools import lru_cache
from typing import List, Optional, Dict, Any, Tuple, Deque

import chromadb
import requests
from fastapi import FastAPI, HTTPException, Form, BackgroundTasks, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from get_paper import ArxivPaperFetcher

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# Load env first
load_dotenv()

# Environment / config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "deepseek"
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
ARXIV_MAX_PER_CALL = int(os.getenv("ARXIV_MAX_PER_CALL", "1000"))
MAIN_SERVER = os.getenv("MAIN_SERVER", "http://1.95.125.201").rstrip("/")
REFRESH_MAX_AGE_HOURS = float(os.getenv("REFRESH_MAX_AGE_HOURS", "6"))
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "deepseek-chat")
CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "6"))
QUERY_CONTEXT_MAX = int(os.getenv("QUERY_CONTEXT_MAX", "20"))
DEFAULT_MAX_RESULTS = int(os.getenv("DEFAULT_MAX_RESULTS", "3"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---- Embedding model ----
@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---- Knowledge base (Chroma) ----
class ArxivKnowledgeBase:
    def __init__(self, persist_directory: str = CHROMA_DB_DIR, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        os.makedirs(persist_directory, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
        except Exception:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(name="arxiv_papers")
        self.embedding_model = get_embedding_model()
        self.existing_ids: set[str] = set()
        self.category_counts: Counter = Counter()
        self.total_count: int = 0
        self._load_existing_stats()

    def _load_existing_stats(self):
        try:
            data = self.collection.get(include=["ids", "metadatas"])
            ids = data.get("ids", []) or []
            metadatas = data.get("metadatas", []) or []
            # metadatas may be nested lists depending on backend
            flat_metas: List[Dict[str, Any]] = []
            for m in metadatas:
                if isinstance(m, list):
                    flat_metas.extend([x for x in m if isinstance(x, dict)])
                elif isinstance(m, dict):
                    flat_metas.append(m)

            self.existing_ids = set(ids)
            self.total_count = len(self.existing_ids)
            self.category_counts = Counter()
            for meta in flat_metas:
                cat = meta.get("primary_category")
                if cat:
                    self.category_counts[cat] += 1
        except Exception as e:
            logger.warning("load_existing_stats failed: %s", e)
            self.existing_ids = set()
            self.category_counts = Counter()
            self.total_count = 0
        logger.info("Knowledge base initialized with %d papers", len(self.existing_ids))

    def count(self) -> int:
        try:
            data = self.collection.get()
            return len(data.get("ids", []))
        except Exception:
            return len(self.existing_ids)

    def get_stats(self, top_k: int = 5) -> Tuple[int, List[Tuple[str, int, float]]]:
        total = self.total_count or len(self.existing_ids)
        if total == 0:
            try:
                total = self.collection.count()
            except Exception:
                total = 0
        if not self.category_counts and total > 0:
            try:
                limit = min(total, 2000)
                metas_raw = self.collection.get(include=["metadatas"], limit=limit).get("metadatas") or []
                flat_metas: List[Dict[str, Any]] = []
                for m in metas_raw:
                    if isinstance(m, list):
                        flat_metas.extend([x for x in m if isinstance(x, dict)])
                    elif isinstance(m, dict):
                        flat_metas.append(m)
                counter = Counter()
                for meta in flat_metas:
                    cat = meta.get("primary_category")
                    if cat:
                        counter[cat] += 1
                self.category_counts = counter
            except Exception as e:
                logger.warning("get_stats reload metadatas failed: %s", e)
        if total == 0:
            return 0, []
        pairs = self.category_counts.most_common(top_k) if self.category_counts else []
        stats = []
        for cat, cnt in pairs:
            pct = (cnt / total * 100) if total else 0
            stats.append((cat, cnt, pct))
        return total, stats

    def add_papers(self, papers: List[Dict[str, Any]]) -> int:
        if not papers:
            return 0

        new_docs = []
        new_metas = []
        new_ids = []

        for p in papers:
            pid = p.get("id")
            if not pid:
                continue
            if pid in self.existing_ids:
                continue

            abstract = p.get("abstract", "") or ""
            # 使用完整摘要提升召回质量
            doc = f"Title: {p.get('title','')}\nAbstract: {abstract}"

            new_docs.append(doc)
            new_metas.append(
                {
                    "title": p.get("title"),
                    "authors": ", ".join(p.get("authors", [])),
                    "published": p.get("published"),
                    "primary_category": p.get("primary_category"),
                    "pdf_url": p.get("pdf_url"),
                    "arxiv_url": p.get("arxiv_url"),
                }
            )
            new_ids.append(pid)

        if not new_ids:
            return 0

        embeddings = self.embedding_model.encode(new_docs, show_progress_bar=False, convert_to_numpy=True)

        self.collection.add(
            documents=new_docs,
            metadatas=new_metas,
            ids=new_ids,
            embeddings=embeddings.tolist(),
        )
        self.existing_ids.update(new_ids)
        self.total_count += len(new_ids)
        for meta in new_metas:
            cat = meta.get("primary_category")
            if cat:
                self.category_counts[cat] += 1
        return len(new_ids)

    def query_similar(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        try:
            q_emb = self.embedding_model.encode([query], convert_to_numpy=True)
            res = self.collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=n_results,
                include=["documents", "metadatas", "distances", "data"],
            )
            ids = []
            data_list = res.get("data") or []
            for data_entry in data_list:
                if isinstance(data_entry, dict) and "id" in data_entry:
                    ids.append(data_entry["id"])

            documents = res.get("documents", [[]])[0] if res.get("documents") else []
            metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
            distances = res.get("distances", [[]])[0] if res.get("distances") else []

            return {"ids": ids, "documents": documents, "metadatas": metadatas, "distances": distances}
        except Exception as e:
            logger.exception("query_similar failed: %s", e)
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}


# ---- LLM recommendation agent ----
class PaperRecommendationAgent:
    def __init__(self, kb: ArxivKnowledgeBase, model_name: str = "deepseek-chat"):
        self.kb = kb
        self.model_name = model_name

        if LLM_PROVIDER == "openai":
            if OpenAI is None or not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is missing")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif LLM_PROVIDER == "deepseek":
            if OpenAI is None or not DEEPSEEK_API_KEY:
                raise RuntimeError("DEEPSEEK_API_KEY is missing")
            self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
        else:
            raise RuntimeError("Unknown LLM_PROVIDER, set LLM_PROVIDER=openai or deepseek")

    def _build_context_from_search(self, search_res: Dict[str, Any]):
        documents = search_res.get("documents", [])
        metadatas = search_res.get("metadatas", [])
        ctx_text = ""
        structured = []

        for i, doc_text in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            if not isinstance(meta, dict):
                meta = {}

            title = meta.get("title", "Unknown title")
            authors = meta.get("authors", "")
            pdf_url = meta.get("pdf_url", "")
            arxiv_url = meta.get("arxiv_url", "")
            published = meta.get("published", "")

            ctx_text += (
                f"Title: {title}\nAuthors: {authors}\nLink: {arxiv_url or pdf_url}\nPublished: {published}\nAbstract: {doc_text}\n\n"
            )
            structured.append(
                {"title": title, "authors": authors, "link": arxiv_url or pdf_url, "abstract": doc_text, "published": published}
            )

        return ctx_text, structured

    def recommend_papers(self, user_query: str, max_papers: int = 3) -> Dict[str, Any]:
        search_res = self.kb.query_similar(user_query, n_results=max_papers)
        if not search_res.get("documents") or not search_res["documents"][0]:
            return {"error": "no related papers found"}

        ctx_text, structured = self._build_context_from_search(search_res)
        prompt = f"""
You are an academic paper recommender. User query: {user_query}

Candidate papers:
{ctx_text}

Return up to {max_papers} recommendations in valid JSON:
{{
  "recommendations": [
    {{
      "title": "...",
      "authors": "...",
      "link": "...",
      "reason": "...",
      "score": "number"
    }}
  ]
}}

Rules:
- Output JSON only.
- Keep score between 1-10.
- Write the "reason" field in concise Chinese; keep titles/authors/links in their original language.
- If there are fewer than {max_papers} candidates, return as many as available.
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a scholarly paper recommendation assistant. Output JSON only. Use Chinese for all explanations (the reason field), while keeping titles/authors/links unchanged.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        text = response.choices[0].message.content.strip()
        parsed = self._robust_json_parse(text, max_papers, structured)
        return parsed

    def _robust_json_parse(self, text: str, max_papers: int, structured: List[Dict[str, Any]]):
        import re

        try:
            parsed = json.loads(text)
            return {"text": None, "json": parsed, "structured": structured}
        except Exception:
            pass

        json_patterns = [r"\{.*\}", r"\[.*\]"]
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    cleaned = self._clean_json_string(match)
                    parsed = json.loads(cleaned)
                    return {"text": None, "json": parsed, "structured": structured}
                except Exception:
                    continue

        extracted = self._extract_recommendations_from_text(text, max_papers)
        if extracted:
            return {"text": None, "json": {"recommendations": extracted}, "structured": structured}

        return {"text": text, "json": None, "structured": structured}

    def _clean_json_string(self, json_str: str) -> str:
        import re

        fixes = [(r",\s*}", "}"), (r",\s*]", "]")]
        cleaned = json_str
        for pattern, replacement in fixes:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned

    def _extract_recommendations_from_text(self, text: str, max_papers: int):
        import re

        recommendations = []
        lines = text.splitlines()
        current = {}

        for line in lines:
            line = line.strip()

            title_match = re.match(r".*title[:：]?\s*(.+)", line, re.IGNORECASE)
            if title_match and "title" not in current:
                current["title"] = title_match.group(1).strip('"\' ')
                continue

            author_match = re.match(r".*author[:：]?\s*(.+)", line, re.IGNORECASE)
            if author_match and "authors" not in current:
                current["authors"] = author_match.group(1).strip('"\' ')
                continue

            link_match = re.search(r"https?://[^\s]+", line)
            if link_match and "link" not in current:
                current["link"] = link_match.group(0)
                continue

            reason_match = re.match(r".*reason[:：]?\s*(.+)", line, re.IGNORECASE)
            if reason_match and "reason" not in current:
                current["reason"] = reason_match.group(1).strip('"\' ')
                continue

            score_match = re.match(r".*score[:：]?\s*([0-9.]+)", line, re.IGNORECASE)
            if score_match and "score" not in current:
                current["score"] = score_match.group(1)
                continue

            if len(current) >= 3:
                recommendations.append(current.copy())
                current = {}
                if len(recommendations) >= max_papers:
                    break

        if current and len(current) >= 3:
            recommendations.append(current)

        return recommendations if recommendations else None


# ---- FastAPI wiring ----
app = FastAPI(title="ArXiv RAG Backend", version="0.1.0")
fetcher = ArxivPaperFetcher(cache_file="arxiv_fetcher_cache.json")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

kb: Optional[ArxivKnowledgeBase] = None
agent: Optional[PaperRecommendationAgent] = None
LAST_REFRESH_TIME: Optional[float] = None
CHAT_SESSIONS: Dict[str, List[Tuple[str, str]]] = {}
QUERY_CONTEXT: Dict[str, deque] = {}
REQUEST_ID_CACHE: Dict[str, float] = {}

def ensure_initialized():
    global kb, agent
    if kb is None:
        kb = ArxivKnowledgeBase()
    if agent is None:
        agent = PaperRecommendationAgent(kb)
    return kb, agent


def _record_refresh_time():
    global LAST_REFRESH_TIME
    LAST_REFRESH_TIME = time.time()


def _should_refresh() -> bool:
    if REFRESH_MAX_AGE_HOURS <= 0:
        return False
    if LAST_REFRESH_TIME is None:
        return True
    return (time.time() - LAST_REFRESH_TIME) >= REFRESH_MAX_AGE_HOURS * 3600


def _refresh_knowledge_base(domains: Optional[List[str]] = None, target_count: Optional[int] = None) -> Tuple[int, int]:
    kb_obj, _ = ensure_initialized()
    domains = domains or fetcher.get_available_domains()
    if not domains:
        raise ValueError("domains list is empty")

    target = target_count if target_count is not None else min(ARXIV_MAX_PER_CALL, 30)
    target = min(target, ARXIV_MAX_PER_CALL)
    papers_per_domain = max(1, target // len(domains))
    new_papers = fetcher.fetch_balanced_papers(
        domains=domains, papers_per_domain=papers_per_domain, target_count=target
    )
    inserted = kb_obj.add_papers(new_papers)
    _record_refresh_time()
    return inserted, len(new_papers)


def _maybe_trigger_background_refresh(
    user_id: Optional[str], background_tasks: Optional[BackgroundTasks]
) -> bool:
    if not _should_refresh():
        return False

    if background_tasks is not None and user_id:
        background_tasks.add_task(run_refresh_and_notify, user_id)
        return True

    try:
        _refresh_knowledge_base()
        return True
    except Exception as e:
        logger.warning("auto refresh failed: %s", e)
        return False


def _build_welcome_message() -> str:
    base = "论文助手接入成功，直接输入你的问题或检索需求即可（支持更新、推荐和自然对话）。"
    try:
        kb_obj, _ = ensure_initialized()
        total, stats = kb_obj.get_stats(top_k=5)
        if total <= 0:
            return base + "\n当前本地库暂无论文，将根据需要自动补充。"

        parts = [base, f"当前本地论文数：{total} 篇。"]
        if stats:
            dist = "; ".join([f"{cat}: {cnt} 篇（{pct:.1f}%）" for cat, cnt, pct in stats])
            parts.append(f"领域分布：{dist}")
        return "\n".join(parts)
    except Exception as e:
        logger.warning("build welcome message failed: %s", e)
        return base


def _parse_date_to_ts(date_str: Optional[str]) -> Optional[float]:
    if not date_str:
        return None
    try:
        # Try ISO format first
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        pass
    try:
        dt = datetime.strptime(date_str[:7], "%Y-%m")
        return dt.timestamp()
    except Exception:
        pass
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.timestamp()
    except Exception:
        return None


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _extract_topic_for_display(query: str) -> str:
    if not query:
        return ""
    try:
        # 尝试从“查/找/检索...关于X的论文/文章”中提取 X
        m = re.search(r"(?:查|找|检索|搜|看看).{0,6}?(?:关于|相关的|有关的)?(.+?)(?:的)?(?:论文|文章)", query)
        if m:
            candidate = m.group(1).strip(" ：:，。！？ ")
            candidate = re.sub(r"(相关|相关的)$", "", candidate).strip()
            if candidate:
                return candidate
    except Exception:
        pass
    return query.strip()


def _infer_published_from_link(link: str, fallback: str) -> str:
    if link:
        # arxiv 链接形如 https://arxiv.org/abs/2511.07161v1 -> year=25, month=11
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d+", link)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            return f"20{year:02d}-{month:02d}"
    return fallback or "日期不详"


@lru_cache(maxsize=256)
def _translate_title(title: str) -> str:
    if not title or _contains_chinese(title):
        return ""
    try:
        client = _get_chat_client()
        resp = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是标题翻译助手，请将英文论文标题翻译为简洁的中文，保留关键信息，只返回翻译文本。"},
                {"role": "user", "content": title},
            ],
            temperature=0.2,
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("translate title failed: %s", e)
        return ""


def _expand_queries(query: str) -> List[str]:
    base = query.strip()
    variants = [base]
    lowered = base.lower()
    if ("物理" in base) and ("模型" in base or "大模型" in base):
        variants.extend(
            [
                "物理 世界模型",
                "物理 具身智能",
                "物理 大模型",
                "物理 模型 机器人",
            ]
        )
    if "机器人" in base or "robot" in lowered:
        variants.extend(
            [
                "机器人 具身智能",
                "机器人 行为规划",
                "机器人 世界模型",
            ]
        )
    # 去重保持顺序
    seen = set()
    uniq = []
    for v in variants:
        if v and v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def _extract_year_pref(query: str) -> Optional[int]:
    # 匹配“24年”“2024年”“24年的”“去年”
    m = re.search(r"(20)?(2[0-9])年", query)
    if m:
        year = int(m.group(2))
        return 2000 + year if year < 100 else year
    if "去年" in query:
        return datetime.now().year - 1
    if re.search(r"最新|近期|最近|近一年|近1年", query):
        return datetime.now().year
    return None


def _should_refresh_for_year(year_pref: Optional[int]) -> bool:
    """老年份的请求不需要触发补库刷新，只在查最新时刷新。"""
    if year_pref is None:
        return True
    current_year = datetime.now().year
    return year_pref >= current_year - 1


def recommend_with_fallback(agent_obj: PaperRecommendationAgent, query: str, max_papers: int = 3, year_pref: Optional[int] = None) -> Tuple[List[Dict[str, str]], str]:
    queries = _expand_queries(query)
    last_query = query
    for q in queries:
        last_query = q
        agent_res = agent_obj.recommend_papers(q, max_papers=max_papers)
        recs = _pick_recommendations(agent_res, q, max_papers=max_papers)
        if year_pref:
            recs = [r for r in recs if _infer_published_from_link(r.get("link") or "", r.get("published") or "").startswith(str(year_pref))]
        if recs:
            return recs, q
    return [], last_query


def _fallback_kb_recommend(kb_obj: ArxivKnowledgeBase, query: str, max_papers: int = 5, year_pref: Optional[int] = None) -> List[Dict[str, str]]:
    try:
        search_res = kb_obj.query_similar(query, n_results=max_papers)
        metadatas = search_res.get("metadatas", [[]])[0] if search_res else []
        docs = search_res.get("documents", [[]])[0] if search_res else []
        recs: List[Dict[str, str]] = []
        for idx, meta in enumerate(metadatas):
            if not isinstance(meta, dict):
                continue
            title = meta.get("title") or "未提供标题"
            link = meta.get("arxiv_url") or meta.get("pdf_url") or ""
            published = meta.get("published") or ""
            reason = "基于相似度的本地检索结果"
            recs.append({"title": title, "link": link, "reason": reason, "published": published, "abstract": docs[idx] if idx < len(docs) else ""})
            if len(recs) >= max_papers:
                break
        if year_pref:
            recs = [r for r in recs if _infer_published_from_link(r.get("link") or "", r.get("published") or "").startswith(str(year_pref))]
        return recs
    except Exception as e:
        logger.warning("fallback kb recommend failed: %s", e)
        return []


def _looks_like_paper_query(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    # 更聚焦学术/论文相关信号，避免日常“推荐/找”触发论文模式
    keywords = ["论文", "paper", "文献", "arxiv", "引用", "综述", "survey", "article"]
    return any(k in text for k in keywords) or any(k in lowered for k in keywords)


def _is_generic_paper_request(text: str) -> bool:
    """
    检测仅包含“推荐几篇论文/文献”但未提供领域/主题的情况，提示用户补充主题。
    """
    if not text:
        return True
    stripped = re.sub(r"[\s，。、！？,.!?]", "", text)
    # 去掉常见泛词后是否还剩主题
    generic_tokens = ["推荐", "几篇", "几条", "论文", "文献", "article", "paper", "给我", "想要", "可以", "吗", "推荐下", "推荐一下"]
    for tok in generic_tokens:
        stripped = stripped.replace(tok, "")
    # 只剩很短时认为缺少主题
    return len(stripped) <= 2


@lru_cache(maxsize=1)
def _get_chat_client():
    if OpenAI is None:
        raise RuntimeError("OpenAI client not available")
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing")
        return OpenAI(api_key=OPENAI_API_KEY)
    if LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is missing")
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    raise RuntimeError("Unknown LLM_PROVIDER, set LLM_PROVIDER=openai or deepseek")


def analyze_intent(user_query: str) -> Dict[str, Any]:
    """
    使用 LLM 分析用户意图，返回结构化信息：
    {
        "intent": "paper" | "chat",
        "query": str,
        "max_results": int,
        "year": int | null,
        "wants_latest": bool,
        "wants_refresh": bool
    }
    """
    client = _get_chat_client()
    prompt = f"""
你是一个意图分类器。请判断用户是想**找学术论文**，还是**闲聊/问生活建议**。

### 🚨 判别核心标准
1. **intent="paper"**: 
   - 用户明确想要**学术文献、参考文献、Arxiv论文**。
   - 关键词：论文、paper、文献、引用、综述、Arxiv、书单。
   - ⚠️ 注意：仅仅出现“推荐”二字**不一定**是找论文！

2. **intent="chat"**: 
   - 生活类问题：旅游、美食、电影、小说、景点推荐。
   - 科普类问题：什么是X、介绍一下X。
   - 日常闲聊：你好、晚安、情感表达。

### 🌰 参考示例 (请严格模仿)
用户输入: "推荐北京好玩的景点" -> {{"intent": "chat", "query": "推荐北京好玩的景点"}}  <-- 生活推荐 = Chat
用户输入: "有什么好吃的川菜推荐吗" -> {{"intent": "chat", "query": "有什么好吃的川菜推荐吗"}} <-- 美食推荐 = Chat
用户输入: "推荐几部好看的科幻电影" -> {{"intent": "chat", "query": "推荐几部好看的科幻电影"}} <-- 娱乐推荐 = Chat
用户输入: "推荐关于自动驾驶的论文" -> {{"intent": "paper", "query": "自动驾驶", "max_results": 3}} <-- 明确找论文 = Paper
用户输入: "帮我找找Transformer相关的文献" -> {{"intent": "paper", "query": "Transformer", "max_results": 3}}
用户输入: "什么是强化学习" -> {{"intent": "chat", "query": "什么是强化学习"}}

### 当前任务
用户输入: "{user_query}"
JSON Output:
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "你是意图分析助手，只输出合法 JSON，不要 Markdown。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=200,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"intent": "paper", "query": user_query, "max_results": DEFAULT_MAX_RESULTS, "year": None, "wants_latest": False, "wants_refresh": False}


def chat_answer(user_query: str, user_id: Optional[str] = None) -> str:
    history: List[Tuple[str, str]] = []
    if user_id:
        history = CHAT_SESSIONS.get(user_id, [])

    try:
        client = _get_chat_client()
        messages = [
            {
                "role": "system",
                "content": "你是一个中文问答助手，回答简洁、准确，不编造事实。不要使用 Markdown，不要输出列表符号或星号，直接用自然语言短句回答。",
            }
        ]
        for q, a in history[-CHAT_HISTORY_MAX_TURNS:]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_query})

        resp = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
            timeout=10,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        is_timeout = "Timeout" in e.__class__.__name__ or "timeout" in str(e).lower()
        if is_timeout:
            logger.warning("chat_answer timeout: %s", e)
        else:
            logger.exception("chat_answer failed: %s", e)
        answer = ""

    if user_id and answer:
        history = CHAT_SESSIONS.get(user_id, [])
        history.append((user_query, answer))
        if len(history) > CHAT_HISTORY_MAX_TURNS:
            history = history[-CHAT_HISTORY_MAX_TURNS :]
        CHAT_SESSIONS[user_id] = history

    return answer or _offline_chat_fallback(user_query)


def _offline_chat_fallback(query: str) -> str:
    brief = _shorten(query.strip(), 60) if query else ""
    if not brief:
        return "我在的，有什么想聊的可以直接告诉我。"
    return f"我在的，先给你一个简单回答：关于“{brief}”，可以再具体说明下需求吗？"


class RefreshRequest(BaseModel):
    domains: Optional[List[str]] = None
    max_fetch: int = 100


class RecommendRequest(BaseModel):
    query: str
    max_results: int = 3


@app.on_event("startup")
async def startup_event():
    ensure_initialized()


@app.post("/")
async def index(request: Request, background_tasks: BackgroundTasks):
    """
    公众号接入后的入口（仅 POST）：
    - 未带内容时：返回 1/2 指令提示。
    - 带 content/type 时：按指令处理（同 /api/chat）。
    """
    content = None
    msg_type = "text"
    from_user = None

    # 1) 尝试表单
    try:
        form = await request.form()
        if form:
            content = form.get("content") or content
            msg_type = form.get("type") or msg_type
            from_user = form.get("from_user") or from_user
    except Exception:
        pass

    # 2) 尝试 JSON
    if content is None:
        try:
            payload = await request.json()
            if isinstance(payload, dict):
                content = payload.get("content") or content
                msg_type = payload.get("type") or msg_type
                from_user = payload.get("from_user") or from_user
        except Exception:
            pass

    # 3) 尝试 query 参数（用于调试）
    if content is None:
        params = request.query_params
        content = params.get("content")
        msg_type = params.get("type") or msg_type
        from_user = params.get("from_user") or from_user

    if content is not None:
        reply = build_reply(content, msg_type, user_id=from_user, background_tasks=background_tasks)
        return PlainTextResponse(_truncate(reply), media_type="text/plain; charset=utf-8")

    return PlainTextResponse(_truncate(_build_welcome_message()), media_type="text/plain; charset=utf-8")


@app.get("/health")
async def health():
    kb_obj, _ = ensure_initialized()
    return {"status": "ok", "paper_count": kb_obj.count()}


@app.post("/refresh")
async def refresh(req: RefreshRequest):
    domains = req.domains or fetcher.get_available_domains()
    if not domains:
        raise HTTPException(status_code=400, detail="domains list is empty")

    try:
        inserted, fetched = _refresh_knowledge_base(domains=domains, target_count=req.max_fetch)
    except ValueError:
        raise HTTPException(status_code=400, detail="domains list is empty")
    return {"requested": req.max_fetch, "fetched": fetched, "inserted": inserted, "domains": domains}


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    _, agent_obj = ensure_initialized()
    result = agent_obj.recommend_papers(req.query, max_papers=req.max_results)
    return result


# -------------------------
#  Minimal Agent-compatible endpoint
# -------------------------
def _truncate(text: str, limit: int = 2000) -> str:
    """微信单条消息最大 2048 字符，这里预防性截断到 2000。"""
    return text[:limit] if text else ""


def _shorten(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    return text[:limit]


def run_refresh_and_notify(openid: str):
    """后台执行刷新，并在完成后通过客服接口通知用户。最多等待约 2 分钟。"""
    start = time.monotonic()
    try:
        inserted, _fetched = _refresh_knowledge_base()
        duration = time.monotonic() - start
        if duration > 120:
            send_custom_message(openid, f"更新完成但耗时较长（{int(duration)}秒），本次新增 {inserted} 篇论文。")
        else:
            send_custom_message(openid, f"论文更新完成，本次新增 {inserted} 篇论文。")
    except ValueError:
        send_custom_message(openid, "更新失败：未配置可用的领域。")
    except Exception as e:
        logger.exception("run_refresh_and_notify failed: %s", e)
        send_custom_message(openid, "更新时出现问题，请稍后重试。")


def run_recommend_and_notify(openid: str, query: str, year_pref: Optional[int] = None, max_results: int = DEFAULT_MAX_RESULTS):
    """后台执行论文推荐并推送结果，避免前台超时。"""
    start = time.monotonic()
    try:
        kb_obj, agent_obj = ensure_initialized()
        if kb_obj.count() == 0:
            send_custom_message(openid, "当前本地库暂无论文，已触发更新，请稍后再试。")
            try:
                _refresh_knowledge_base()
            except Exception as refresh_exc:
                logger.warning("background refresh after empty kb failed: %s", refresh_exc)
            return
        if year_pref is None:
            year_pref = _extract_year_pref(query)
        recs, used_query = recommend_with_fallback(agent_obj, query, max_papers=max_results, year_pref=year_pref)
        if not recs and _should_refresh_for_year(year_pref):
            try:
                _refresh_knowledge_base()
            except Exception as refresh_exc:
                logger.warning("refresh after empty recommend failed: %s", refresh_exc)
            # 再试一次
            recs, used_query = recommend_with_fallback(agent_obj, query, max_papers=max_results, year_pref=year_pref)
        if not recs:
            recs = _fallback_kb_recommend(kb_obj, query, max_papers=max_results, year_pref=year_pref)
            used_query = query
        message = _format_recommendations(used_query, recs) if recs else f"没有找到与“{query}”相关的论文，已补库，可稍后再试。"
        resp = send_custom_message(openid, message)
        if isinstance(resp, dict) and resp.get("errcode") not in (0, None):
            logger.warning("recommend send failed, resp=%s, trying fallback short message", resp)
            # 简短兜底消息，避免过长或编码问题
            short_lines = []
            for rec in recs[:5]:
                title = rec.get("title") or "未提供标题"
                translated = _translate_title(title)
                display_title = f"{title}（{translated}）" if translated else title
                link = rec.get("link") or ""
                short_lines.append(f"- {display_title} | {link}")
            fallback_msg = "\n".join(short_lines) if short_lines else "已生成推荐结果，但推送失败，请稍后再试。"
            send_custom_message(openid, fallback_msg)
        duration = time.monotonic() - start
        logger.info("recommend_and_notify done in %.2fs, query=%s, used_query=%s, results=%d", duration, query, used_query, len(recs))
    except Exception as e:
        logger.exception("run_recommend_and_notify failed: %s", e)
        send_custom_message(openid, "检索时出现问题，请稍后重试。")


def _pick_recommendations(agent_res: Dict[str, Any], query: str, max_papers: int = 3) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []

    if agent_res:
        json_part = agent_res.get("json")
        if isinstance(json_part, dict):
            items = json_part.get("recommendations") or json_part.get("papers") or []
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    recs.append(
                        {
                            "title": item.get("title") or item.get("paper_title") or "",
                            "link": item.get("link") or item.get("url") or "",
                            "reason": item.get("reason") or item.get("summary") or "",
                            "published": item.get("published") or item.get("date") or item.get("published_date") or "",
                        }
                    )
                    if len(recs) >= max_papers:
                        break

    if not recs:
        structured = agent_res.get("structured") if agent_res else []
        if isinstance(structured, list):
            for item in structured:
                if not isinstance(item, dict):
                    continue
                recs.append(
                    {
                        "title": item.get("title") or "",
                        "link": item.get("link") or "",
                        "reason": f"与你的检索“{query}”相关",
                        "published": item.get("published") or "",
                    }
                )
                if len(recs) >= max_papers:
                    break

    # 根据链接补充发布时间
    for rec in recs:
        link = rec.get("link") or ""
        rec["published"] = _infer_published_from_link(link, rec.get("published") or "")

    # 按发布时间降序优先返回最新
    def _rec_sort_key(rec: Dict[str, str]) -> float:
        ts = _parse_date_to_ts(rec.get("published"))
        return ts if ts is not None else float("-inf")

    recs.sort(key=_rec_sort_key, reverse=True)
    return recs


def _format_recommendations(query: str, recs: List[Dict[str, str]]) -> str:
    if not recs:
        return f"没有找到与“{query}”相关的论文。"

    topic = _extract_topic_for_display(query)
    lines = [f"为你找到 {len(recs)} 篇与“{topic}”相关的论文："]
    for idx, rec in enumerate(recs, 1):
        title = rec.get("title") or "未提供标题"
        link = rec.get("link") or "无链接"
        reason = rec.get("reason") or "相关性较高"
        published = _infer_published_from_link(link, rec.get("published") or "")

        display_title = title
        translated = _translate_title(title)
        if translated:
            display_title = f"{title}（{translated}）"

        lines.append(f"{idx}. {display_title}")
        lines.append(f"日期：{published}")
        lines.append(f"链接：{link}")
        lines.append(f"理由：{reason}")
    return "\n".join(lines)

def run_chat_and_notify(openid: str, query: str):
    """后台执行聊天生成，并通过客服消息推送"""
    # 这里的 chat_answer 会负责维护 history
    answer = chat_answer(query, user_id=openid)
    if answer:
        send_custom_message(openid, answer)
def handle_text_command(content: str, user_id: Optional[str] = None, background_tasks: Optional[BackgroundTasks] = None) -> str:
    """
    处理公众号消息：支持更新、论文推荐和自然语言问答。
    包含防重逻辑，防止微信 5s 超时重试导致重复回复。
    """
    text = (content or "").strip()
    if not text:
        return _build_welcome_message()

    # 🛑 0. 防重逻辑 (Deduplication) 🛑
    # 微信超时重试通常在 5s 和 15s 左右，我们设置 15s 窗口
    if user_id:
        try:
            current_ts = time.time()
            # 简单清理：如果缓存太大，清空一次（防止内存泄漏）
            if len(REQUEST_ID_CACHE) > 5000:
                REQUEST_ID_CACHE.clear()
            
            # 生成唯一 Key：用户ID + 文本内容
            req_key = f"{user_id}:{text}"
            last_ts = REQUEST_ID_CACHE.get(req_key)
            
            # 如果 15 秒内收到完全一样的内容，判定为微信重试，直接忽略
            if last_ts and (current_ts - last_ts < 15.0):
                logger.info(f"Duplicate request detected from {user_id}: {text[:10]}... (Ignored)")
                # 直接返回空串，微信收到后不会报错，也不会再重试
                return ""
            
            # 记录本次请求时间
            REQUEST_ID_CACHE[req_key] = current_ts
        except Exception as e:
            logger.warning("Deduplication check failed: %s", e)

    # ---------------- 以下是原有业务逻辑 ----------------

    # 1. 首次接入自检 / 帮助指令
    if text == "5":
        welcome = _build_welcome_message()
        return (
            f"{welcome}\n"
            "我可以：\n"
            "- 论文检索与推荐：直接输入关键词即可返回推荐。\n"
            "- 自动刷新：根据新鲜度阈值自动补库，需要手动时可输入 1 强制更新。\n"
            "- 自然语言问答：直接提问即可获取答案。"
        )

    # 2. 指令 "1": 强制刷新
    if text.startswith("1"):
        if user_id and background_tasks is not None:
            background_tasks.add_task(run_refresh_and_notify, user_id)
            return "正在检索并更新最新论文，预计 2 分钟内完成；完成后会通知你。你也可以直接输入关键词获取推荐。"
        # 无用户信息时，降级为同步更新
        try:
            domains = fetcher.get_available_domains()
            if not domains:
                return "更新失败：未配置可用的领域。"
            inserted, _fetched = _refresh_knowledge_base(domains=domains)
            return f"本地数据库已更新，本次新增 {inserted} 篇论文。"
        except Exception as e:
            logger.exception("handle_text_command update failed: %s", e)
            return "更新时出现问题，请稍后重试。"

    # 3. 指令 "2": 强制推荐 (跳过意图识别)
    if text.startswith("2"):
        query = text[1:].lstrip(" ：:").strip()
        if not query:
            return "请直接输入检索关键词，例如“图神经网络”。"

        try:
            # 如果有用户信息，走后台异步推送，避免前台超时
            if user_id and background_tasks is not None:
                year_pref = _extract_year_pref(query)
                background_tasks.add_task(run_recommend_and_notify, user_id, query, year_pref, DEFAULT_MAX_RESULTS)
                return "正在检索并生成推荐，稍后将把结果推送给你。"

            # 无法异步时，同步返回（可能较慢）
            kb_obj, agent_obj = ensure_initialized()
            if kb_obj.count() == 0:
                try:
                    _refresh_knowledge_base()
                except Exception as e:
                    logger.warning("sync refresh when kb empty failed: %s", e)
                return "当前本地库暂无论文，正在补充，请稍后再试。"
            
            year_pref = _extract_year_pref(query)
            recs, used_query = recommend_with_fallback(agent_obj, query, max_papers=DEFAULT_MAX_RESULTS, year_pref=year_pref)
            
            if not recs and _should_refresh_for_year(year_pref):
                try:
                    _refresh_knowledge_base()
                except Exception as e:
                    logger.warning("sync refresh after empty recs failed: %s", e)
                recs = _fallback_kb_recommend(kb_obj, query, max_papers=DEFAULT_MAX_RESULTS, year_pref=year_pref)
                used_query = query
            
            return _format_recommendations(used_query, recs) if recs else f"没有找到与“{query}”相关的论文。"
        except Exception as e:
            logger.exception("handle_text_command recommend failed: %s", e)
            return "检索时出现问题，请稍后重试。"

    # 4. 智能意图识别
    try:
        intent_res = analyze_intent(text)
    except Exception as e:
        logger.exception("handle_text_command intent analyze failed: %s", e)
        # 如果 LLM 挂了，兜底回一般聊天
        return chat_answer(text, user_id=user_id) or _offline_chat_fallback(text)

    # 5. 解析 LLM 结果
    intent = (intent_res.get("intent") or "chat").lower() if isinstance(intent_res, dict) else "chat"
    
    # 提取参数
    extracted_query = intent_res.get("query") if isinstance(intent_res, dict) else None
    extracted_query = extracted_query or text
    
    max_results = intent_res.get("max_results") if isinstance(intent_res, dict) else DEFAULT_MAX_RESULTS
    try:
        max_results = max(1, min(int(max_results), 20))
    except Exception:
        max_results = DEFAULT_MAX_RESULTS
        
    year_pref = intent_res.get("year") if isinstance(intent_res, dict) else None
    if year_pref is None:
        year_pref = _extract_year_pref(extracted_query)
        
    wants_latest = bool(intent_res.get("wants_latest")) if isinstance(intent_res, dict) else False
    wants_refresh = bool(intent_res.get("wants_refresh")) if isinstance(intent_res, dict) else False

    # 打印决策日志
    logger.info("LLM Decision: intent=%s query=%s max_results=%s year=%s", intent, extracted_query, max_results, year_pref)

    # 6. 分支执行：Paper 意图
    if intent == "paper":
        if _is_generic_paper_request(text):
            return "想推荐哪些领域或主题的论文？请提供关键词，例如“机器人控制”“大模型安全”“强化学习”。"

        _save_query_context(user_id, extracted_query, year_pref)

        if user_id and background_tasks is not None:
            background_tasks.add_task(run_recommend_and_notify, user_id, extracted_query, year_pref, max_results)
            return "正在检索并生成推荐，稍后将把结果推送给你。"

        try:
            kb_obj, agent_obj = ensure_initialized()
            # 库为空的处理
            if kb_obj.count() == 0:
                if wants_latest or _should_refresh_for_year(year_pref) or wants_refresh:
                    try:
                        _refresh_knowledge_base()
                    except Exception as e:
                        logger.warning("sync refresh when kb empty (intent) failed: %s", e)
                return "当前本地库暂无论文，正在补充，请稍后再试。"

            # 检索
            recs, used_query = recommend_with_fallback(agent_obj, extracted_query, max_papers=max_results, year_pref=year_pref)
            
            # 结果为空尝试刷新
            if not recs and (wants_latest or _should_refresh_for_year(year_pref) or wants_refresh):
                try:
                    _refresh_knowledge_base()
                except Exception as e:
                    logger.warning("sync refresh after empty recs (intent) failed: %s", e)
                recs, used_query = recommend_with_fallback(agent_obj, extracted_query, max_papers=max_results, year_pref=year_pref)
            
            # 最后的兜底检索
            if not recs:
                recs = _fallback_kb_recommend(kb_obj, extracted_query, max_papers=max_results, year_pref=year_pref)
                used_query = extracted_query
            
            if recs:
                return _format_recommendations(used_query, recs)
            return f"没有找到与“{extracted_query}”相关的论文。"
        except Exception as e:
            logger.exception("Paper execution failed: %s", e)
            return "检索时出现问题，请稍后重试。"

    # 7. 分支执行：Chat 意图 (默认)
    # 包括了 "chat" 以及任何未识别的意图
    
    # 🚀 关键修改：如果有 user_id (来自微信)，走后台异步，防止 5秒 超时
    if user_id and background_tasks is not None:
        background_tasks.add_task(run_chat_and_notify, user_id, text)
        # 返回一个“正在输入”的状态，或者空字符串（微信收到空串不会报错，稍后客服消息会弹出）
        # 这里建议返回空字符串，用户体验就是：发完消息 -> 顿一下 -> 收到回复
        return "" 

    # 如果是调试模式（没有 user_id），才同步等待
    answer = chat_answer(text, user_id=user_id)
    if not answer:
        return _offline_chat_fallback(text)
    return answer


def build_reply(content: str, msg_type: str, user_id: Optional[str] = None, background_tasks: Optional[BackgroundTasks] = None) -> str:
    if msg_type == "text":
        return handle_text_command(content, user_id=user_id, background_tasks=background_tasks)
    if msg_type == "image":
        return f"收到图片，URL 是 {content}"
    return f"收到 {msg_type} 类型消息"


def _split_message(content: str, limit_bytes: int = 1200) -> List[str]:
    if not content:
        return [""]
    parts: List[str] = []
    remaining = content
    while remaining:
        # if fits, append and break
        if len(remaining.encode("utf-8")) <= limit_bytes:
            parts.append(remaining)
            break
        # walk through string until byte limit
        byte_count = 0
        cut = 0
        for idx, ch in enumerate(remaining):
            byte_count += len(ch.encode("utf-8"))
            if byte_count <= limit_bytes:
                cut = idx + 1
            else:
                break
        # try newline before cut
        newline_pos = remaining.rfind("\n", 0, cut)
        if newline_pos != -1 and newline_pos >= int(cut * 0.5):
            cut = newline_pos
        parts.append(remaining[:cut].rstrip("\n"))
        remaining = remaining[cut:].lstrip("\n")
    return parts


def send_custom_message(openid: str, content: str):
    """
    通过客服接口发送文本消息，遵守：
    - POST JSON
    - 不需要额外 header 或鉴权
    - 仅 message_type=text
    """
    url = f"{MAIN_SERVER}/send_custom_message"
    chunks = _split_message(content, limit_bytes=1800)
    last_resp: Dict[str, Any] = {}
    for idx, chunk in enumerate(chunks):
        data = {"openid": openid, "message_type": "text", "content": chunk}
        try:
            resp = requests.post(
                url,
                data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            status = resp.status_code
            try:
                parsed = resp.json()
            except Exception:
                parsed = {"errcode": -2, "errmsg": f"non-json response: {resp.text[:200]}", "status": status}
            if status != 200 or (isinstance(parsed, dict) and parsed.get("errcode") not in (0, None)):
                logger.warning("send_custom_message returned status=%s resp=%s (chunk %d/%d)", status, parsed, idx + 1, len(chunks))
            else:
                logger.info("send_custom_message ok chunk %d/%d len=%d", idx + 1, len(chunks), len(chunk))
            last_resp = parsed
        except Exception as e:
            logger.warning("send_custom_message failed on chunk %d/%d: %s", idx + 1, len(chunks), e)
            last_resp = {"errcode": -1, "errmsg": str(e)}
    return last_resp


def process_and_send(from_user: str, reply: str):
    """后台任务：发送已生成的客服消息。"""
    send_custom_message(from_user, reply)


@app.get("/message")
async def message_get_probe():
    from fastapi.responses import PlainTextResponse
    """
    探活 / 调试接口：
    - 不参与公众号正式回调
    - 仅用于浏览器 / 人工 / 运维确认 Agent 是否在线
    """
    return PlainTextResponse(
        "Agent is alive. Please POST to /message with form data.",
        media_type="text/plain; charset=utf-8"
    )
@app.post("/api/chat")  # 兼容示例路径，依旧只支持 POST 表单
async def receive_message(
    background_tasks: BackgroundTasks,
    from_user: str = Form(...),
    content: str = Form(...),
    type: str = Form(...),
):
    """
    Agent 接入规范：
    - 仅 POST，application/x-www-form-urlencoded
    - 固定参数：from_user, content, type (text|image)
    - 返回纯文本（<=2000 字符），5 秒内
    """
    if type not in ("text", "image"):
        raise HTTPException(status_code=400, detail="unsupported message type")

    reply = build_reply(content, type, user_id=from_user, background_tasks=background_tasks)

    background_tasks.add_task(process_and_send, from_user, reply)
    return _truncate(reply)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
def _save_query_context(user_id: Optional[str], query: str, year_pref: Optional[int]):
    if not user_id:
        return
    dq = QUERY_CONTEXT.get(user_id)
    if dq is None:
        dq = deque(maxlen=QUERY_CONTEXT_MAX)
        QUERY_CONTEXT[user_id] = dq
    dq.append({"query": query, "year": year_pref, "ts": time.time()})


def _get_last_query_context(user_id: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if not user_id:
        return None, None
    dq = QUERY_CONTEXT.get(user_id)
    if not dq:
        return None, None
    last = dq[-1]
    return last.get("query"), last.get("year")
