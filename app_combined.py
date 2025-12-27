# app_combined.py
"""
一体化：arXiv 拉取 -> embedding -> Chroma 持久化 -> RAG 推荐 -> Streamlit UI
运行前请:
  pip install arxiv chromadb sentence-transformers openai streamlit python-dotenv
并在 .env 中设置:
  OPENAI_API_KEY=sk-...
  # 可选: DEEPSEEK_API_KEY=...
  # 可选: LLM_PROVIDER=openai 或 deepseek
"""
import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone

import arxiv
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from get_paper import ArxivPaperFetcher
import logging
from dateutil.relativedelta import relativedelta
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# LLM client (OpenAI-compatible)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "deepseek"
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
ARXIV_MAX_PER_CALL = int(os.getenv("ARXIV_MAX_PER_CALL", "1000"))

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
#  arXiv fetching utilities
# -------------------------
import time
from datetime import datetime, timedelta, timezone
import arxiv
MAX_RETRIES = 5  # 最大重试次数
RETRY_DELAY = 3  # 初次重试前的等待时间（秒）
MAX_RESULTS_PER_CALL = 100  # 每次抓取的最大条数（arXiv API 限制）

# -------------------------
#  Knowledge base (Chroma)
# -------------------------
class ArxivKnowledgeBase:
    def __init__(self, persist_directory=CHROMA_DB_DIR, embedding_model_name=EMBEDDING_MODEL_NAME):
        # Chroma persistent client
        os.makedirs(persist_directory, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
        except Exception:
            # older chroma may use chromadb.Client
            self.client = chromadb.Client()
        # collection
        self.collection = self.client.get_or_create_collection(name="arxiv_papers")
        # embedding model (SentenceTransformer)
        self.embedding_model = get_embedding_model()
        # keep set of ids in memory for fast dedupe (also persisted in chroma)
        self._load_existing_ids()

    def _load_existing_ids(self):
        try:
            data = self.collection.get()
            ids = data.get("ids", [])
            self.existing_ids = set(ids)
        except Exception:
            self.existing_ids = set()
        logger.info("知识库初始化，已有 %d 篇论文记录", len(self.existing_ids))

    def count(self):
        # safe count
        try:
            data = self.collection.get()
            return len(data.get("ids", []))
        except Exception:
            return len(self.existing_ids)

    def add_papers(self, papers):
        """
        papers: list of dict objects (as produced by fetch_recent_papers)
        会跳过已有 id（去重），并把新文档与 embeddings 一起插入
        """
        if not papers:
            logger.info("没有抓取到论文，直接返回 0")
            return 0

        logger.info(f"准备处理 {len(papers)} 篇论文")
        new_docs = []
        new_metas = []
        new_ids = []

        for p in papers:
            pid = p.get("id")
            if not pid:
                logger.warning(f"论文缺少 id，跳过: {p.get('title','未知标题')}")
                continue
            if pid in self.existing_ids:
                logger.debug(f"跳过已有论文: {pid}")
                continue
            abstract = p.get("abstract", "") or ""
            abstract_preview = abstract[:1000]
            doc = f"标题: {p.get('title','')}\n摘要: {abstract_preview}"
            new_docs.append(doc)
            new_metas.append({
                "title": p.get("title"),
                "authors": ", ".join(p.get("authors", [])),
                "published": p.get("published"),
                "primary_category": p.get("primary_category"),
                "pdf_url": p.get("pdf_url"),
                "arxiv_url": p.get("arxiv_url")
            })
            new_ids.append(pid)

        logger.info(f"过滤重复后，将插入 {len(new_ids)} 篇新论文")

        if not new_ids:
            logger.info("没有新论文需要添加（全部已存在）")
            return 0

        try:
            embeddings = self.embedding_model.encode(new_docs, show_progress_bar=False, convert_to_numpy=True)
            logger.info("成功生成 embeddings")
        except Exception as e:
            logger.exception("生成 embeddings 出错: %s", e)
            return 0

        try:
            self.collection.add(documents=new_docs, metadatas=new_metas, ids=new_ids, embeddings=embeddings.tolist())
            self.existing_ids.update(new_ids)
            logger.info(f"已向知识库添加 {len(new_ids)} 篇新论文")
            return len(new_ids)
        except Exception as e:
            logger.exception("向知识库添加文档出错: %s", e)
            return 0

    def query_similar(self, query, n_results=5):
        """
        查询：显式生成查询 embedding 并使用 query_embeddings 参数（更稳定）
        返回一个 dict: {'ids':[], 'documents':[], 'metadatas':[], 'distances':[]}
        """
        try:
            # 生成查询向量
            q_emb = self.embedding_model.encode([query], convert_to_numpy=True)

            # 查询 Chroma
            res = self.collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=n_results,
                include=['documents', 'metadatas', 'distances', 'data']  # 'data' 包含 id
            )

            # 提取 ids，确保安全
            ids = []
            data_list = res.get('data') or []
            for data_entry in data_list:
                if data_entry is not None and isinstance(data_entry, dict) and 'id' in data_entry:
                    ids.append(data_entry['id'])

            # 提取其他字段，确保返回值总是列表
            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []

            # 调试日志
            logger.info(f"query_similar: 查询 '{query}' 返回 {len(ids)} 条结果")
            logger.debug(f"返回的 IDs: {ids}")
            logger.debug(f"返回的文档数: {len(documents)}")

            return {
                'ids': ids,
                'documents': documents,
                'metadatas': metadatas,
                'distances': distances,
            }

        except Exception as e:
            logger.exception("query_similar 出错: %s", e)
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

# -------------------------
#  LLM recommendation agent
# -------------------------

class PaperRecommendationAgent:
    def __init__(self, kb: ArxivKnowledgeBase, model_name: str = "deepseek-chat"):
        self.kb = kb
        self.model_name = model_name

        # setup client
        if LLM_PROVIDER == "openai":
            if OpenAI is None or not OPENAI_API_KEY:
                raise RuntimeError("缺少 OpenAI SDK 或 OPENAI_API_KEY")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.base_url = None

        elif LLM_PROVIDER == "deepseek":
            if OpenAI is None or not DEEPSEEK_API_KEY:
                raise RuntimeError("缺少 DeepSeek API key")
            # DeepSeek 可以复用 OpenAI SDK，但需要指定 base_url
            self.client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1"
            )
        else:
            raise RuntimeError("未知 LLM_PROVIDER，请设置环境变量 LLM_PROVIDER=openai 或 deepseek")

    def _build_context_from_search(self, search_res):
        documents = search_res.get("documents", [])
        metadatas = search_res.get("metadatas", [])
        ctx_text = ""
        structured = []

        for i, doc_text in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}

            if not isinstance(meta, dict):
                logger.warning(f"meta 非 dict 类型: {type(meta)} -> {meta}")
                meta = {}

            title = meta.get("title", "未知标题")
            authors = ", ".join(meta.get("authors", [])) if isinstance(meta.get("authors"), list) else meta.get("authors", "")
            pdf_url = meta.get("pdf_url", "")
            arxiv_url = meta.get("arxiv_url", "")

            ctx_text += f"标题: {title}\n作者: {authors}\n链接: {arxiv_url or pdf_url}\n摘要: {doc_text}\n\n"

            structured.append({
                "标题": title,
                "作者": authors,
                "链接": arxiv_url or pdf_url,
                "摘要": doc_text,
            })

        return ctx_text, structured

    def recommend_papers(self, user_query, max_papers=5):
        """
        调用知识库 + 大模型生成论文推荐
        """
        # 查询知识库
        search_res = self.kb.query_similar(user_query, n_results=max_papers)
        if not search_res.get('documents') or not search_res['documents'][0]:
            return {"error": "未检索到相关论文", "text": "抱歉，未找到相关论文。"}

        ctx_text, structured = self._build_context_from_search(search_res)

        # 改进的提示词，更严格地要求JSON格式
        prompt = f"""
    你是一个专业学术论文推荐助手。用户查询: {user_query}

    候选论文：
    {ctx_text}

    请严格按照以下JSON格式输出推荐，最多 {max_papers} 篇：

    {{
      "recommendations": [
        {{
          "标题": "论文标题",
          "作者": "作者列表",
          "链接": "论文链接",
          "推荐理由": "推荐理由",
          "相关性分数": "分数"
        }}
      ]
    }}

    要求：
    1. 只输出JSON，不要有其他文字
    2. 确保JSON格式完全正确
    3. 相关性分数使用1-10的数值
    4. 如果候选论文不足{max_papers}篇，按实际数量输出
    """

        # 调用 LLM (DeepSeek / OpenAI)
        max_retries = 2
        last_exc = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个学术论文推荐助手，必须严格按JSON格式输出结果。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500  # 增加token数量以容纳更多推荐
                )
                text = response.choices[0].message.content.strip()

                # 改进的JSON解析策略
                parsed_result = self._robust_json_parse(text, max_papers, structured)
                if parsed_result:
                    return parsed_result

            except Exception as e:
                logger.exception("LLM 调用失败 (attempt %d): %s", attempt + 1, e)
                time.sleep(1 + attempt * 2)
                last_exc = e

        return {"error": "LLM 调用失败", "text": str(last_exc)}

    def _robust_json_parse(self, text, max_papers, structured):
        """
        健壮的JSON解析方法，处理各种可能的输出格式
        """
        import re
        import json

        # 策略1: 直接解析整个文本
        try:
            parsed = json.loads(text)
            return {"text": None, "json": parsed, "structured": structured}
        except:
            pass

        # 策略2: 提取JSON对象
        json_patterns = [
            r'\{.*\}',  # 匹配 {...}
            r'\[.*\]',  # 匹配 [...]
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # 尝试清理和修复常见的JSON问题
                    cleaned = self._clean_json_string(match)
                    parsed = json.loads(cleaned)
                    return {"text": None, "json": parsed, "structured": structured}
                except:
                    continue

        # 策略3: 如果JSON解析完全失败，尝试从文本中提取结构化的推荐
        extracted_recommendations = self._extract_recommendations_from_text(text, max_papers)
        if extracted_recommendations:
            return {
                "text": None,
                "json": {"recommendations": extracted_recommendations},
                "structured": structured
            }

        # 策略4: 返回原始文本
        return {"text": text, "json": None, "structured": structured}

    def _clean_json_string(self, json_str):
        """
        清理和修复JSON字符串中的常见问题
        """
        import re

        # 修复常见的JSON问题
        fixes = [
            # 修复未转义的特殊字符
            (r'(".*?[^\\])"(.*?":)', r'\1\\"\2'),  # 修复未转义的引号
            (r',\s*}', '}'),  # 修复尾随逗号
            (r',\s*]', ']'),  # 修复尾随逗号
            # 修复可能的中文标点问题
            (r'，"', ', "'),
            (r'：\s*"', ': "'),
            (r'，\s*"', ', "'),
        ]

        cleaned = json_str
        for pattern, replacement in fixes:
            cleaned = re.sub(pattern, replacement, cleaned)

        return cleaned

    def _extract_recommendations_from_text(self, text, max_papers):
        """
        当JSON解析失败时，从文本中提取推荐信息
        """
        import re

        recommendations = []
        lines = text.split('\n')

        current_rec = {}
        for line in lines:
            line = line.strip()

            # 匹配标题
            title_match = re.match(r'.*[Tt]itle[：:]?\s*(.+)', line) or re.match(r'.*标题[：:]?\s*(.+)', line)
            if title_match and '标题' not in current_rec:
                current_rec['标题'] = title_match.group(1).strip('"\' ')
                continue

            # 匹配作者
            author_match = re.match(r'.*[Aa]uthor[：:]?\s*(.+)', line) or re.match(r'.*作者[：:]?\s*(.+)', line)
            if author_match and '作者' not in current_rec:
                current_rec['作者'] = author_match.group(1).strip('"\' ')
                continue

            # 匹配链接
            link_match = re.match(r'.*[Ll]ink[：:]?\s*(.+)', line) or re.match(r'.*链接[：:]?\s*(.+)', line) or re.search(r'https?://[^\s]+', line)
            if link_match and '链接' not in current_rec:
                if 'http' in line:
                    current_rec['链接'] = link_match.group(0) if isinstance(link_match, re.Match) and link_match.groups() else link_match.group(0) if isinstance(link_match, re.Match) else link_match
                else:
                    current_rec['链接'] = link_match.group(1).strip('"\' ') if link_match.groups() else link_match.group(0)
                continue

            # 匹配推荐理由
            reason_match = re.match(r'.*[Rr]eason[：:]?\s*(.+)', line) or re.match(r'.*推荐理由[：:]?\s*(.+)', line)
            if reason_match and '推荐理由' not in current_rec:
                current_rec['推荐理由'] = reason_match.group(1).strip('"\' ')
                continue

            # 匹配分数
            score_match = re.match(r'.*[Ss]core[：:]?\s*([0-9.]+)', line) or re.match(r'.*分数[：:]?\s*([0-9.]+)', line) or re.match(r'.*相关性[：:]?\s*([0-9.]+)', line)
            if score_match and '相关性分数' not in current_rec:
                current_rec['相关性分数'] = score_match.group(1)
                continue

            # 如果收集完一个推荐的所有字段，保存并重置
            if len(current_rec) >= 4:  # 至少标题、作者、链接、推荐理由
                recommendations.append(current_rec.copy())
                current_rec = {}

                if len(recommendations) >= max_papers:
                    break

        # 处理最后一个推荐
        if current_rec and len(current_rec) >= 3:  # 至少标题、作者、链接
            recommendations.append(current_rec)

        return recommendations if recommendations else None

# -------------------------
#  Streamlit app
# -------------------------
st.set_page_config(page_title="ArXiv 论文推荐助手", page_icon="📚", layout="wide")

# 初始化 fetcher，使用持久化缓存
fetcher = ArxivPaperFetcher(cache_file="arxiv_fetcher_cache.json")

# Lazy init helper
def initialize_kb_and_agent():
    kb = ArxivKnowledgeBase()
    agent = PaperRecommendationAgent(kb)
    return kb, agent

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

st.title("📚 ArXiv 论文推荐助手 — 实时检索 + 推荐 (RAG)")
st.markdown("从 arXiv 拉取最新论文，嵌入到 Chroma，结合 LLM 给出推荐。")

with st.sidebar:
    st.header("配置与操作")

    # 用户设置每次抓取的论文数量
    max_fetch = st.number_input("每次抓取最大条数 (max_results)",
                               min_value=1,
                               max_value=ARXIV_MAX_PER_CALL,
                               value=100)

    # 让用户选择领域
    st.subheader("选择领域")
    available_domains = fetcher.get_available_domains()
    selected_domains = st.multiselect(
        "选择要抓取的领域:",
        options=available_domains,
        default=['CV', 'NLP', 'ML']
    )

    max_recommend = st.slider("推荐最大数量", 1, 10, 5)

    if not st.session_state['initialized']:
        if st.button("初始化知识库与 Agent"):
            with st.spinner("正在初始化（可能会较慢）..."):
                try:
                    kb, agent = initialize_kb_and_agent()
                    st.session_state['kb'] = kb
                    st.session_state['agent'] = agent
                    st.session_state['initialized'] = True
                    st.success("初始化完成")
                except Exception as e:
                    st.error(f"初始化失败: {e}")
    else:
        st.success("已初始化")
        kb = st.session_state['kb']
        st.markdown(f"知识库论文数量: **{kb.count()}**")

    st.markdown("---")
    st.markdown("手动更新知识库（从 arXiv 抓取）")

    # 显示缓存信息
#     cache_info = fetcher.get_cache_info()
#     st.markdown(f"**缓存信息**: {cache_info['cached_paper_count']} 篇已记录论文")

    if st.button("抓取并更新论文"):
        with st.spinner("从 arXiv 抓取并更新知识库..."):
            try:
                # 使用用户设置的 max_fetch 和选择的领域
                if not selected_domains:
                    st.error("请至少选择一个领域")
                else:
                    # 计算每个领域应该抓取的数量
                    papers_per_domain = max_fetch // len(selected_domains)

                    new_papers = fetcher.fetch_balanced_papers(
                        domains=selected_domains,
                        papers_per_domain=papers_per_domain,
                        target_count=max_fetch
                    )

                    added = st.session_state['kb'].add_papers(new_papers)
                    st.success(f"抓取到 {len(new_papers)} 篇候选论文，新增 {added} 篇到知识库")

                    # 更新缓存信息显示
#                     cache_info = fetcher.get_cache_info()
#                     st.info(f"更新后缓存记录: {cache_info['cached_paper_count']} 篇论文")

            except Exception as e:
                st.error(f"更新失败: {e}")

#     st.markdown("---")
#     st.subheader("缓存管理")
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         if st.button("清空抓取缓存"):
#             cleared_count = fetcher.clear_cache()
#             st.success(f"已清空抓取缓存，移除了 {cleared_count} 个论文记录")
#             # 刷新页面以更新显示
#             st.rerun()
#
#     with col2:
#         if st.button("查看缓存详情"):
#             cache_info = fetcher.get_cache_info()
#             st.info(f"""
#             **缓存详情**:
#             - 缓存文件: `{cache_info['cache_file']}`
#             - 已记录论文: {cache_info['cached_paper_count']} 篇
#             - 缓存状态: {'✅ 已启用' if cache_info['cache_exists'] else '❌ 未找到'}
#             """)

st.subheader("🔍 查询并获取推荐论文")
query = st.text_area("请输入你的研究兴趣或问题：", height=140, placeholder="例如：graph neural networks for recommender systems")

if st.button("获取推荐"):
    if not query.strip():
        st.warning("请输入查询内容")
    else:
        if not st.session_state.get('initialized'):
            with st.spinner("自动初始化中..."):
                try:
                    kb, agent = initialize_kb_and_agent()
                    st.session_state['kb'] = kb
                    st.session_state['agent'] = agent
                    st.session_state['initialized'] = True
                except Exception as e:
                    st.error(f"初始化失败: {e}")
                    st.stop()

        agent = st.session_state['agent']
        with st.spinner("检索相似论文并向模型请求推荐..."):
            result = agent.recommend_papers(query, max_papers=max_recommend)

        # 展示结果
        if result.get("error"):
            st.error(result.get("error"))

        elif result.get("json"):
            parsed = result["json"]

            # 有时返回的是列表（你现在的情况），有时是字典
            papers = parsed if isinstance(parsed, list) else parsed.get("recommendations", [])

            if not papers:
                st.warning("未找到推荐论文。")
            else:
                st.markdown("### 🎯 推荐论文列表")
                for i, rec in enumerate(papers, start=1):
                    title = rec.get("标题") or rec.get("title", "未知标题")
                    authors = rec.get("作者") or rec.get("authors", "未知作者")
                    link = rec.get("链接") or rec.get("link", "")
                    reason = rec.get("推荐理由") or rec.get("reason", "")
                    score = rec.get("相关性分数") or rec.get("score", "")

                    # 每篇论文用卡片样式渲染
                    st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9;">
          <h4>{i}. <a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h4>
          <p><b>作者：</b>{authors}</p>
          <p><b>推荐理由：</b>{reason}</p>
          <p><b>相关性分数：</b>{score}</p>
        </div>
        """, unsafe_allow_html=True)

        elif result.get("text"):
            raw_text = result.get("text", "").strip()

            import re, json

            # 尝试在整个输出中捕获 JSON 块（无论前面是否有其他文字）
            json_candidate = re.search(r'(\[.*\]|\{.*\})', raw_text, re.S)

            if json_candidate:
                try:
                    json_str = json_candidate.group(1)

                    parsed = json.loads(json_str)
                    papers = parsed if isinstance(parsed, list) else parsed.get("recommendations", [])

                    if papers:
                        st.markdown("### 🎯 推荐论文列表")
                        for i, rec in enumerate(papers, start=1):
                            title = rec.get("标题") or rec.get("title", "未知标题")
                            authors = rec.get("作者") or rec.get("authors", "未知作者")
                            link = rec.get("链接") or rec.get("link", "")
                            reason = rec.get("推荐理由") or rec.get("reason", "")
                            score = rec.get("相关性分数") or rec.get("score", "")

                            st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9;">
          <h4>{i}. <a href="{link}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title}</a></h4>
          <p><b>作者：</b>{authors}</p>
          <p><b>推荐理由：</b>{reason}</p>
          <p><b>相关性分数：</b>{score}</p>
        </div>
        """, unsafe_allow_html=True)
                    else:
                        st.warning("未检测到推荐论文内容。")
                except Exception as e:
                    st.markdown("### 模型返回（非结构化文本）")
                    st.write(raw_text)
                    st.error(f"⚠️ JSON解析失败: {e}")
            else:
                st.markdown("### 模型返回（非结构化）")
                st.write(raw_text)

        else:
            st.warning("未能获得有效结果，请检查日志或重试。")

st.markdown("----")
##st.markdown("**提示**: 系统会自动记录已抓取的论文，避免重复。如需重新抓取所有论文，请使用侧边栏的'清空抓取缓存'功能。")