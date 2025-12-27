import chromadb
from sentence_transformers import SentenceTransformer
import json

class ArxivKnowledgeBase:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="arxiv_papers")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级模型

    def add_papers(self, papers_json_file):
        """
        将论文数据添加到向量数据库
        """
        with open(papers_json_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)

        documents = []
        metadatas = []
        ids = []

        for paper in papers:
            # 组合标题和摘要作为检索内容
            content = f"标题: {paper['title']}\n摘要: {paper['abstract'][:500]}..."  # 限制长度

            documents.append(content)
            metadatas.append({
                'title': paper['title'],
                'authors': ', '.join(paper['authors']),
                'published': paper['published'],
                'primary_category': paper['primary_category'],
                'pdf_url': paper['pdf_url'],
                'arxiv_url': paper['arxiv_url']
            })
            ids.append(paper['id'])

        # 添加到向量数据库
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"成功添加 {len(papers)} 篇论文到知识库")

    def search_similar_papers(self, query, n_results=5):
        """
        语义搜索相关论文
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results

# 初始化知识库
def initialize_knowledge_base():
    kb = ArxivKnowledgeBase()

    # 如果还没有数据，先获取并添加
    if kb.collection.count() == 0:
        print("知识库为空，正在获取初始数据...")
        papers = fetch_recent_papers(max_results=200)  # 获取200篇初始数据
        save_papers_to_json(papers, "initial_papers.json")
        kb.add_papers("initial_papers.json")

    return kb