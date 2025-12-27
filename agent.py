import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class PaperRecommendationAgent:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        # 配置DeepSeek API
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

    def recommend_papers(self, user_query, max_papers=5):
        """
        基于用户查询推荐论文
        """
        # 1. 语义搜索相关论文
        search_results = self.kb.search_similar_papers(user_query, n_results=max_papers)

        if not search_results['documents']:
            return "抱歉，没有找到相关的论文。"

        # 2. 构建推荐上下文
        papers_context = self._build_papers_context(search_results)

        # 3. 调用LLM生成推荐
        recommendation = self._generate_recommendation(user_query, papers_context)

        return recommendation

    def _build_papers_context(self, search_results):
        """
        构建论文信息上下文
        """
        context = "相关论文信息：\n\n"

        for i, (doc, metadata) in enumerate(zip(
            search_results['documents'][0],
            search_results['metadatas'][0]
        )):
            context += f"{i+1}. 标题: {metadata['title']}\n"
            context += f"   作者: {metadata['authors']}\n"
            context += f"   发布时间: {metadata['published']}\n"
            context += f"   分类: {metadata['primary_category']}\n"
            context += f"   链接: {metadata['arxiv_url']}\n"
            context += f"   摘要: {doc.split('摘要: ')[1] if '摘要: ' in doc else doc}\n\n"

        return context

    def _generate_recommendation(self, user_query, papers_context):
        """
        调用LLM生成推荐回答
        """
        prompt = f"""
        你是一个专业的学术论文推荐助手。基于以下论文信息，为用户推荐最相关的论文。

        用户查询: {user_query}

        {papers_context}

        请根据用户的需求，从以上论文中推荐最相关的3-5篇，并简要说明推荐理由。
        回答格式要求：
        1. 首先简要总结检索结果
        2. 然后按相关性从高到低列出推荐论文
        3. 每篇论文包含：标题、作者、推荐理由、链接
        4. 使用友好的语气
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的学术论文推荐助手，擅长根据用户需求推荐相关的研究论文。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"生成推荐时出现错误: {str(e)}"