# get_paper.py
import requests
import xml.etree.ElementTree as ET
import time
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import os

class ArxivPaperFetcher:
    """
    arXiv论文获取器 - 支持按领域平衡获取论文，带有持久化缓存
    """

    # 定义领域和对应的分类
    DOMAIN_CATEGORIES = {
        'CV': ['cs.CV', 'eess.IV'],  # 计算机视觉
        'NLP': ['cs.CL', 'cs.AI'],   # 自然语言处理
        'ML': ['cs.LG', 'stat.ML'],  # 机器学习
        'AI': ['cs.AI', 'cs.NE'],    # 人工智能
        'Robotics': ['cs.RO'],       # 机器人学
        'Security': ['cs.CR']        # 安全
    }

    def __init__(self, max_retries=3, request_delay=1, timeout=30, cache_file="fetched_papers_cache.json"):
        """
        初始化论文获取器

        :param max_retries: 最大重试次数
        :param request_delay: 请求延迟（秒）
        :param timeout: 请求超时时间（秒）
        :param cache_file: 缓存文件路径，用于持久化记录已抓取的论文ID
        """
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.timeout = timeout
        self.cache_file = cache_file
        self.seen_ids = self._load_seen_ids()  # 从文件加载已见ID

    def _load_seen_ids(self):
        """从缓存文件加载已抓取的论文ID"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"从缓存文件加载了 {len(data.get('seen_ids', []))} 个已抓取论文ID")
                    return set(data.get('seen_ids', []))
            except Exception as e:
                print(f"加载缓存文件失败: {e}")
        print("无缓存文件或加载失败，使用空缓存")
        return set()

    def _save_seen_ids(self):
        """保存已抓取的论文ID到缓存文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.cache_file) if os.path.dirname(self.cache_file) else '.', exist_ok=True)

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({'seen_ids': list(self.seen_ids)}, f, ensure_ascii=False, indent=2)
            print(f"已保存 {len(self.seen_ids)} 个论文ID到缓存文件")
        except Exception as e:
            print(f"保存缓存文件失败: {e}")

    def fetch_balanced_papers(self, domains=['CV', 'NLP'], papers_per_domain=500, target_count=1000):
        """
        按领域平衡获取论文，会自动保存seen_ids
        """
        all_papers = []

        print(f"开始按领域平衡获取论文:")
        print(f"  目标领域: {', '.join(domains)}")
        print(f"  每个领域目标数量: {papers_per_domain}")
        print(f"  总目标数量: {target_count}")
        print(f"  当前缓存论文ID数量: {len(self.seen_ids)}")

        domain_stats = {}

        for domain in domains:
            if len(all_papers) >= target_count:
                break

            print(f"\n=== 处理 {domain} 领域 ===")
            categories = self.DOMAIN_CATEGORIES.get(domain, [domain])

            # 获取该领域的论文
            domain_papers = self._fetch_domain_papers(
                domain=domain,
                categories=categories,
                target_count=papers_per_domain
            )

            # 更新统计信息
            domain_stats[domain] = len(domain_papers)

            # 添加到总列表
            all_papers.extend(domain_papers)
            print(f"{domain} 领域获取 {len(domain_papers)} 篇论文，总计 {len(all_papers)} 篇")

        # 如果某些领域数量不足，尝试重新分配
        if len(all_papers) < target_count:
            print(f"\n某些领域数量不足，尝试补充...")
            all_papers = self._supplement_missing_papers(all_papers, domains, domain_stats, target_count)

        # 在方法的最后保存seen_ids
        self._save_seen_ids()

        print(f"\n✅ 最终获取 {len(all_papers)} 篇唯一论文")

        # 显示领域分布
        self._show_domain_distribution(all_papers, domains)

        return all_papers[:target_count]

    def fetch_single_domain_papers(self, domain, target_count=1000):
        """
        获取单个领域的论文
        """
        categories = self.DOMAIN_CATEGORIES.get(domain, [domain])
        papers = self._fetch_domain_papers(domain, categories, target_count)
        # 保存缓存
        self._save_seen_ids()
        return papers

    def add_domain(self, domain_name, categories):
        """
        添加新的领域定义

        :param domain_name: 领域名称
        :param categories: 对应的arXiv分类列表
        """
        self.DOMAIN_CATEGORIES[domain_name] = categories
        print(f"已添加领域 '{domain_name}': {categories}")

    def get_available_domains(self):
        """
        获取所有可用的领域

        :return: 领域列表
        """
        return list(self.DOMAIN_CATEGORIES.keys())

    def clear_cache(self):
        """清空缓存（用于重新开始收集所有论文）"""
        cache_count = len(self.seen_ids)
        self.seen_ids.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print(f"已清空抓取缓存，移除了 {cache_count} 个论文ID记录")
        return cache_count

    def get_cache_info(self):
        """获取缓存信息"""
        return {
            'cache_file': self.cache_file,
            'cached_paper_count': len(self.seen_ids),
            'cache_exists': os.path.exists(self.cache_file)
        }

    def _fetch_domain_papers(self, domain, categories, target_count):
        """
        获取特定领域的论文（内部方法）
        """
        domain_papers = []

        # 时间扩展策略
        time_periods = [1, 2, 3, 6, 12]

        for months in time_periods:
            if len(domain_papers) >= target_count:
                break

            # 计算时间范围
            end_date = datetime.now()
            start_date = end_date - relativedelta(months=months)

            # 构建查询
            categories_query = "+OR+".join([f"cat:{cat}" for cat in categories])
            date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}000000+TO+{end_date.strftime('%Y%m%d')}235959]"
            full_query = f"({categories_query})+AND+{date_query}"

            url = f"https://export.arxiv.org/api/query?search_query={full_query}&sortBy=submittedDate&sortOrder=descending&start=0&max_results={target_count * 2}"

            print(f"  获取最近{months}个月的{domain}论文...")
            papers_from_period = self._fetch_from_arxiv(url)

            # 去重并添加到领域列表
            new_papers = []
            for paper in papers_from_period:
                if paper['id'] not in self.seen_ids:
                    self.seen_ids.add(paper['id'])
                    new_papers.append(paper)
                    if len(domain_papers) + len(new_papers) >= target_count:
                        break

            domain_papers.extend(new_papers)
            print(f"  最近{months}个月获取 {len(new_papers)} 篇，领域总计 {len(domain_papers)} 篇")

            # 如果这个时间段没有新论文且已有一些论文，停止扩展
            if len(new_papers) == 0 and len(domain_papers) > 0:
                print(f"  最近{months}个月没有新{domain}论文，停止扩展时间范围")
                break

            time.sleep(self.request_delay)

        return domain_papers[:target_count]

    def _supplement_missing_papers(self, all_papers, domains, domain_stats, target_count):
        """
        补充缺失的论文数量（内部方法）
        """
        # 计算每个领域还需要多少论文
        remaining_by_domain = {}
        for domain in domains:
            current_count = domain_stats.get(domain, 0)
            expected_count = target_count // len(domains)
            remaining = expected_count - current_count
            if remaining > 0:
                remaining_by_domain[domain] = remaining

        if not remaining_by_domain:
            return all_papers

        print(f"需要补充的领域: {remaining_by_domain}")

        # 为每个需要补充的领域尝试获取更多论文
        for domain, remaining in remaining_by_domain.items():
            if len(all_papers) >= target_count:
                break

            categories = self.DOMAIN_CATEGORIES.get(domain, [domain])
            print(f"为 {domain} 领域补充 {remaining} 篇论文...")

            # 使用更宽泛的时间范围
            supplemental_papers = self._fetch_domain_papers_supplemental(
                categories=categories,
                target_count=remaining
            )

            all_papers.extend(supplemental_papers)
            print(f"{domain} 领域补充 {len(supplemental_papers)} 篇，总计 {len(all_papers)} 篇")

        # 如果仍然不足，使用关键词补充
        if len(all_papers) < target_count:
            print("使用关键词补充剩余论文...")
            all_papers = self._supplement_with_keywords(all_papers, target_count)

        return all_papers

    def _fetch_domain_papers_supplemental(self, categories, target_count):
        """
        为补充目的获取论文，使用更宽泛的查询（内部方法）
        """
        supplemental_papers = []

        # 使用更长的时间范围
        end_date = datetime.now()
        start_date = end_date - relativedelta(months=24)  # 2年

        categories_query = "+OR+".join([f"cat:{cat}" for cat in categories])
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}000000+TO+{end_date.strftime('%Y%m%d')}235959]"
        full_query = f"({categories_query})+AND+{date_query}"

        url = f"https://export.arxiv.org/api/query?search_query={full_query}&sortBy=submittedDate&sortOrder=descending&start=0&max_results={target_count * 3}"

        papers = self._fetch_from_arxiv(url)

        # 去重并添加到列表
        for paper in papers:
            if paper['id'] not in self.seen_ids:
                self.seen_ids.add(paper['id'])
                supplemental_papers.append(paper)
                if len(supplemental_papers) >= target_count:
                    break

        return supplemental_papers

    def _supplement_with_keywords(self, all_papers, target_count):
        """
        使用关键词补充论文（内部方法）
        """
        keywords = ["machine+learning", "deep+learning", "neural+network"]

        for keyword in keywords:
            if len(all_papers) >= target_count:
                break

            url = f"https://export.arxiv.org/api/query?search_query=all:{keyword}&sortBy=submittedDate&sortOrder=descending&start=0&max_results=200"
            papers = self._fetch_from_arxiv(url)

            # 去重
            new_papers = [p for p in papers if p['id'] not in self.seen_ids]

            # 添加到总列表
            for paper in new_papers:
                self.seen_ids.add(paper['id'])
                all_papers.append(paper)
                if len(all_papers) >= target_count:
                    break

            print(f"关键词 '{keyword}' 补充 {len(new_papers)} 篇，总计 {len(all_papers)} 篇")
            time.sleep(self.request_delay)

        return all_papers

    def _fetch_from_arxiv(self, url):
        """
        从arXiv API获取论文（内部方法）
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                return self._parse_atom_feed(root)

            except Exception as e:
                print(f"获取论文时出错 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.request_delay * (2 ** attempt)
                    time.sleep(wait_time)

        return []

    def _parse_atom_feed(self, root):
        """解析Atom响应（内部方法）"""
        papers = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('atom:entry', ns):
            try:
                paper_id = entry.find('atom:id', ns).text.split('/')[-1]

                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No Title"

                summary_elem = entry.find('atom:summary', ns)
                abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else "No Abstract"

                published_elem = entry.find('atom:published', ns)
                published = published_elem.text if published_elem is not None else "Unknown"

                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                categories = []
                primary_category = None
                for category in entry.findall('atom:category', ns):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                        if primary_category is None:
                            primary_category = term

                paper_info = {
                    'id': paper_id,
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'published': published,
                    'primary_category': primary_category,
                    'categories': categories,
                    'pdf_url': f"https://arxiv.org/pdf/{paper_id}.pdf",
                    'arxiv_url': f"https://arxiv.org/abs/{paper_id}"
                }

                papers.append(paper_info)

            except Exception as e:
                continue

        return papers

    def _show_domain_distribution(self, papers, domains):
        """
        显示论文的领域分布（内部方法）
        """
        domain_count = defaultdict(int)
        category_count = defaultdict(int)

        for paper in papers:
            primary_cat = paper.get('primary_category', 'Unknown')

            # 确定论文属于哪个领域
            paper_domain = 'Other'
            for domain, categories in self.DOMAIN_CATEGORIES.items():
                if primary_cat in categories:
                    paper_domain = domain
                    break

            domain_count[paper_domain] += 1
            category_count[primary_cat] += 1

        print(f"\n📊 论文领域分布:")
        for domain in domains:
            count = domain_count.get(domain, 0)
            percentage = (count / len(papers)) * 100 if papers else 0
            print(f"  {domain}: {count} 篇 ({percentage:.1f}%)")

        if domain_count.get('Other', 0) > 0:
            other_percentage = (domain_count['Other'] / len(papers)) * 100 if papers else 0
            print(f"  Other: {domain_count['Other']} 篇 ({other_percentage:.1f}%)")

        print(f"\n📊 主要分类分布:")
        for category, count in sorted(category_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(papers)) * 100 if papers else 0
            print(f"  {category}: {count} 篇 ({percentage:.1f}%)")

# 使用示例
if __name__ == "__main__":
    # 创建获取器实例
    fetcher = ArxivPaperFetcher(max_retries=3, request_delay=1)

    # 显示缓存信息
    cache_info = fetcher.get_cache_info()
    print(f"缓存文件: {cache_info['cache_file']}")
    print(f"已缓存论文数: {cache_info['cached_paper_count']}")

    # 示例1: 获取CV和NLP领域各500篇论文
    print("=" * 60)
    papers = fetcher.fetch_balanced_papers(
        domains=['CV', 'NLP'],
        papers_per_domain=500,
        target_count=1000
    )

    # 示例2: 获取单个领域的论文
    print("\n" + "=" * 60)
    cv_papers = fetcher.fetch_single_domain_papers('CV', 300)

    # 显示更新后的缓存信息
    cache_info = fetcher.get_cache_info()
    print(f"\n更新后缓存论文数: {cache_info['cached_paper_count']}")