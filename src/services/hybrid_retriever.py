"""
Enterprise RAG SaaS — 混合检索引擎 (Hybrid Retrieval, V3.0)
============================================================
本模块是整个 RAG 系统的召回核心，实现「向量语义 + BM25 全文」双路检索 + RRF 融合。

======================================================================
架构设计理念
======================================================================

为什么需要混合检索？
  - 纯向量检索 (dense): 擅长语义相似但难以精确匹配关键词（如人名、术语）
  - 纯 BM25 检索 (sparse): 擅长关键词匹配但无法理解语义变体（如"HR 政策" vs "人事制度"）
  - 混合检索: 取两者之长 → 分别召回 → RRF 无参数融合 → 效果超越单一方案

为什么用 RRF (Reciprocal Rank Fusion) 而不是加权求和？
  - 不需要 tuning 超参数（权值），k=60 是通用最优
  - 对两路召回的数量差异不敏感
  - 工业界广泛验证（Elasticsearch 8.x 原生支持）

======================================================================
BM25 引擎架构
======================================================================

两阶段设计:
  V1 (当前): 本地 rank-bm25 + jieba 分词 → 零外部依赖，适合 MVP/开发
  V2 (未来): Elasticsearch 8.x → 分布式 + 中文 ik_smart 分词 + 同义词

切换方式: 设置 ES_ENABLED=true 环境变量，引擎自动检测并切换

======================================================================
数据流
======================================================================

  用户问题
    │
    ├──→ [1. 向量召回] Embedding → Milvus search → top_k 候选
    │
    ├──→ [2. BM25 召回] jieba 分词 → BM25Okapi → top_k 候选
    │
    └──→ [3. RRF 融合] chunk_id 去重 → RRF(k=60) 加权 → 排序 → 最终 top_k
"""

from typing import Dict, List, Optional
from src.config.settings import Settings
from src.services.vector_service import VectorService
from src.services.embedding_service import EmbeddingService


class BM25Engine:
    """
    BM25 全文检索引擎

    多租户隔离策略:
      - 单租户: _local_index[key=kb_id]            仅按知识库隔离
      - 多租户: _local_index[key={tenant_id}:{kb_id}] 租户+知识库复合键
      - 这样即使不同租户有同名 kb_id，索引也不会交叉污染
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = "local"  # local | elasticsearch
        self._es_client = None
        # 本地 BM25 索引: {复合键: {"bm25": BM25Okapi实例, "chunks": [原始chunk], "tokenized": [分词结果]}}
        self._local_index = {}

        if settings.ES_ENABLED:
            self._init_es()  # 尝试连接 ES，失败自动降级

    def _index_key(self, kb_id: str, tenant_id: str = "default") -> str:
        """
        生成索引隔离键（多租户安全的核心）
        为什么用复合键而不是分别过滤？
          → BM25Okapi 是内存结构，无法做 SQL 式 filter，
          → 只能通过物理隔离（不同索引实例）确保数据不交叉
        """
        if self.settings.MULTI_TENANT_ENABLED:
            return f"{tenant_id}:{kb_id}"
        return kb_id

    def _init_es(self):
        """尝试连接 ES，失败则静默降级为 local（不影响核心功能）"""
        try:
            from elasticsearch import Elasticsearch
            self._es_client = Elasticsearch(self.settings.ES_HOST)
            if self._es_client.ping():
                self.backend = "elasticsearch"
                print(f"[BM25] Elasticsearch 已连接: {self.settings.ES_HOST}")
                self._ensure_es_index()
            else:
                print("[BM25] ES ping 失败，使用本地 BM25")
        except Exception as e:
            print(f"[BM25] ES 连接失败: {e}，使用本地 BM25")

    def _ensure_es_index(self):
        """创建 ES 索引 — 预定义 mapping 以支持中文文本检索"""
        index_name = self.settings.ES_INDEX_PREFIX
        if not self._es_client.indices.exists(index=index_name):
            self._es_client.indices.create(index=index_name, body={
                "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}}},
                "mappings": {"properties": {
                    "chunk_id":  {"type": "keyword"},     # 精确匹配
                    "content":   {"type": "text"},         # 全文检索
                    "doc_name":  {"type": "keyword"},      # 过滤条件
                    "kb_id":     {"type": "keyword"},      # 知识库隔离
                    "tenant_id": {"type": "keyword"},      # 租户隔离
                    "page_num":  {"type": "integer"},
                }},
            })
            print(f"[BM25] ES 索引 '{index_name}' 已创建")

    # ====== 公共接口 ======

    def index(self, chunks: List[Dict], doc_name: str, kb_id: str = "default", tenant_id: str = "default"):
        """索引文档块 — 自动路由到 local 或 ES 后端"""
        if self.backend == "elasticsearch":
            self._index_es(chunks, doc_name, kb_id, tenant_id)
        else:
            self._index_local(chunks, kb_id, tenant_id)

    def search(self, query: str, kb_id: str = "default", tenant_id: str = "default", size: int = 20) -> List[Dict]:
        """BM25 检索 — 自动路由到 local 或 ES 后端"""
        if self.backend == "elasticsearch":
            return self._search_es(query, kb_id, tenant_id, size)
        return self._search_local(query, kb_id, tenant_id, size)

    # ====== Local BM25 实现 ======

    def _index_local(self, chunks: List[Dict], kb_id: str, tenant_id: str = "default"):
        """
        本地 BM25 索引
        为什么用 jieba 分词？
          → rank-bm25 默认按空格分词（英语），中文需要 jieba 先分词
          → jieba 是纯 Python 实现，无需额外编译依赖
        """
        from rank_bm25 import BM25Okapi
        import jieba

        key = self._index_key(kb_id, tenant_id)
        if key not in self._local_index:
            self._local_index[key] = {"bm25": None, "chunks": [], "tokenized": []}

        store = self._local_index[key]
        for chunk in chunks:
            store["chunks"].append(chunk)
            store["tokenized"].append(list(jieba.cut(chunk["content"])))

        # 全量重建 BM25（增量写入需要重新计算 IDF，所以每次全量）
        store["bm25"] = BM25Okapi(store["tokenized"])

    def _search_local(self, query: str, kb_id: str, tenant_id: str = "default", size: int = 20) -> List[Dict]:
        """
        本地 BM25 检索
        为什么需要 token 重叠校验？
          → BM25 对单文档索引可能产生负分，导致 score>0 过滤误排除有效结果
          → 改用 token 重叠作为底线保障：至少有一个查询词出现在文档中
        """
        import jieba

        key = self._index_key(kb_id, tenant_id)
        store = self._local_index.get(key)
        if not store or not store["bm25"]:
            return []

        tokenized = list(jieba.cut(query))
        scores = store["bm25"].get_scores(tokenized)

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:size]

        results = []
        for idx, score in indexed:
            chunk = store["chunks"][idx]
            # 底线保障: 至少一个查询 token 出现在文档内容中
            if not any(t in chunk["content"] for t in tokenized):
                continue
            results.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "content": chunk["content"],
                "doc_name": chunk.get("doc_name", ""),
                "page_num": chunk.get("page_num", 0),
                "score": float(score),
                "source": "bm25_local",
            })
        return results

    # ====== Elasticsearch 实现 ======

    def _index_es(self, chunks, doc_name, kb_id, tenant_id):
        """ES bulk 索引 — 批量写入，减少网络开销"""
        from elasticsearch.helpers import bulk
        actions = [
            {"_index": self.settings.ES_INDEX_PREFIX,
             "_id": c.get("chunk_id", f"{doc_name}_{i}"),
             "_source": {"chunk_id": c.get("chunk_id", f"{doc_name}_{i}"),
                          "content": c["content"], "doc_name": doc_name,
                          "kb_id": kb_id, "tenant_id": tenant_id,
                          "page_num": c.get("page_num", 0)}}
            for i, c in enumerate(chunks)
        ]
        bulk(self._es_client, actions, refresh=True)

    def _search_es(self, query, kb_id, tenant_id, size):
        """ES 全文检索 + kb_id/tenant_id 精准过滤"""
        must = [{"match": {"content": query}}, {"term": {"kb_id": kb_id}}]
        if self.settings.MULTI_TENANT_ENABLED:
            must.append({"term": {"tenant_id": tenant_id}})

        resp = self._es_client.search(
            index=self.settings.ES_INDEX_PREFIX,
            body={"query": {"bool": {"must": must}}, "size": size},
        )
        return [{"chunk_id": h["_source"]["chunk_id"],
                 "content": h["_source"]["content"],
                 "doc_name": h["_source"].get("doc_name"),
                 "page_num": h["_source"].get("page_num", 0),
                 "score": h["_score"], "source": "bm25_es"}
                for h in resp["hits"]["hits"]]


class HybridRetriever:
    """
    混合检索器 — 向量 + BM25 → RRF 融合

    核心流程:
      1. 向量召回: Milvus 语义搜索 → VECTOR_TOP_K 候选
      2. BM25 召回: 关键词搜索 → BM25_TOP_K 候选
      3. 合并去重: chunk_id 作为唯一键
      4. RRF 融合: score(chunk) = Σ 1/(k+rank+1)，k=60
      5. 最终排序: rrf_score 降序取 top_k
    """

    def __init__(self, settings: Settings, vector_service: VectorService, embedding_service: EmbeddingService):
        self.settings = settings
        self.vector = vector_service
        self.embedding = embedding_service
        self.bm25 = BM25Engine(settings)

    def retrieve(self, query: str, kb_id: str = "default", tenant_id: str = "default", top_k: int = 5) -> Dict:
        """混合检索主入口"""
        # 1. 向量召回
        query_vec = self.embedding.encode_query(query)
        vector_hits = self.vector.search(query_vec, top_k=self.settings.VECTOR_TOP_K,
                                          kb_id=kb_id, tenant_id=tenant_id)

        # 2. BM25 召回
        bm25_hits = self.bm25.search(query, kb_id=kb_id, tenant_id=tenant_id,
                                      size=self.settings.BM25_TOP_K)

        # 3. 合并去重 — chunk_id 作为唯一键，后来者覆盖
        seen = {}
        for h in vector_hits:
            cid = h.get("chunk_id", h.get("id"))
            if cid not in seen:
                seen[cid] = h
        for h in bm25_hits:
            cid = h["chunk_id"]
            if cid not in seen:
                seen[cid] = h

        # 4. RRF 融合排序
        merged = self._rrf_fusion(
            list(seen.values()), vector_hits, bm25_hits,
            k=self.settings.RRF_K, top_k=top_k,
        )

        bm25_active = len(bm25_hits) > 0
        method = "hybrid" if bm25_active else "vector_only"

        return {
            "contexts": merged,
            "sources": [{"content": h["content"], "doc_name": h.get("doc_name", ""),
                          "page_num": h.get("page_num", 0),
                          "score": round(h.get("rrf_score", h.get("score", 0)), 4),
                          "source": h.get("source", "vector")} for h in merged],
            "method": method,
        }

    def index_to_bm25(self, chunks, doc_name, kb_id="default", tenant_id="default"):
        """索引文档块到 BM25 引擎"""
        self.bm25.index(chunks, doc_name, kb_id, tenant_id)

    def _rrf_fusion(self, candidates: List[Dict], vector_results: List[Dict],
                     bm25_results: List[Dict], k: int = 60, top_k: int = 5) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合排序算法

        公式: score(chunk) = Σ 1/(k + rank_i + 1)
          - k=60: 经典常数，使前几名的权重差异化（排名1≈1/62, 排名20≈1/81）
          - 同时对向量和 BM25 的排名贡献求和
          - 不需要归一化，不需要调参数

        为什么不是加权求和？
          - 向量 score 是内积 (0~1)，BM25 score 是无界值
          - 直接加权需要做 min-max 归一化，引入更多超参数
          - RRF 天然无参数，只用排名信息
        """
        chunk_ranks: Dict[str, float] = {}

        # 向量排名贡献
        for rank, hit in enumerate(vector_results):
            cid = hit.get("chunk_id", hit.get("id"))
            chunk_ranks[cid] = chunk_ranks.get(cid, 0) + 1.0 / (k + rank + 1)

        # BM25 排名贡献
        for rank, hit in enumerate(bm25_results):
            cid = hit["chunk_id"]
            chunk_ranks[cid] = chunk_ranks.get(cid, 0) + 1.0 / (k + rank + 1)

        # 按 RRF 分数降序
        ranked = sorted(chunk_ranks.items(), key=lambda x: x[1], reverse=True)
        chunk_map = {c.get("chunk_id", c.get("id")): c for c in candidates}

        results = []
        for cid, score in ranked[:top_k]:
            if cid in chunk_map:
                r = chunk_map[cid].copy()
                r["rrf_score"] = score
                results.append(r)
        return results
