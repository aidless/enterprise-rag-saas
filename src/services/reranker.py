"""
Enterprise RAG SaaS — Reranker 精排服务 (V3.0)
=================================================
对混合检索的候选结果进行二次精排，提升最终答案质量。

======================================================================
三后端可插拔架构
======================================================================

为什么需要多后端？
  BGE-Reranker-v2-m3 (自部署):
    优点: 免费、数据不出网、中文效果好
    缺点: 需 GPU 推理，首 Token ~2.4s，内存 ~2GB

  Cohere Rerank 3.5 (API):
    优点: 延迟 ~0.4s (6x 快于 BGE)、ELO 评分 1451、无需 GPU
    缺点: 网络依赖、按调用付费

  Jina Reranker v2 (API):
    优点: 开源替代、多语言支持
    缺点: 延迟高于 Cohere

后端选择策略 (RerankerService._select_backend):
  auto 模式: Cohere > Jina > BGE（按检测是否已配置 API Key 自动降级）
  显式模式: 通过 .env 中 RERANK_BACKEND=bge|cohere|jina 指定

======================================================================
去重算法
======================================================================
Jaccard 相似度去重: intersection / union > 0.8 → 判定为重复
为什么是 Jaccard 而非 cosine？
  → Jaccard 对字符级重复更敏感（适合检测近乎相同的段落）
  → cosine 可能把内容不同但语义相似的段落误判为重复
  → 这里的目标是去重，不是语义聚类
"""

import os
from typing import List, Dict
from src.config.settings import Settings


class BGEReranker:
    """
    BGE-Reranker-v2-m3 Cross-Encoder (自部署)
    延迟 ~2.4s, 需要 ~2GB GPU 显存
    适用于: 数据不出网的合规场景、离线批量评估
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self.enabled = settings.RERANK_ENABLED
        if self.enabled:
            self._init_model()

    def _init_model(self):
        """延迟加载模型 — 首次调用时才初始化，避免启动时 OOM"""
        try:
            from FlagEmbedding import FlagReranker
            # use_fp16=True: 半精度推理，显存减半，精度损失 < 1%
            self._model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
            print("[BGE Reranker] BGE-Reranker-v2-m3 加载完成")
        except Exception as e:
            print(f"[BGE Reranker] 加载失败: {e}")
            self._model = None

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Cross-Encoder 精排
        与 Bi-Encoder 不同: 输入是 (query, doc) pair，直接输出相关性分数
        精度 > Bi-Encoder + cosine，但速度慢（每个 pair 都要过模型）
        """
        if not self._model or not documents:
            return documents[:top_k]
        try:
            pairs = [[query, d["content"]] for d in documents]
            scores = self._model.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
            for i, score in enumerate(scores):
                documents[i]["rerank_score"] = float(score)
            documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return documents[:top_k]
        except Exception as e:
            print(f"[BGE Reranker] 出错: {e}")
            return documents[:top_k]  # 降级: 返回原始排序


class CohereReranker:
    """
    Cohere Rerank 3.5 API
    延迟 ~0.4s, ELO 评分 1451, 企业级 SLA
    适用于: 生产环境（有预算）、对延迟敏感的实时问答
    """

    def __init__(self, api_key: str = None, model: str = "rerank-v3.5"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY", "")
        self.model = model
        self.enabled = bool(self.api_key)
        if self.enabled:
            import cohere
            self._client = cohere.ClientV2(self.api_key)
            print(f"[Cohere Reranker] 已连接: {model}")
        else:
            self._client = None

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not self._client or not documents:
            return documents[:top_k]
        try:
            resp = self._client.rerank(
                model=self.model, query=query,
                documents=[d["content"] for d in documents],
                top_n=top_k, return_documents=True,
            )
            for r in resp.results:
                idx = r.index
                if idx < len(documents):
                    documents[idx]["rerank_score"] = float(r.relevance_score)
            documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return documents[:top_k]
        except Exception as e:
            print(f"[Cohere Reranker] 出错: {e}")
            return documents[:top_k]


class JinaReranker:
    """Jina Reranker v2 API — Cohere 的开源替代方案"""

    def __init__(self, api_key: str = None, model: str = "jina-reranker-v2-base-multilingual"):
        self.api_key = api_key or os.getenv("JINA_API_KEY", "")
        self.model = model
        self.enabled = bool(self.api_key)
        self.base_url = "https://api.jina.ai/v1/rerank"
        if self.enabled:
            print(f"[Jina Reranker] 已连接: {model}")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not self.enabled or not documents:
            return documents[:top_k]
        try:
            import requests
            resp = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "query": query,
                      "documents": [d["content"] for d in documents], "top_n": top_k},
                timeout=10,
            )
            data = resp.json()
            for r in data.get("results", []):
                idx = r.get("index", 0)
                if idx < len(documents):
                    documents[idx]["rerank_score"] = float(r.get("relevance_score", 0))
            documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return documents[:top_k]
        except Exception as e:
            print(f"[Jina Reranker] 出错: {e}")
            return documents[:top_k]


class RerankerService:
    """
    精排服务统一入口
    自动检测可用后端: Cohere(0.4s) > Jina > BGE(2.4s)
    内置 Jaccard 去重
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.RERANK_ENABLED
        self._backend = None
        self._backend_name = "none"
        if self.enabled:
            self._select_backend()

    def _select_backend(self):
        """
        后端自动选择 — 按优先级尝试已配置 API Key 的后端
        为什么 'auto' 模式优先 Cohere > Jina > BGE？
          → Cohere: 最快 (0.4s)，付费 API
          → Jina: 中等，开源替代
          → BGE: 最慢 (2.4s)，免费自部署（兜底）
        """
        backend = (self.settings.RERANK_BACKEND.lower()
                   if hasattr(self.settings, 'RERANK_BACKEND') else "auto")

        if backend == "auto":
            candidates = [("cohere", CohereReranker()), ("jina", JinaReranker())]
            for name, instance in candidates:
                if instance.enabled:
                    self._backend, self._backend_name = instance, name
                    return
            self._backend, self._backend_name = BGEReranker(self.settings), "bge"
        elif backend == "cohere":
            self._backend, self._backend_name = CohereReranker(), "cohere"
        elif backend == "jina":
            self._backend, self._backend_name = JinaReranker(), "jina"
        elif backend == "bge":
            self._backend, self._backend_name = BGEReranker(self.settings), "bge"

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """精排 + 去重"""
        if not self._backend or not documents:
            return documents[:top_k]
        documents = self._deduplicate(documents)  # 先去重
        return self._backend.rerank(query, documents, top_k)

    def _deduplicate(self, documents: List[Dict], threshold: float = 0.8) -> List[Dict]:
        """
        Jaccard 字符集去重
        原理: 两个文档的字符集交集 / 并集 > 0.8 → 判定为近似重复
        为什么不是句子级？→ 字符级更高效 (O(n*m))，对短文本(<500字符)够用
        """
        if len(documents) <= 1:
            return documents
        kept = []
        for doc in sorted(documents, key=lambda x: x.get("score", 0), reverse=True):
            is_dup = False
            doc_tokens = set(doc.get("content", ""))
            for k in kept:
                k_tokens = set(k.get("content", ""))
                if not doc_tokens or not k_tokens:
                    continue
                jaccard = len(doc_tokens & k_tokens) / len(doc_tokens | k_tokens)
                if jaccard > threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(doc)
        return kept
