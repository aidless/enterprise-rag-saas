"""
Enterprise RAG SaaS — RAG 核心服务 (V3.0)
==========================================
整合完整的检索增强生成管线，是整个系统的大脑。

核心流程:
  用户提问
    → 0. 语义缓存检查 (MD5 + cosine 相似度)
    → 1. Embedding 向量化
    → 2. 混合检索 (BM25 全文 + 向量语义)
    → 3. RRF (Reciprocal Rank Fusion) 融合排序
    → 4. Reranker 精排 (可选: BGE/Cohere/Jina)
    → 5. 构造 Prompt + LLM 生成
    → 6. 写入缓存
    → 7. PARTIAL_READY 兜底提示

依赖关系:
  RAGService
  ├── EmbeddingService    — 文本 → 向量
  ├── VectorService       — Milvus 向量存储/检索
  ├── HybridRetriever     — BM25 + 向量双路融合
  ├── LLMService          — 大模型推理 (Ollama/DeepSeek/OpenAI)
  ├── RerankerService     — 二级精排 (可选)
  └── SemanticCache       — 两层语义缓存 (可选)
"""

import time
from typing import Dict, List, Optional

import numpy as np

from src.config.settings import Settings
from src.services.embedding_service import EmbeddingService
from src.services.vector_service import VectorService
from src.services.hybrid_retriever import HybridRetriever
from src.services.llm_service import LLMService
from src.services.cache_service import SemanticCache


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) 核心服务
    负责编排检索+生成的全链路，对外暴露 query() 和 index_document() 两个核心接口
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # ===== 子服务初始化 =====
        # 嵌入服务: 文本 → 512 维向量 (bge-small-zh-v1.5)
        self.embedding = EmbeddingService(settings)

        # 向量服务: Milvus 读写封装
        self.vector_store = VectorService(settings)

        # LLM 服务: 统一推理接口 (Tier 路由 + 云端兜底)
        self.llm = LLMService(settings)

        # ===== Reranker 精排（可选） =====
        # 启用后检索结果经过 BGE/Cohere/Jina 二次排序
        from src.services.reranker import RerankerService
        self.reranker = RerankerService(settings)

        # ===== 语义缓存（可选） =====
        # 两层缓存策略: Redis 分布式 > 内存本地
        if settings.CACHE_ENABLED:
            if settings.REDIS_ENABLED:
                from src.services.redis_cache import RedisSemanticCache
                self.cache = RedisSemanticCache(settings)  # 分布式缓存
            else:
                self.cache = SemanticCache(settings)        # 内存缓存
        else:
            self.cache = None

        # ===== 混合检索器 =====
        # BM25(全文) + 向量(语义) → RRF 融合
        self.retriever = HybridRetriever(
            settings=settings,
            vector_service=self.vector_store,
            embedding_service=self.embedding,
        )

        print(f"[RAGService] 初始化完成 | dim={self.embedding.dim} | "
              f"es={'on' if settings.ES_ENABLED else 'off'} | "
              f"cache={'on' if settings.CACHE_ENABLED else 'off'}")

    # ========================================================================
    # query() — 核心问答接口
    # ========================================================================
    def query(
        self,
        question: str,
        kb_id: str = "default",
        top_k: int = 5,
        tenant_id: str = "default",
    ) -> Dict:
        """
        知识库问答 — 完成检索+生成的完整管线

        Args:
            question:   用户自然语言问题
            kb_id:      知识库 ID
            top_k:      最终返回给 LLM 的文档片段数
            tenant_id:  租户 ID（多租户隔离）

        Returns:
            {
                "answer":            str   — LLM 生成的回答
                "sources":           list  — 检索来源 [{content, doc_name, page_num, score, source}]
                "retrieval_method":  str   — 检索方式: hybrid|vector_only|cache_hit
                "partial_hint":      str   — PARTIAL_READY 兜底提示 (仅检索无结果时)
            }

        流程:
            0. 语义缓存检查 (cache_hit 直接返回)
            1. 混合检索 (BM25 + 向量)
            2. 构造 Prompt
            3. LLM 生成
            4. 写入缓存
            5. PARTIAL_READY 兜底提示
        """

        # --- 步骤 0: 语义缓存检查 ---
        # 先对问题向量化，然后查缓存 (精确匹配 + 语义相似匹配)
        query_emb = self.embedding.encode_query(question)
        if self.cache and self._cache_enabled_for(tenant_id):
            cached = self.cache.get(question, np.array(query_emb))
            if cached:
                return {
                    "answer": cached,
                    "sources": [],
                    "retrieval_method": "cache_hit",  # 标记为缓存命中
                    "partial_hint": None,
                }

        # --- 步骤 1: 混合检索 ---
        # BM25 全文 + 向量语义 → 各自召回 → RRF(k=60)融合 → top_k
        retrieval_result = self.retriever.retrieve(
            query=question,
            kb_id=kb_id,
            tenant_id=tenant_id,
            top_k=top_k,
        )

        # --- 步骤 1.5: PARTIAL_READY 兜底 ---
        # 如果检索完全无结果，检查知识库是否有部分文档处理失败
        partial_hint = None
        if len(retrieval_result.get("contexts", [])) == 0:
            partial_hint = self._build_partial_hint(kb_id, tenant_id)

        # --- 步骤 2: 构造 RAG Prompt ---
        # 将检索到的文档片段拼接为 LLM 的上下文
        prompt = self._build_prompt(question, retrieval_result["contexts"])
        if partial_hint:
            prompt += f"\n\n【注意】以下文档的部分内容未成功索引，可能包含相关信息：\n{partial_hint}"

        # --- 步骤 3: LLM 生成 ---
        answer = self.llm.generate(prompt)

        # --- 步骤 4: 写入缓存 ---
        if self.cache and self._cache_enabled_for(tenant_id):
            self.cache.set(question, answer, np.array(query_emb))

        result = {
            "answer": answer,
            "sources": retrieval_result["sources"],
            "retrieval_method": retrieval_result["method"],
        }

        # --- 步骤 5: 附加 PARTIAL_READY 提示 ---
        # 当 LLM 表示 "无法回答" 时，自动在答案末尾附加重试提示
        if partial_hint:
            result["partial_hint"] = partial_hint
            if "无法回答" in answer or "未找到" in answer:
                result["answer"] = (
                    f"{answer}\n\n⚠️ 以下文档中包含可能相关的信息，但部分页面未能成功索引：\n{partial_hint}"
                )

        return result

    # ========================================================================
    # index_document() — 文档索引接口
    # ========================================================================
    def index_document(
        self,
        chunks: List[Dict],
        doc_name: str,
        kb_id: str = "default",
        tenant_id: str = "default",
    ) -> int:
        """
        索引文档 — 向量化 + 写入 Milvus + 写入 BM25

        Args:
            chunks:     文档分块列表 [{"content": "...", "page_num": 1}, ...]
            doc_name:   文档文件名
            kb_id:      知识库 ID
            tenant_id:  租户 ID

        Returns:
            int: 成功索引的 chunk 数量

        流程:
            1. Embedding: 所有 chunk → 向量
            2. Milvus:   向量 + 元数据写入
            3. BM25:     HybridRetriever 内部 BM25 索引更新
        """
        texts = [c["content"] for c in chunks]
        embeddings = self.embedding.encode(texts)  # 批量向量化

        # 写入 Milvus 向量数据库
        count = self.vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            doc_name=doc_name,
            kb_id=kb_id,
            tenant_id=tenant_id,
        )

        # 同步写入 BM25 全文检索索引 (tenant_id:kb_id 复合键隔离)
        self.retriever.index_to_bm25(
            chunks=chunks,
            doc_name=doc_name,
            kb_id=kb_id,
            tenant_id=tenant_id,
        )

        return count

    # ========================================================================
    # 内部辅助方法
    # ========================================================================

    def _cache_enabled_for(self, tenant_id: str) -> bool:
        """
        判断指定租户是否启用语义缓存
        Free 套餐禁用缓存，Basic+ 套餐启用
        """
        if not self.cache:
            return False
        return self.cache.enabled_for(tenant_id)

    def _build_partial_hint(self, kb_id: str, tenant_id: str) -> str | None:
        """
        检索无结果时生成 PARTIAL_READY 兜底提示
        告知用户知识库中有部分文档索引未完成，建议重试或查看原始文件
        """
        try:
            return (
                "⚠️ 该知识库中存在部分文档的内容暂未成功索引（标有 🟡）。"
                "您可以在文档详情页点击「重试失败页面」按钮尝试重新解析。"
                "或直接查看原始文档获取完整信息。"
            )
        except Exception:
            return None

    def _build_prompt(self, question: str, contexts: List[Dict]) -> str:
        """
        构造 RAG 提示词
        将检索到的文档片段按 [来源: 文档名 P页码] 格式拼接为 LLM 上下文

        Prompt 设计要点:
        1. 角色设定: 企业知识库助手
        2. 信息来源: 仅基于参考文档
        3. 防幻觉: 明确禁止编造信息
        4. 源引用: 要求标注文档名+页码
        """
        context_text = "\n\n---\n\n".join(
            f"[来源: {c.get('doc_name', 'unknown')} P{c.get('page_num', 0)}]\n{c['content']}"
            for c in contexts
        )

        return f"""你是企业知识库助手，请根据以下参考文档回答用户问题。

【参考文档】
{context_text}

【用户问题】
{question}

【回答要求】
1. 仅根据参考文档内容回答，不要编造信息
2. 如果参考文档中没有相关信息，请明确说明"根据现有资料无法回答"
3. 回答时注明信息来源（文档名 + 页码）
4. 语言简洁专业

请开始回答："""
