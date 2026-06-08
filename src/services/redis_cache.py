"""
Enterprise RAG SaaS — Redis 分布式语义缓存服务 (V2.0)
=====================================================
基于 Redis 实现多进程共享的语义缓存，支持自动降级到内存缓存。

缓存架构:
  用户提问
    → 1. MD5 精确匹配 (Redis Hash: rag:exact:{tenant_id})
    → 2. cosine 语义相似匹配 (Redis Sorted Set + 逐条计算)
    → 3. Miss → 写入两层缓存

降级策略:
  Redis 可用 → 分布式缓存（多 worker 共享）
  Redis 不可用 → 自动降级为内存 SemanticCache

Redis 数据结构设计:
  exact:{tenant_id}       → Hash    {md5_hash: answer_json}
  semantic:{tenant_id}    → Sorted Set {embedding_b64: 1.0}  (仅分数占位)
  semantic_data:{tenant_id} → Hash   {embedding_b64: answer_json}

注意事项:
  - 语义缓存在 Redis 层面需要逐条计算相似度（非向量检索）
  - 当缓存条目数 > 1000 时建议接入 Milvus 做向量索引
"""

import hashlib
import json
import base64
from typing import Optional, Dict

import numpy as np

from src.config.settings import Settings
from src.services.cache_service import SemanticCache  # V1 内存版


class RedisSemanticCache:
    """
    Redis 分布式语义缓存
    接口与 SemanticCache 完全一致，通过自动检测 Redis 可用性切换后端
    """

    # 各套餐的缓存开关策略
    TIER_CACHE_POLICY = {
        "free":       False,   # 免费套餐: 禁用缓存
        "basic":      True,    # 基础套餐: 启用
        "pro":        True,    # 专业套餐: 启用
        "enterprise": True,    # 企业套餐: 启用
    }

    def __init__(self, settings: Settings, max_size: int = 1000):
        self.settings = settings
        self.threshold = settings.CACHE_SIMILARITY_THRESHOLD  # cosine 相似阈值
        self._redis = None

        # 尝试连接 Redis
        try:
            import redis
            r = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
            r.ping()  # 验证连通性
            self._redis = r
            print(f"[RedisCache] 已连接: {settings.REDIS_URL}")
        except Exception as e:
            # Redis 不可用 → 自动降级为内存缓存
            print(f"[RedisCache] 不可用({e})，降级为内存缓存")
            self._fallback = SemanticCache(settings, max_size=max_size)

        self.hits = 0
        self.misses = 0

    # ====== Redis Key 命名规则 ======

    @property
    def is_redis_available(self) -> bool:
        return self._redis is not None

    def _key_exact(self, tenant_id: str) -> str:
        """精确匹配 Key: rag:exact:{tenant_id}"""
        return f"rag:exact:{tenant_id}"

    def _key_semantic(self, tenant_id: str) -> str:
        """语义缓存 SortedSet Key: rag:semantic:{tenant_id}"""
        return f"rag:semantic:{tenant_id}"

    def _key_semantic_data(self, tenant_id: str) -> str:
        """语义缓存数据 Key: rag:semantic_data:{tenant_id}"""
        return f"rag:semantic_data:{tenant_id}"

    # ====== 查询缓存（两层匹配） ======

    def get(self, question: str, query_embedding: np.ndarray = None, tenant_id: str = "default") -> Optional[str]:
        """
        查询缓存 — MD5 精确匹配 + cosine 语义相似匹配
        返回 None 表示未命中
        """
        # 降级模式
        if self._redis is None:
            return self._fallback.get(question, query_embedding)

        # 1. MD5 精确匹配
        md5 = hashlib.md5(question.encode()).hexdigest()
        cached = self._redis.hget(self._key_exact(tenant_id), md5)
        if cached:
            self.hits += 1
            data = json.loads(cached)
            return data["answer"]

        # 2. 语义相似匹配 (cosine 逐条计算)
        if query_embedding is not None:
            emb_b64 = self._encode_embedding(query_embedding)
            semantic_key = self._key_semantic(tenant_id)

            all_embs = self._redis.zrange(semantic_key, 0, -1)
            for cached_emb_b64 in all_embs:
                cached_emb = self._decode_embedding(cached_emb_b64)
                sim = self._cosine_similarity(query_embedding, cached_emb)
                if sim >= self.threshold:
                    cached = self._redis.hget(self._key_semantic_data(tenant_id), cached_emb_b64)
                    if cached:
                        self.hits += 1
                        return json.loads(cached)["answer"]

        self.misses += 1
        return None

    # ====== 写入缓存 ======

    def set(self, question: str, answer: str, embedding: np.ndarray = None, tenant_id: str = "default"):
        """写入两层缓存（精确 + 语义）"""
        if self._redis is None:
            self._fallback.set(question, answer, embedding)
            return

        md5 = hashlib.md5(question.encode()).hexdigest()
        data = json.dumps({"answer": answer, "question": question})

        # 精确匹配层
        self._redis.hset(self._key_exact(tenant_id), md5, data)

        # 语义缓存层
        if embedding is not None:
            emb_b64 = self._encode_embedding(embedding)
            self._redis.zadd(self._key_semantic(tenant_id), {emb_b64: 1.0})
            self._redis.hset(self._key_semantic_data(tenant_id), emb_b64, data)

    # ====== 辅助方法 ======

    def enabled_for(self, tenant_id: str, tier: str = "basic") -> bool:
        """租户套餐是否启用缓存"""
        return self.TIER_CACHE_POLICY.get(tier, False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict:
        """缓存统计信息"""
        if self._redis is None:
            return self._fallback.stats()
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "backend": "redis",
            "exact_entries": self._redis.hlen(self._key_exact("default")),
        }

    # ====== 向量编解码工具 ======

    @staticmethod
    def _encode_embedding(emb: np.ndarray) -> str:
        """float32 向量 → Base64 字符串"""
        return base64.b64encode(emb.astype(np.float32).tobytes()).decode()

    @staticmethod
    def _decode_embedding(b64: str) -> np.ndarray:
        """Base64 字符串 → float32 向量"""
        return np.frombuffer(base64.b64decode(b64), dtype=np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度 (range: -1 ~ 1)"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
