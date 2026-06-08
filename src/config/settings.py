"""
Enterprise RAG SaaS — 统一配置管理 (V3.0)
=========================================
所有配置项支持环境变量覆盖，遵循 12-Factor App 原则。
开发/测试/生产环境通过 .env 文件或 docker-compose.yml 自动切换。

配置分类:
  环境 & 服务 — ENV, HOST, PORT, DEBUG
  向量数据库 — MILVUS_URI, MILVUS_DIM, MILVUS_COLLECTION
  全文检索   — ES_HOST, BM25_TOP_K
  缓存层     — REDIS_URL, CACHE_ENABLED, CACHE_SIMILARITY_THRESHOLD
  LLM 推理   — LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY
  Embedding  — EMBEDDING_MODEL, EMBEDDING_USE_MOCK
  检索参数   — BM25_TOP_K, VECTOR_TOP_K, RRF_K
  安全       — JWT_SECRET, MULTI_TENANT_ENABLED
  数据库     — DATABASE_URL (SQLite 开发 / PostgreSQL 生产)
  文档处理   — CHUNK_SIZE, CHUNK_OVERLAP, MAX_UPLOAD_SIZE_MB

启动校验: __post_init__() 在应用启动时自动检查关键配置的安全性。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """
    应用全局配置 — 所有字段自动从环境变量读取，无环境变量时使用默认值
    用法: from src.config.settings import settings
    """

    # ==================== 环境标识 ====================
    ENV: str = os.getenv("ENV", "development")  # development | testing | production
    DEBUG: bool = ENV == "development"          # 开发模式自动启用 DEBUG

    # ==================== 服务监听地址 ====================
    HOST: str = "0.0.0.0"                       # 绑定所有网卡（容器内必须用 0.0.0.0）
    PORT: int = int(os.getenv("PORT", "8000"))   # 默认 8000

    # ==================== Milvus 向量数据库 ====================
    # 开发环境: 本地文件模式 ./data/milvus.db
    # 生产环境: Milvus 服务器 tcp://host:19530
    MILVUS_URI: str = os.getenv("MILVUS_URI", "./data/milvus.db")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "enterprise_docs")
    MILVUS_DIM: int = int(os.getenv("MILVUS_DIM", "512"))  # 向量维度（bge-small-v1.5 = 512）

    # ==================== Elasticsearch BM25 全文检索 ====================
    # V1 默认使用本地 rank-bm25，ES_ENABLED=true 时切换 Docker ES
    ES_HOST: str = os.getenv("ES_HOST", "http://localhost:9200")
    ES_INDEX_PREFIX: str = os.getenv("ES_INDEX_PREFIX", "chunks")
    ES_ENABLED: bool = os.getenv("ES_ENABLED", "false").lower() == "true"

    # ==================== Redis 缓存 ====================
    # Docker Compose 内: redis://redis:6379/0
    # 本地开发: redis://localhost:6379/0
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"

    # ==================== LLM 大语言模型推理 ====================
    # provider: ollama(本地CPU) | deepseek(云端API) | openai | vllm
    # model: qwen2:7b | deepseek-chat | gpt-4o-mini
    # base_url: 对 ollama 是 http://localhost:11434/v1，对云端 API 是 https://api.deepseek.com/v1
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2:7b")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "ollama")       # Ollama 不需要真实 key
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # 低温度 = 更确定
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))

    # ==================== Embedding 嵌入模型 ====================
    # 使用 ModelScope bge-small-zh-v1.5，输出 512 维向量
    # EMBEDDING_USE_MOCK=true 时使用随机向量（用于测试，不依赖模型下载）
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "AI-ModelScope/bge-small-zh-v1.5"
    )
    EMBEDDING_USE_MOCK: bool = (
        os.getenv("EMBEDDING_USE_MOCK", "false").lower() == "true"
    )

    # ==================== Reranker 重排序（可选） ====================
    # 启用后检索结果会经过 BGE/Cohere/Jina 二次精排
    RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "false").lower() == "true"

    # ==================== 语义缓存 ====================
    # 两层缓存: MD5 精确匹配 + cosine 语义相似 (threshold=0.85)
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "false").lower() == "true"
    CACHE_SIMILARITY_THRESHOLD: float = float(
        os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85")
    )

    # ==================== 检索参数（Hybrid RAG） ====================
    # BM25 + 向量 双路召回 → 各自取 top_k → RRF 融合 → 取最终 top_k
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))      # BM25 召回数量
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "20"))  # 向量召回数量
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "5"))     # 最终返回给 LLM 的数量
    RRF_K: int = int(os.getenv("RRF_K", "60"))                # RRF 融合常数 k

    # ==================== 多租户安全配置 ====================
    MULTI_TENANT_ENABLED: bool = (
        os.getenv("MULTI_TENANT_ENABLED", "false").lower() == "true"
    )
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
    JWT_SECRET_KEY: str = JWT_SECRET            # 别名，兼容不同模块
    JWT_ALGORITHM: str = "HS256"                # HMAC-SHA256
    JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

    # ==================== 数据库（PostgreSQL / SQLite） ====================
    # 开发: sqlite:///./data/enterprise.db
    # 生产: postgresql://rag:password@postgres:5432/enterprise_rag
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./data/enterprise.db"
    )

    def __post_init__(self):
        """
        启动时自动校验关键配置的安全性
        - 多租户必须设置非默认 JWT_SECRET
        - 生产环境禁止使用默认密钥
        - 生产环境建议使用 Milvus 服务器而非本地文件模式
        开发环境仅打印警告，生产环境直接抛出 ValueError 阻止启动
        """
        errors = []

        if self.MULTI_TENANT_ENABLED and self.JWT_SECRET == "dev-secret-change-in-production":
            errors.append(
                "MULTI_TENANT_ENABLED=true 但 JWT_SECRET 仍为默认值，"
                "请设置环境变量 JWT_SECRET"
            )

        if self.ENV == "production" and "change-in-production" in self.JWT_SECRET:
            errors.append(
                "生产环境必须设置 JWT_SECRET，当前使用不安全默认值"
            )

        if self.ENV == "production" and self.MILVUS_URI.startswith("./"):
            errors.append(
                "生产环境建议使用 Milvus 服务器而非本地文件模式"
            )

        if errors:
            for e in errors:
                print(f"[CONFIG ERROR] {e}")
            if self.ENV == "production":
                raise ValueError("\n".join(errors))

    # ==================== 文档处理配置 ====================
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))        # 分块大小（字符数）
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))   # 分块重叠（避免切断语义）
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))  # 最大上传文件

    # ==================== 数据目录 ====================
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")   # 持久化数据目录
    TEMP_DIR: str = os.getenv("TEMP_DIR", "./temp")   # 临时文件目录（上传处理用）


# 全局单例 — 所有模块通过 import 此对象获取统一配置
settings = Settings()
