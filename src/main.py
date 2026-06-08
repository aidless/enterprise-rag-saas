"""
Enterprise RAG SaaS Platform — 主应用入口 (V3.0)
================================================
基于 FastAPI 的多租户企业级 RAG (Retrieval-Augmented Generation) 平台。

架构概览:
  ┌─────────────┐    ┌──────────────┐    ┌────────────┐
  │  客户端请求   │───▶│  FastAPI 网关 │───▶│  业务服务   │
  │ (HTTP/SSE)  │    │ (限流/认证)    │    │ (RAG/文档) │
  └─────────────┘    └──────────────┘    └────────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
          ┌──────────┐ ┌────────┐ ┌──────────┐
          │ PostgreSQL│ │ Redis  │ │ Milvus   │
          │ (元数据)  │ │ (缓存) │ │  (向量)  │
          └──────────┘ └────────┘ └──────────┘

路由结构:
  /health                     — 系统健康检查（含 Ollama/Milvus 下游探测）
  /metrics                    — Prometheus 指标端点
  /ready                      — K8s readiness probe
  /api/v1/health              — API 版本健康检查
  /api/v1/kbs/{kb_id}/query   — 知识库问答（标准模式，JSON 返回）
  /api/v1/kbs/{kb_id}/query/stream — 知识库问答（SSE 流式）
  /api/v1/kbs/{kb_id}/documents/upload — 文档上传+处理
  /api/v1/kbs/{kb_id}/documents       — 文档列表
  /api/v1/kbs/{kb_id}/cache/stats     — 语义缓存统计
  /api/v1/tenants/auth/token          — JWT 认证
  /api/v1/tenants/{id}/usage          — 租户用量
  /api/v1/feedback                     — 用户反馈
  /api/v1/shadow/stats                 — 影子模式统计(V3)
  /api/v1/experiments/                 — A/B 实验平台(V3)

版本历史:
  V1: 功能闭环 — 混合检索 + 多租户 + 文档处理 + 可观测性
  V2: 生产升级 — PostgreSQL + Redis + Celery + SSE + Reranker
  V3: 智能自治 — 影子模式 + RAGAS + A/B实验 + DSPy优化
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings

# ============================================================================
# Prometheus 指标定义
# 用于 Grafana 仪表盘监控，所有指标均可通过 /metrics 端点抓取
# ============================================================================
from prometheus_client import Counter, Histogram, Gauge

# 请求计数器 — 按 method(方法)、endpoint(路径)、status(状态码) 分组
REQUEST_COUNT = Counter(
    "rag_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

# 请求延迟直方图 — P50/P95/P99 分位数
REQUEST_LATENCY = Histogram(
    "rag_api_request_latency_seconds",
    "API request latency",
    ["method", "endpoint"],
)

# RAG 端到端延迟 — 从检索到 LLM 生成的完整耗时
SEARCH_LATENCY = Histogram(
    "rag_search_latency_seconds",
    "RAG search latency (retrieval + generation)",
)

# 文档索引计数器 — 累计已处理的文档数量
DOCS_INDEXED = Counter(
    "rag_docs_indexed_total",
    "Total documents indexed",
)

# 缓存命中率仪表 — 实时监控语义缓存效率
CACHE_HIT_RATE = Gauge(
    "rag_cache_hit_rate",
    "Semantic cache hit rate",
)


# ============================================================================
# RAG 服务 — 懒加载单例
# 仅在首次请求时初始化 Milvus/BM25 连接，避免启动阻塞
# ============================================================================
_rag_service = None


def get_rag_service():
    """
    获取 RAG 服务单例（懒加载）
    首次调用时创建 RAGService 实例，后续复用。
    避免每次请求都重建 Milvus 连接。
    """
    global _rag_service
    if _rag_service is None:
        from src.services.rag_service import RAGService
        _rag_service = RAGService(settings)
    return _rag_service


# ============================================================================
# 应用生命周期管理
# 启动时初始化数据库表，关闭时优雅释放 Milvus 连接
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期上下文管理器
    - 启动阶段: 打印环境信息 + 创建数据库表
    - 关闭阶段: 释放 Milvus 连接
    """
    # === 启动日志 ===
    print(f"[启动] Enterprise RAG API v1 | ENV={settings.ENV}")
    print(f"[启动] DB: {settings.DATABASE_URL.split('://')[0]}")
    print(f"[启动] LLM: {settings.LLM_PROVIDER}/{settings.LLM_MODEL}")
    print(f"[启动] ES: {'enabled' if settings.ES_ENABLED else 'disabled'}")
    print(f"[启动] Cache: {'enabled' if settings.CACHE_ENABLED else 'disabled'}")
    print(f"[启动] Multi-tenant: {'enabled' if settings.MULTI_TENANT_ENABLED else 'disabled'}")

    # === 数据库初始化 ===
    from src.models.schema import init_db
    import os as _os
    _os.makedirs("./data", exist_ok=True)  # 确保持久化目录存在
    _db_engine = init_db(settings.DATABASE_URL)
    app.state.db_engine = _db_engine  # 挂载到 app.state 供下游使用
    print(f"[启动] 数据库表已就绪")

    # --- 服务运行中 ---
    yield

    # === 优雅关闭 ===
    print("[关闭] 释放 Milvus 连接...")
    try:
        rag = get_rag_service()
        rag.vector_store.close()
    except Exception:
        pass
    print("[关闭] Enterprise RAG API stopped")


# ============================================================================
# FastAPI 应用实例
# ============================================================================
app = FastAPI(
    title="Enterprise RAG API",
    version="1.0.0",
    docs_url="/api/docs",           # Swagger UI 地址
    redoc_url="/api/redoc",         # ReDoc 文档地址
    openapi_url="/api/openapi.json",  # OpenAPI schema
    lifespan=lifespan,
)


# ============================================================================
# 中间件注册（按顺序执行）
# ============================================================================

# 1. CORS 跨域 — 允许所有来源（生产环境需限制 CORS_ORIGINS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 多租户上下文 — 从 JWT/API Key 提取 tenant_id 注入 request.state
from src.middleware.tenant import TenantContextMiddleware
app.add_middleware(TenantContextMiddleware)

# 3. KB 权限中间件 — 校验用户对知识库的 owner/editor/viewer 权限
from src.middleware.permissions import KBPermissionMiddleware
app.add_middleware(KBPermissionMiddleware)

# 4. 速率限制 — Free 2qps / Pro 50qps 基于 JWT tier 字段
from src.middleware.rate_limiter import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)


# ============================================================================
# 全局 Metrics 收集中间件
# 记录每个 HTTP 请求的方法、路径、状态码和延迟
# ============================================================================
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(latency)
    return response


# ============================================================================
# 全局异常处理
# 将 Python 异常映射为结构化 JSON 错误响应
# ============================================================================
from fastapi.responses import JSONResponse as FastAPIJSONResponse


class AppError(Exception):
    """
    应用层业务异常
    用法: raise AppError("资源不存在", code="NOT_FOUND", status=404)
    """
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status: int = 500):
        self.message = message
        self.code = code
        self.status = status


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """业务异常 → 自定义状态码+code"""
    return FastAPIJSONResponse(
        status_code=exc.status,
        content={"error": exc.message, "code": exc.code},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局兜底异常处理
    将未知异常按类型映射到 HTTP 状态码:
      ConnectionError → 503  SERVICE_UNAVAILABLE
      TimeoutError    → 408  UPSTREAM_TIMEOUT
      SecurityViolation → 403  SECURITY_VIOLATION
      其他            → 500  INTERNAL_ERROR
    """
    import traceback

    error_detail = str(exc)[:500]  # 截断过长错误信息

    if isinstance(exc, ConnectionError) or "Connection" in str(type(exc).__name__):
        status, code = 503, "SERVICE_UNAVAILABLE"
    elif isinstance(exc, TimeoutError) or "Timeout" in str(type(exc).__name__):
        status, code = 408, "UPSTREAM_TIMEOUT"
    elif "SecurityViolation" in str(type(exc).__name__):
        status, code = 403, "SECURITY_VIOLATION"
    elif hasattr(exc, "status_code"):
        status = getattr(exc, "status_code", 500)
        code = f"HTTP_{status}"
    else:
        status, code = 500, "INTERNAL_ERROR"
        traceback.print_exc()

    return FastAPIJSONResponse(
        status_code=status,
        content={"error": error_detail, "code": code},
    )


# ============================================================================
# 系统级路由（根路径，不限于 API 版本）
# 独立于 /api/v1 前缀，用于基础设施监控
# ============================================================================
import httpx


@app.get("/health")
async def health():
    """
    系统健康检查 — 探测下游依赖（Ollama + Milvus）可用性
    任一依赖不可用时返回 HTTP 503 + status=degraded
    用于负载均衡器健康探测和告警
    """
    checks = {"api": "ok", "version": "1.0.0"}

    # 检查 Ollama 是否在线
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:11434/api/tags", timeout=3)
            checks["ollama"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        checks["ollama"] = "unavailable"

    # 检查 Milvus 向量数据库
    try:
        from src.config.settings import settings as _s
        from pymilvus import MilvusClient
        mc = MilvusClient(uri=_s.MILVUS_URI)
        checks["milvus"] = "ok" if mc.has_collection(_s.MILVUS_COLLECTION) else "degraded"
        mc.close()
    except Exception:
        checks["milvus"] = "unavailable"

    failed = [k for k, v in checks.items() if v == "unavailable"]
    status_code = 503 if failed else 200

    return FastAPIJSONResponse(
        content={"status": "degraded" if failed else "healthy", "checks": checks},
        status_code=status_code,
    )


@app.get("/ready")
async def ready():
    """K8s readiness probe — 应用已启动但可能尚未完全就绪时也返回 ready"""
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    """Prometheus 指标端点 — 返回文本格式的指标数据供 Prometheus 抓取"""
    from prometheus_client import generate_latest
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type="text/plain")


# ============================================================================
# API v1 路由定义
# FastAPI 的 APIRouter 自动处理 OpenAPI 文档分组
# ============================================================================
v1_router = APIRouter(prefix="/api/v1", tags=["API v1"])
health_router = APIRouter(prefix="/api/v1", tags=["Health"])
kb_router = APIRouter(prefix="/api/v1/kbs", tags=["Knowledge Bases"])


# --- 健康检查 (/api/v1/health) ---
@health_router.get("/health")
def v1_health():
    """API v1 版本健康状态"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "env": settings.ENV,
        "es_enabled": settings.ES_ENABLED,
    }


# --- 请求/响应模型 ---
from pydantic import BaseModel, Field
from typing import Optional, List


class QueryRequest(BaseModel):
    """
    知识库问答请求
    - question: 用户自然语言问题
    - top_k:    返回的最相关文档片段数 (1-20)
    - mode:     检索模式 simple(简洁)/expert(专家)
    """
    question: str = Field(..., description="用户问题", examples=["公司的核心价值观是什么？"])
    top_k: int = Field(5, description="返回结果数量", ge=1, le=20)
    mode: str = Field("simple", description="检索模式: simple(简洁) | expert(专家)")


class QueryResponse(BaseModel):
    """
    知识库问答响应
    - answer:           LLM 生成的完整回答
    - sources:          检索来源列表 [{content, doc_name, page_num, score, source}]
    - retrieval_method: 检索方式 hybrid(混合)/vector_only(纯向量)/cache_hit(缓存命中)
    - latency_ms:       端到端延迟(毫秒)
    - cache_hit:        是否命中语义缓存
    - partial_hint:     PARTIAL_READY 文档兜底提示（检索无结果时返回）
    """
    answer: str = Field(..., description="LLM 生成的回答")
    sources: List[dict] = Field(default_factory=list, description="检索来源列表")
    retrieval_method: str = Field(..., description="检索方式: hybrid | vector_only | cache_hit")
    latency_ms: float = Field(..., description="端到端延迟(毫秒)")
    cache_hit: bool = Field(False, description="是否命中语义缓存")
    partial_hint: Optional[str] = Field(None, description="PARTIAL_READY 文档兜底提示")


class CacheStatsResponse(BaseModel):
    """语义缓存统计"""
    hits: int = Field(..., description="缓存命中次数")
    misses: int = Field(..., description="缓存未命中次数")
    hit_rate: float = Field(..., description="缓存命中率 (0-1)")
    exact_entries: int = Field(..., description="精确匹配缓存条目数")
    semantic_entries: int = Field(..., description="语义缓存条目数")


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    doc_id: str = Field(..., description="文档唯一ID")
    filename: str = Field(..., description="原始文件名")
    status: str = Field(..., description="处理状态: ready | partial_ready | failed")
    total_pages: int = Field(..., description="文档总页数")
    success_pages: int = Field(..., description="成功解析页数")
    failed_pages: int = Field(..., description="失败页数")
    chunk_count: int = Field(..., description="索引块数量")


# --- 缓存统计端点 ---
@kb_router.get("/{kb_id}/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(kb_id: str):
    """
    查询知识库语义缓存命中率
    Returns: {hits, misses, hit_rate, exact_entries, semantic_entries}
    """
    rag = get_rag_service()
    if not rag.cache:
        return CacheStatsResponse(hits=0, misses=0, hit_rate=0, exact_entries=0, semantic_entries=0)
    stats = rag.cache.stats()
    CACHE_HIT_RATE.set(stats["hit_rate"])
    return CacheStatsResponse(**stats)


# --- 核心问答端点（JSON 同步返回） ---
@kb_router.post("/{kb_id}/query", response_model=QueryResponse)
async def query_kb(kb_id: str, req: QueryRequest, request: Request):
    """
    知识库问答（Hybrid RAG）— 标准同步模式
    流程: 混合检索(BM25+向量) → RRF融合 → LLM生成 → 双重校验 → 影子记录
    安全: 多租户隔离 + ResponseValidator 防止跨租户数据泄露
    """
    start = time.time()
    rag = get_rag_service()

    # 提取租户 ID — 多租户模式下从 request.state 获取（由中间件注入）
    tenant_id = (
        getattr(request.state, "tenant_id", "default")
        if settings.MULTI_TENANT_ENABLED
        else "default"
    )

    # === 核心 RAG 检索+生成 ===
    result = rag.query(
        question=req.question,
        kb_id=kb_id,
        top_k=req.top_k,
        tenant_id=tenant_id,
    )

    # === 安全校验: ResponseValidator 双重检查 ===
    # 确保返回的文档片段确实属于当前租户
    if settings.MULTI_TENANT_ENABLED:
        from src.middleware.tenant import ResponseValidator
        ResponseValidator.validate(result.get("sources", []), tenant_id)

    latency = (time.time() - start) * 1000
    SEARCH_LATENCY.observe(latency / 1000)

    # === V3: 影子模式 — 静默记录查询日志 ===
    # 不阻塞用户请求，失败不影响正常流程
    try:
        from src.services.shadow_logger import shadow_logger
        shadow_logger.log(
            tenant_id=tenant_id,
            query=req.question,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            retrieval_method=result.get("retrieval_method", ""),
            latency_ms=latency,
            cache_hit=result.get("retrieval_method") == "cache_hit",
            kb_id=kb_id,
            model_tier=getattr(request.state, "tier", "free"),
            retrieval_top_k=req.top_k,
            bm25_top_k=settings.BM25_TOP_K,
            vector_top_k=settings.VECTOR_TOP_K,
            rrf_k=settings.RRF_K,
        )
    except Exception:
        pass

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieval_method=result.get("retrieval_method", "hybrid"),
        latency_ms=round(latency, 2),
        cache_hit=result.get("retrieval_method") == "cache_hit",
    )


# ============================================================================
# V3: 影子模式统计端点
# 获取离线评估数据积累情况
# ============================================================================
v3_router = APIRouter(prefix="/api/v1/shadow", tags=["V3 Shadow"])


@v3_router.get("/stats")
async def shadow_stats():
    """影子模式数据积累统计"""
    from src.services.shadow_logger import shadow_logger
    return shadow_logger.stats()


# --- SSE 流式问答端点 ---
@app.include_router(v3_router)  # 注册 V3 路由
@kb_router.post("/{kb_id}/query/stream")
async def query_stream(kb_id: str, req: QueryRequest, request: Request):
    """
    知识库问答 — SSE (Server-Sent Events) 流式返回
    适用场景: ChatGPT 式逐字输出体验，减少首字延迟感知
    流程: 检索(非流式) → 缓存命中直接返回 → LLM 流式生成
    """
    from fastapi.responses import StreamingResponse

    tenant_id = (
        getattr(request.state, "tenant_id", "default")
        if settings.MULTI_TENANT_ENABLED
        else "default"
    )
    tier = getattr(request.state, "tier", "free")  # 租户等级影响模型选择

    rag = get_rag_service()

    # 先执行检索（非流式）
    result = rag.query(question=req.question, kb_id=kb_id, top_k=req.top_k, tenant_id=tenant_id)

    # 缓存命中 → 直接返回完整答案
    if result.get("retrieval_method") == "cache_hit":
        async def cache_stream():
            yield f"data: {result['answer']}\n\n"
        return StreamingResponse(cache_stream(), media_type="text/event-stream")

    # 构建提示词 → LLM 流式生成
    prompt = rag._build_prompt(req.question, result.get("contexts", []))

    async def sse_stream():
        """SSE 生成器 — 逐块 yield LLM 输出 + [DONE] 结束标记"""
        for chunk in rag.llm.generate_stream(prompt, tier=tier):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


# ============================================================================
# 文档管理路由
# 支持上传 → 解析 → 分块 → 索引的完整管线
# 状态机: UPLOADED → PROCESSING → CHUNKED → READY / PARTIAL_READY / FAILED
# ============================================================================
from fastapi import UploadFile, File as FastAPIFile, HTTPException
import os as _os


# DocumentService 懒加载单例
_doc_service = None


def get_doc_service():
    """获取文档服务单例"""
    global _doc_service
    if _doc_service is None:
        from src.services.document_service import DocumentService
        _doc_service = DocumentService(settings, get_rag_service())
    return _doc_service


@kb_router.post("/{kb_id}/documents/upload")
async def upload_document(kb_id: str, file: UploadFile = FastAPIFile(...), request: Request = None):
    """
    文档上传 + 索引处理
    支持格式: PDF / Markdown / TXT / CSV
    支持异步模式: ?async=true 时使用 Celery 后台任务
    处理流程: 保存临时文件 → 解析内容 → 分块 → 向量化 → 写入 Milvus
    """
    tenant_id = (
        getattr(request.state, "tenant_id", "default")
        if settings.MULTI_TENANT_ENABLED and request
        else "default"
    )

    # 保存上传文件到临时目录
    temp_dir = settings.TEMP_DIR
    _os.makedirs(temp_dir, exist_ok=True)
    temp_path = _os.path.join(temp_dir, f"upload_{uuid.uuid4().hex[:8]}_{file.filename}")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"文件大小超过 {settings.MAX_UPLOAD_SIZE_MB}MB 限制")

    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        doc_service = get_doc_service()

        # Celery 异步模式 (?async=true 时进入后台任务队列)
        use_async = request.query_params.get("async") == "true" if request else False

        if use_async:
            from src.services.tasks import submit_document_task
            task_id = submit_document_task(temp_path, file.filename, kb_id, tenant_id)
            return {"task_id": task_id, "status": "processing", "filename": file.filename}

        # 同步处理模式
        record = doc_service.upload(
            file_path=temp_path,
            filename=file.filename,
            kb_id=kb_id,
            tenant_id=tenant_id,
        )

        DOCS_INDEXED.inc()
        return {
            "doc_id": record.doc_id,
            "filename": record.filename,
            "status": record.status.value,
            "total_pages": record.total_pages,
            "success_pages": record.success_pages,
            "failed_pages": len(record.failed_pages),
            "chunk_count": record.chunk_count,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"文档处理失败: {str(e)[:200]}")
    finally:
        # 清理临时文件
        if _os.path.exists(temp_path):
            _os.remove(temp_path)


@kb_router.get("/{kb_id}/documents")
async def list_documents(kb_id: str, request: Request = None):
    """获取知识库下的所有文档列表（按租户隔离）"""
    tenant_id = (
        getattr(request.state, "tenant_id", "default")
        if settings.MULTI_TENANT_ENABLED and request
        else "default"
    )
    doc_service = get_doc_service()
    docs = doc_service.list_documents(kb_id, tenant_id)
    return {
        "kb_id": kb_id,
        "documents": [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "status": d.status.value,
                "total_pages": d.total_pages,
                "success_pages": d.success_pages,
                "failed_pages": len(d.failed_pages),
                "chunk_count": d.chunk_count,
                "created_at": d.created_at,
            }
            for d in docs
        ],
    }


@kb_router.get("/{kb_id}/documents/{doc_id}/status")
async def document_status(kb_id: str, doc_id: str):
    """查询单个文档的处理状态和详情"""
    doc_service = get_doc_service()
    record = doc_service.get_status(doc_id)
    if not record:
        raise HTTPException(404, f"文档不存在: {doc_id}")
    return {
        "doc_id": record.doc_id,
        "filename": record.filename,
        "status": record.status.value,
        "total_pages": record.total_pages,
        "success_pages": record.success_pages,
        "failed_pages": record.failed_pages,
        "chunk_count": record.chunk_count,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


@kb_router.post("/{kb_id}/documents/{doc_id}/retry")
async def retry_document(kb_id: str, doc_id: str):
    """
    重试 PARTIAL_READY 文档的失败页面
    仅当文档状态为 partial_ready 时可用
    """
    doc_service = get_doc_service()
    record = doc_service.get_status(doc_id)
    if not record:
        raise HTTPException(404, f"文档不存在: {doc_id}")

    if record.status.value != "partial_ready":
        raise HTTPException(400, f"仅 PARTIAL_READY 状态可重试，当前: {record.status.value}")

    temp_path = _os.path.join(settings.TEMP_DIR, record.filename)
    if not _os.path.exists(temp_path):
        raise HTTPException(400, "原始文件已过期，请重新上传")

    record = doc_service.retry_failed_pages(doc_id, temp_path)
    return {
        "doc_id": record.doc_id,
        "status": record.status.value,
        "success_pages": record.success_pages,
        "failed_pages": record.failed_pages,
    }


# ============================================================================
# 租户管理路由
# JWT 认证 + 全因子计费（LLM/Embedding/API/Storage）
# ============================================================================
tenant_router = APIRouter(prefix="/api/v1/tenants", tags=["Tenants"])


@tenant_router.post("/auth/token")
async def create_token(tenant_id: str, user_id: str = "admin", role: str = "admin"):
    """
    生成 JWT 认证 Token
    包含 tenant_id / user_id / role / tier 等字段
    生产环境需增加用户名密码验证
    """
    from src.services.auth_service import auth_service
    token = auth_service.create_token(tenant_id=tenant_id, user_id=user_id, role=role)
    return {"access_token": token, "token_type": "bearer"}


@tenant_router.get("/{tenant_id}/usage")
async def tenant_usage(tenant_id: str):
    """
    租户用量统计（全因子计费模型）
    维度: LLM Token / Embedding Token / Rerank 调用 / API 请求 / 存储量
    """
    from src.services.billing_service import billing_service
    return billing_service.get_tenant_usage(tenant_id)


@tenant_router.get("/usage/all")
async def all_tenants_usage():
    """所有租户用量汇总（管理视角）"""
    from src.services.billing_service import billing_service
    return {"tenants": billing_service.get_all_tenants()}


# ============================================================================
# 用户反馈路由
# 👍/👎 评分 → 写回影子模式 → 驱动质量优化
# ============================================================================
feedback_router = APIRouter(prefix="/api/v1", tags=["Feedback"])


class FeedbackRequest(BaseModel):
    """用户反馈请求"""
    question: str = Field(..., description="用户问题")
    answer: str = Field(..., description="系统回答")
    rating: str = Field(..., description="useful | not_useful")
    kb_id: str = Field("default", description="知识库ID")
    comment: str = Field("", description="用户评语")
    tenant_id: str = Field("default", description="租户ID")


@feedback_router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    提交用户反馈
    用于 RAGAS 离线评估 + Hard Negative 挖掘
    """
    from src.models.schema import init_db
    from src.services.feedback import FeedbackService
    from src.config.settings import settings

    engine = init_db(settings.DATABASE_URL)
    fb = FeedbackService(engine)
    fb.record(req.tenant_id, req.question, req.answer, req.rating, req.kb_id, req.comment)
    return {"status": "ok", "rating": req.rating}


@feedback_router.get("/feedback/stats")
async def feedback_stats(tenant_id: str = "default"):
    """反馈统计 — 展示 tenant 的 useful/not_useful 分布"""
    from src.models.schema import init_db
    from src.services.feedback import FeedbackService
    from src.config.settings import settings

    engine = init_db(settings.DATABASE_URL)
    fb = FeedbackService(engine)
    return fb.stats(tenant_id)


# ============================================================================
# V3: A/B 实验平台 API
# 支持多实验并行 + 粘性分流 + t-检验自动推广
# ============================================================================
exp_router = APIRouter(prefix="/api/v1/experiments", tags=["Experiments"])


@exp_router.get("/")
async def list_experiments():
    """列出所有实验及其状态"""
    from experiments.router import experiment_router
    return {"experiments": experiment_router.list_experiments()}


@exp_router.get("/{experiment_name}/variant")
async def get_variant(experiment_name: str, tenant_id: str = "default"):
    """
    获取租户在指定实验中的分配变体
    使用 FNV-1a hash 实现粘性分流（同一 tenant 始终分配到同一变体）
    """
    from experiments.router import experiment_router
    variant = experiment_router.get_variant(experiment_name, tenant_id)
    if not variant:
        return {"error": "Experiment not found or not running"}
    return {"experiment": experiment_name, "variant": variant.name, "params": variant.params}


@exp_router.get("/params")
async def active_params(tenant_id: str = "default"):
    """获取租户当前所有活跃实验的合并参数配置"""
    from experiments.router import experiment_router
    return {"tenant_id": tenant_id, "params": experiment_router.get_active_params(tenant_id)}


# ============================================================================
# 路由注册汇总
# 将各模块的 APIRouter 挂载到 app 上
# ============================================================================
app.include_router(health_router)
app.include_router(kb_router)
app.include_router(tenant_router)
app.include_router(feedback_router)
app.include_router(exp_router)


# ============================================================================
# 本地开发启动入口
# 生产环境使用: uvicorn src.main:app --host 0.0.0.0 --port 8000
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
