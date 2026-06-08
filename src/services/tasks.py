"""
Enterprise RAG SaaS — Celery 异步任务 (V2.0)
==============================================
将文档处理等耗时操作从 HTTP 请求中剥离，放入后台任务队列。

======================================================================
为什么需要异步？
======================================================================
问题: 同步模式下 upload() 会在 HTTP 连接中等待文档解析+分块+索引
  → 用户上传 10MB PDF → 等待 ~30s → 超时风险
  → FastAPI 默认 60s timeout，大文件处理容易触发

方案: Celery 异步任务
  → HTTP 立即返回 {"task_id": "...", "status": "processing"}
  → 后台 Worker 处理文档
  → 前端轮询 /tasks/{task_id}/status 获取进度

======================================================================
重试策略设计
======================================================================
指数退避 (Exponential Backoff):
  第 1 次失败 → 等待 60s  后重试
  第 2 次失败 → 等待 120s 后重试
  第 3 次失败 → 等待 240s 后重试 → 标记 FAILED

为什么用退避而非等间隔？
  → 如果错误是下游服务抖动（如 Milvus OOM），退避给下游恢复时间
  → 如果错误是永久性的（文件损坏），快速失败而非无限重试

======================================================================
自动重试 vs 手动重试
======================================================================
autoretry_for: IOError/TimeoutError/ConnectionError → 自动重试（临时故障）
非可重试异常 → 标记 FAILED → 用户手动重试（永久故障，如文件格式错误）
"""

import os, time, uuid
from datetime import datetime
from celery import Celery
from celery.result import AsyncResult
from src.config.settings import settings

# ============================================================================
# Celery 应用配置
# ============================================================================
app = Celery(
    "enterprise_rag",
    # Broker: 任务队列。SQLite 开发 / Redis 生产
    broker=settings.DATABASE_URL.replace("sqlite:///", "sqla+sqlite:///"),
    # Backend: 结果存储。db+ 前缀让 Celery 用 SQLAlchemy 存储结果
    backend=f"db+{settings.DATABASE_URL}",
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,

    # === 可靠性配置 ===
    task_acks_late=True,               # 任务完成后才 ACK（而非取到任务就 ACK）
    task_reject_on_worker_lost=True,   # Worker 崩溃时任务自动重新分配

    # === 超时保护 ===
    task_soft_time_limit=600,          # 软超时 10min: 抛出 SoftTimeLimitExceeded（可捕获清理）
    task_time_limit=900,               # 硬超时 15min: 直接 SIGKILL（不可恢复）

    # === 并发控制 ===
    worker_prefetch_multiplier=1,      # 每次只预取 1 个任务（公平分配，避免长任务堆积）

    # === 结果过期 ===
    result_expires=3600,               # 任务结果保留 1 小时后自动清理
)


# ============================================================================
# process_document — 文档异步处理
# ============================================================================
@app.task(
    bind=True,                          # bind=True 使 self 指向当前任务实例
    max_retries=3,
    default_retry_delay=60,             # 首次重试: 60s
    retry_backoff=True,                 # 指数退避: 60→120→240s
    retry_backoff_max=600,              # 最大退避: 600s
    autoretry_for=(IOError, TimeoutError, ConnectionError),  # 自动重试的异常类型
)
def process_document(self, doc_id: str, file_path: str, filename: str,
                     kb_id: str = "default", tenant_id: str = "default"):
    """
    异步文档处理任务

    为什么分开 try/except 两层？
      → 第一层: 可重试异常 → self.retry() → 指数退避
      → 第二层: 不可重试异常 → 直接标记 FAILED

    为什么 Exception catch 里不 self.retry()？
      → 文件解析错误是永久性故障，重试也不会成功
      → 无限重试只会浪费 Worker 资源
    """
    try:
        from src.services.document_service import DocumentService
        svc = DocumentService(settings)
        record = svc.upload_sync(
            file_path=file_path, filename=filename,
            kb_id=kb_id, tenant_id=tenant_id, doc_id=doc_id)

        return {"doc_id": doc_id, "status": record.status.value,
                "chunk_count": record.chunk_count,
                "success_pages": record.success_pages,
                "failed_pages": len(record.failed_pages)}

    except (IOError, TimeoutError, ConnectionError) as e:
        # 可重试异常 → Celery 自动退避
        raise self.retry(exc=e)

    except Exception as e:
        # 不可重试异常 → 标记失败
        from src.services.document_service import DocStatus
        try:
            from src.models.kb_members_pg import PostgresKBMemberStore
            from src.models.schema import init_db
            engine = init_db(settings.DATABASE_URL)
            store = PostgresKBMemberStore(engine)
            store.update_document_status(doc_id, status=DocStatus.FAILED.value)
        except Exception:
            pass
        return {"doc_id": doc_id, "status": "failed", "error": str(e)[:200]}


# ============================================================================
# reconciliation_task — 定时对账
# ============================================================================
@app.task
def reconciliation_task(kb_id: str = "default", mode: str = "incremental"):
    """ES-Milvus 数据一致性对账 (Celery Beat 调度)"""
    from src.services.reconciliation import ReconciliationJob
    from elasticsearch import Elasticsearch
    from src.services.vector_service import VectorService

    if not settings.ES_ENABLED:
        return {"status": "skipped", "reason": "ES disabled"}

    es = Elasticsearch(settings.ES_HOST)
    vector = VectorService(settings)
    job = ReconciliationJob(es, vector)
    return job.full_check(kb_id) if mode == "full" else job.incremental_check(kb_id)


# ============================================================================
# Celery Beat 定时调度
# ============================================================================
app.conf.beat_schedule = {
    "reconciliation-incremental": {
        "task": "src.services.tasks.reconciliation_task",
        "schedule": 900.0,               # 每 15 分钟 (900 秒)
        "kwargs": {"mode": "incremental"},
    },
}


# ============================================================================
# 任务提交辅助函数
# ============================================================================
def submit_document_task(file_path: str, filename: str,
                         kb_id: str = "default", tenant_id: str = "default") -> str:
    """提交文档处理任务 → 返回 Celery task_id 供前端轮询"""
    doc_id = str(uuid.uuid4())[:12]
    result = process_document.delay(
        doc_id=doc_id, file_path=file_path, filename=filename,
        kb_id=kb_id, tenant_id=tenant_id)
    return result.id


def get_task_status(task_id: str) -> dict:
    """查询 Celery 任务状态"""
    result = AsyncResult(task_id, app=app)
    return {"task_id": task_id, "status": result.state,
            "result": result.result if result.ready() else None}
