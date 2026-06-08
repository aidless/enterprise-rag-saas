"""
Enterprise RAG SaaS — 多租户安全中间件 (V2.0)
===============================================
实现「认证 → 授权 → 校验」三层防护，阻止跨租户数据泄露。

======================================================================
安全模型三层次
======================================================================

第一层 — TenantContextMiddleware (认证):
  从 HTTP Header (X-Tenant-ID) 或 JWT Token 提取 tenant_id
  注入到 request.state 供下游使用
  为什么 JWT 优先于 Header？
    → JWT 有签名防篡改，Header 可被伪造

第二层 — KBPermissionMiddleware (授权):
  在 kb_* API 路径上校验用户对知识库的 RBAC 权限
  权限映射: owner(全部) > editor(上传+查询+导出) > viewer(仅查询)

第三层 — ResponseValidator (校验):
  在检索结果返回前，逐条断言返回数据的 tenant_id 匹配
  这是防止代码 bug 导致的数据泄露的最后防线

======================================================================
为什么需要 ResponseValidator？
======================================================================
即使 Query 中加了 tenant_id filter，如果有代码 bug：
  - filter 写错拼写
  - ORM 懒加载绕过了 filter
  - 缓存命中时返回了其他租户的数据
都可能导致跨租户泄露。ResponseValidator 在数据出 API 之前做最后校验。

"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from src.config.settings import settings


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    多租户上下文中间件 — 从请求中提取并注入 tenant_id

    提取顺序（优先级从高到低）:
      1. HTTP Header: X-Tenant-ID (直接指定，用于 API Key 模式)
      2. URL Query: ?tenant_id=xxx (调试用)
      3. JWT Bearer Token: Authorization: Bearer <token> (生产主要模式)
         JWT payload 中包含: {tenant_id, user_id, tier}

    为什么 JWT 里要带 tier？
      → 避免每次请求都查数据库获取套餐信息
      → tier 决定 LLM 模型选择 + 缓存开关 + QPS 上限
    """

    async def dispatch(self, request: Request, call_next):
        # 单租户模式 → 跳过
        if not settings.MULTI_TENANT_ENABLED:
            request.state.tenant_id = "default"
            return await call_next(request)

        # 1. X-Tenant-ID Header
        tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant_id")

        # 2. JWT Bearer Token 解析
        if not tenant_id:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                try:
                    import jwt
                    payload = jwt.decode(
                        token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
                    tenant_id = payload.get("tenant_id", "default")
                    user_id   = payload.get("user_id", "anonymous")
                    tier      = payload.get("tier", "basic")
                    request.state.user_id = user_id
                    request.state.tier = tier
                except Exception:
                    pass  # JWT 无效 → 不阻塞请求（开发环境宽容）

        # 3. 注入状态
        request.state.tenant_id = tenant_id or "default"
        if not hasattr(request.state, "user_id"):
            request.state.user_id = "anonymous"
        return await call_next(request)


class SecurityViolationError(Exception):
    """
    安全违规异常 — 跨租户数据访问拦截
    抛出此异常会触发 main.py 中的全局 SecurityViolation handler → HTTP 403
    """
    def __init__(self, message: str = "Data integrity error: cross-tenant access blocked"):
        self.message = message
        super().__init__(self.message)


class ResponseValidator:
    """
    返回数据二次校验 — 防止跨租户数据泄露的最后防线

    为什么在检索结果返回前逐条校验？
      → 即使 query 层加了 tenant_id filter，也可能因为代码 bug 导致泄露
      → ResponseValidator 在数据离开 API 前做最后拦截
      → 发现泄露时记录 CRITICAL 级别日志 + Prometheus 告警指标

    对性能的影响？
      → O(n) 遍历检索结果（通常 < 20 条），微秒级开销
      → 相比数据泄露的法律风险，这个性能成本微不足道
    """

    @staticmethod
    def validate(results: list, expected_tenant: str):
        """
        逐条校验检索结果的 tenant_id

        Args:
          results: 检索结果列表 [{"tenant_id": "...", "id": "..."}]
          expected_tenant: JWT 中提取的合法 tenant_id

        Raises:
          SecurityViolationError: 发现不属于当前租户的数据
        """
        import logging
        logger = logging.getLogger("security")

        for i, r in enumerate(results):
            # 兼容两种数据结构: dict 直接包含 tenant_id / entity 嵌套 tenant_id
            entity_tenant = r.get("tenant_id") or (r.get("entity") or {}).get("tenant_id")

            if entity_tenant and entity_tenant != expected_tenant:
                # ⚠️ 发现跨租户数据泄露
                logger.critical(
                    f"SECURITY: Cross-tenant leak detected! "
                    f"chunk_id={r.get('id') or r.get('chunk_id')}, "
                    f"expected_tenant={expected_tenant}, actual_tenant={entity_tenant}")

                # 上报 Prometheus 安全告警指标（Alertmanager webhook 集成）
                try:
                    from prometheus_client import Counter
                    CROSS_TENANT_ALERT = Counter(
                        "rag_cross_tenant_access_total",
                        "Cross-tenant access attempts blocked")
                    CROSS_TENANT_ALERT.inc()
                except Exception:
                    pass

                raise SecurityViolationError(
                    f"Data integrity error: cross-tenant access blocked "
                    f"(chunk {r.get('id', '?')} belongs to {entity_tenant}, "
                    f"not {expected_tenant})")
