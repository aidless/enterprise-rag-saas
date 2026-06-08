# 企业级多租户 RAG SaaS 平台 —— 架构修订方案 v2.0

> **修订日期**：2025-06-04
> **修订范围**：逐一回应 Plan B 架构设计的 10 项 critique
> **编制团队**：智数分析专家团
> **技术栈基线**：Python/FastAPI、Milvus 2.4+、PostgreSQL 16、MinIO、Redis 7、Kafka、Elasticsearch 8.x、vLLM、Ollama

---

## 执行摘要

本次架构修订围绕三条核心原则展开：**安全隔离优先、成本可控演进、可观测性闭环**。

P0 级变更包括四项：(1) 引入 Elasticsearch 8.x 作为独立 BM25 全文检索引擎，与 Milvus 向量检索构成 RRF 混合检索；(2) 从「Partition per Tenant + 查询过滤」升级为「Collection per Tenant + 中间件双重校验」，消除代码漏写导致的数据泄露风险；(3) 补全 Prometheus + Grafana + Alertmanager 可观测性体系，覆盖 10 项核心 RAG 指标；(4) 建立全因子计费模型，纳入 Embedding Token 和 Rerank 调用成本。

总体演进路线为 V1 筑牢基础 → V2 能力增强 → V3 智能自治，每阶段均有明确的交付门槛和验收标准。

---

## 修订总览

### 修订优先级矩阵

| 优先级 | 编号 | 修订项 | 风险等级 | 实施成本 | V1 最小实现 | V2 完善方向 |
|:---:|:---:|------|:---:|:---:|------|------|
| **P0** | #1 | 引入 ES BM25 混合检索引擎 | 🔴 高 | 中 | ES 单节点 + ik_smart 分词 + RRF 融合 | Milvus 2.5 原生 BM25 替代 ES |
| **P0** | #2 | Collection per Tenant 安全隔离 | 🔴 高 | 低 | Partition Key + 中间件拦截器双重校验 | Collection per Tenant 动态创建 |
| **P0** | #6 | Prometheus + Grafana 可观测性 | 🟡 中 | 低 | 10 项核心指标 + Grafana 三屏看板 | 分布式 Tracing + 自动告警规则 |
| **P0** | #7 | 全因子计费模型 | 🟡 中 | 低 | tiktoken 统计 Embedding/Rerank Token | 预付费 + 用量预警 |
| **P1** | #3 | 文档处理容错编排 | 🟡 中 | 中 | Celery 重试 + 死信队列 + 状态机 | OCR 降级 + 置信度评估 |
| **P1** | #4 | LLM 推理 Tier 分层 + 缓存 | 🟡 中 | 高 | 7B/72B 双 Tier + GPTCache | 云端 API 弹性兜底 |
| **P1** | #8 | RAGAS 评估 + 反馈闭环 | 🟢 低 | 中 | RAGAS 离线评估 + 👍/👎 收集 | A/B 实验平台 + 自动优化 |
| **P2** | #5 | Agentic RAG 渐进式上线 | 🟢 低 | 中 | 简洁/专家模式路由 | Agent 插件化触发 |
| **P2** | #9 | 多组件灾备方案 | 🟡 中 | 中 | PG 逻辑备份 + MinIO 版本控制 | 异地多活 + 自动故障转移 |
| **P2** | #10 | 审计日志分级脱敏 | 🟡 中 | 低 | Presidio 自动脱敏 + PIPL 清单 | 细粒度脱敏策略配置 |

---

## 一、检索层加固：Elasticsearch BM25 混合引擎

> 对应 Critique #1：数据层缺少专门的全文搜索引擎

### 1.1 修订前问题

上一版架构在概念设计中提及「Hybrid RAG（向量 + BM25）」但数据层组件清单中仅列出 PostgreSQL、Milvus、MinIO、Redis、Kafka、ClickHouse。**BM25 的实现载体未明确**，可能导致团队用 PostgreSQL 的 `tsvector` 凑合实现全文检索，在中文分词、相关性排序和扩展性上存在严重不足。

### 1.2 修订后架构

```
                          ┌─────────────────────┐
                          │    Query Router      │
                          └──────┬──────┬───────┘
                                 │      │
                    ┌────────────┘      └────────────┐
                    ▼                                 ▼
          ┌─────────────────┐              ┌─────────────────┐
          │  Elasticsearch   │              │     Milvus       │
          │  (BM25 关键词)    │              │  (向量语义检索)   │
          │  index: chunks   │              │  collection: kb  │
          └────────┬────────┘              └────────┬────────┘
                   │                                 │
                   │  各召回 top_k=20                  │
                   └────────┐      ┌─────────────────┘
                            ▼      ▼
                    ┌─────────────────────┐
                    │   RRF Fusion (k=60)  │
                    │   取融合后 Top-10     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │    Reranker (可选)    │
                    │   BGE-Reranker-v2    │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │    LLM Generation    │
                    └─────────────────────┘
```

### 1.3 chunk_id 对齐机制

Elasticsearch 和 Milvus 使用**统一的 `chunk_id`** 作为跨引擎对齐键：

- 文档经 Unstructured/paddle-ocr 解析后，由统一的 Chunking Service 生成 `chunk_id = sha256(doc_id + chunk_index)`
- ES 索引文档时以 `chunk_id` 作为 `_id`，Milvus 插入向量时以 `chunk_id` 作为主键
- RRF 融合时以 `chunk_id` 去重并对排名分累加，避免文本微小差异导致重复

### 1.4 RRF 融合策略

采用业界标准的 **Reciprocal Rank Fusion (RRF)** 算法：

```
RRF_score(d) = Σ [ 1 / (k + rank_r(d)) ]
```

关键参数配置：

| 参数 | 推荐值 | 说明 |
|------|:---:|------|
| `k`（平滑常数） | **60** | 业界标准值，避免低排名文档得分趋近于 0 |
| `bm25_top_k` | **20** | 约为最终返回数的 4 倍，保证足够候选池 |
| `vector_top_k` | **20** | 同上 |
| `final_top_k` | **5** | 送入 LLM 的最终上下文数量 |

**中文分词器选型建议**：索引侧使用 `ik_max_word`（最细粒度，最大化召回），搜索侧使用 `ik_smart`（粗粒度，提升精度）。同时维护行业术语自定义词典，避免专有名词被错误切分。

### 1.5 V1 最小实现

- Elasticsearch 8.x 单节点部署，内存分配 2-4GB
- 安装 `elasticsearch-analysis-ik` 插件
- 实现 `HybridRetriever` 类，内部并行调用 ES `search()` 和 Milvus `search()`
- RRF 融合在应用层完成（约 30 行 Python）

### 1.6 V2 迁移评估：Milvus 2.5 原生 BM25

Milvus 2.4 起支持内置 BM25 Function（通过 `BM25EmbeddingFunction` 生成稀疏向量），2.5 版本进一步增强。V2 可评估以下迁移收益：

- **减少组件依赖**：去掉独立 ES 节点，运维复杂度降低
- **统一存储**：向量和稀疏向量共存于同一 Collection
- **迁移成本**：需重建索引，稀疏向量维度与词表大小相关，存储量增加约 20-30%

**建议**：V1 先用 ES（成熟稳定、生态丰富），V2 在 Milvus 2.5 稳定后评估迁移。

### 1.7 落地风险：ES 与 Milvus 的数据一致性

文档上传后同时向 ES 和 Milvus 双写。若一方成功另一方失败（例如 ES 超时但 Milvus 写入成功），会出现「BM25 漏召回」或「向量漏召回」的**静默错误**——用户不会报错，但检索结果不完整。

**V1 对账方案（增量 + 全量双模）**：

当文档量达到百万级 chunk 时，每小时全量对比会产生大量 I/O。采用**分级对账策略**：

| 模式 | 频率 | 范围 | 触发条件 |
|------|------|------|------|
| **增量对账** | 每 15 分钟 | 最近 15 分钟内新增的 chunk | Cron 自动触发 |
| **全量对账** | 每日凌晨 3:00 | 全部 chunk_id 集合 | 系统低负载窗口 |

```python
# V1 增量对账（高频低开销）
@app.task
def reconcile_incremental():
    """仅对比最近 15 分钟新增的 chunk"""
    since = datetime.utcnow() - timedelta(minutes=15)
    es_new = es_client.list_chunk_ids(created_since=since)
    milvus_new = milvus_client.list_chunk_ids(created_since=since)
    
    missing_in_es = milvus_new - es_new
    missing_in_mv = es_new - milvus_new
    
    if missing_in_es or missing_in_mv:
        logger.warning(f"增量对账发现差异：ES缺{len(missing_in_es)}，Milvus缺{len(missing_in_mv)}")
        # 触发针对性补写
        for chunk_id in missing_in_es:
            reindex_to_es.delay(chunk_id)
        for chunk_id in missing_in_mv:
            reindex_to_milvus.delay(chunk_id)
    
    # 上报 Prometheus 指标
    reconciliation_gauge.labels(type="incremental", kb_id=kb_id).set(
        len(missing_in_es) + len(missing_in_mv)
    )

# V1 全量对账（低频全覆盖）
@app.task
def reconcile_full():
    """仅在凌晨执行，全量对比"""
    if datetime.now().hour != 3:  # 仅在 3:00-4:00 窗口执行
        return
    # 分页对比，每页 10,000 条，避免内存溢出
    offset = 0
    while True:
        es_batch = es_client.list_chunk_ids(offset=offset, limit=10000)
        mv_batch = milvus_client.list_chunk_ids(offset=offset, limit=10000)
        if not es_batch and not mv_batch:
            break
        # 对当前批次做差集对比...
        offset += 10000
```

**告警阈值**：
- 增量差异 > 10 条 → P2（常规通知）
- 全量差异 > 0.1% 总 chunk → P1（需排查双写链路）
- 全量差异 > 1% 总 chunk → P0（严重，可能影响大量用户检索）

**V2 改进方向**：
- 采用 **Outbox 模式**：Chunk 写入先入 PostgreSQL `chunk_outbox` 表（事务内），独立 Worker 分别同步至 ES 和 Milvus，从源头减少不一致
- 或采用**事务消息**（Kafka Transactions）：Chunking 完成后在同一个 Kafka 事务中发布 ES 写入事件和 Milvus 写入事件

---

## 二、安全层加固：Collection per Tenant + 双重校验

> 对应 Critique #2：Milvus 集合按 kb_id 分区 + 查询强制过滤，代码漏写可能导致跨租户数据泄露

### 2.1 隔离方案对比

| 维度 | Partition Key per Tenant | Collection per Tenant | Database per Tenant |
|------|:---:|:---:|:---:|
| **隔离强度** | ⭐⭐ 逻辑隔离 | ⭐⭐⭐⭐ 物理隔离 | ⭐⭐⭐⭐⭐ 完全物理隔离 |
| **代码漏写风险** | 🔴 高（依赖查询过滤） | 🟢 低（元数据层面隔离） | 🟢 极低 |
| **资源开销** | 低（共享 Collection） | 中（每租户独立 Collection） | 高（每租户独立 DB 连接） |
| **租户数上限** | ~10000 | ~1000（Collection 数量限制） | ~100 |
| **跨租户聚合查询** | ✅ 容易 | ⚠️ 需跨 Collection 查询 | ❌ 需跨 DB 查询 |
| **适用阶段** | V1 MVP | V1 生产 / V2 | 大型企业专属部署 |

### 2.2 推荐方案：Partition Key + 中间件双重校验（V1 → Collection per Tenant（V2）

**V1 实现**：利用 Milvus 2.3+ 的 **Partition Key** 特性，以 `tenant_id` 作为分区键，Milvus 在写入时自动根据 `tenant_id` 值路由到对应分区。查询时 Partition Key 由 Milvus **服务端强制过滤**，不会出现应用层漏写过滤条件的问题——这是相较于手动 `filter` 表达式的关键提升。

**双重校验机制**：

```python
# 中间件层：统一注入租户上下文
class TenantContextMiddleware:
    """FastAPI 中间件：从 JWT Token 提取 tenant_id 并注入请求上下文"""
    
    async def __call__(self, request, call_next):
        tenant_id = self.extract_tenant_from_token(request)
        request.state.tenant_id = tenant_id
        return await call_next(request)


# 查询层：强制注入 Partition Key
class TenantAwareMilvusClient:
    """所有 Milvus 操作强制携带 tenant_id，编译器级别保证"""
    
    def search(self, tenant_id: str, collection: str, vector: list, **kwargs):
        # 编译时强制，不存在「漏写」可能
        return milvus.search(
            collection_name=collection,
            data=[vector],
            partition_names=[f"tenant_{tenant_id}"],  # Partition Key 隔离
            **kwargs
        )


# 返回层：二次校验
class ResponseValidator:
    """对检索返回的 chunk 进行 tenant_id 二次校验"""
    
    def validate(self, results: list, expected_tenant: str):
        for r in results:
            if r.get("tenant_id") != expected_tenant:
                logger.critical(f"Cross-tenant leak detected: {r['id']}")
                raise SecurityViolationError()
```

### 2.3 V1 最小实现

- Milvus 启用 Partition Key（需 Milvus 2.3+）
- FastAPI 中间件从 JWT 提取 `tenant_id` 并注入 `request.state`
- 所有 Milvus 操作通过 `TenantAwareMilvusClient` 封装，禁止直接调用裸 SDK
- 返回结果增加 `tenant_id` 断言校验

### 2.4 V2 完善方向

- 迁移至 **Collection per Tenant**：通过 Milvus RBAC 为每个租户创建独立 Collection 和 API Key
- 结合 Kubernetes Namespace 实现网络层隔离
- 支持大型企业客户的专属部署模式

### 2.5 落地风险：Partition Key 的租户数量天花板

Milvus 2.4 中，单个 Collection 的 Partition 数量上限为 **4,096**。对于 SaaS 平台而言，这意味着当租户数接近该上限时，Partition Key 方案将无法新建租户。

**量化约束与监控**：

| 里程碑 | 租户数 | 操作 |
|--------|--------|------|
| < 2,000 | 安全区 | 正常运营 |
| 2,000-3,000 | 预警区 | 触发 Grafana 面板告警，开始评估迁移方案 |
| 3,000-3,800 | 计划迁移 | 启动 Collection per Tenant 灰度迁移 |
| > 3,800 | 紧急 | 暂停新租户注册，加速迁移 |

```python
# Milvus 租户容量监控
class TenantCapacityMonitor:
    MAX_PARTITIONS = 4096
    WARN_THRESHOLD = 3000   # ~73% 使用率
    BLOCK_THRESHOLD = 3800  # ~93% 使用率
    
    def check_capacity(self):
        used = milvus.get_partition_count()
        if used >= self.BLOCK_THRESHOLD:
            alert("CRITICAL: 租户分区数接近上限，暂停新租户创建")
            feature_flags.disable("new_tenant_registration")
        elif used >= self.WARN_THRESHOLD:
            alert("WARNING: 租户分区数超过 3000，建议启动 Collection per Tenant 迁移")
```

**V1 应对策略**：
- 架构文档中明确标注 `max_tenants=3800`（安全上限）
- 在 Grafana 看板中增加「租户容量使用率」指标（每分钟采集）
- 当租户数到达 3,000 时，自动触发迁移计划告警

**V2 完善方向**：
- 提前启动 Collection per Tenant 迁移脚本，支持新旧租户分片共存
- 新租户直接创建独立 Collection，存量租户逐步迁移

### 2.6 落地风险：管理后台 RBAC 细粒度控制

当前文档提到「超管 / 管理员 / 成员 / 只读」四级角色，但 SaaS 场景中一个企业租户内部可能有多个知识库，不同成员对不同知识库的权限可能不同（如「知识库 A 编辑者 + 知识库 B 只读者」）。扁平化的全局角色无法覆盖。

**V1 RBAC + ABAC 混合模型**：

```python
# 权限模型：谁（主体）对什么（资源）能做什么（操作）在什么条件下（环境）
class PermissionModel:
    """
    RBAC:  角色 → 权限集合（如 editor = [kb.upload, kb.query, kb.delete]）
    ABAC:  属性约束（如 IP 白名单、时间段、MFA 状态）
    """
    pass

# 数据库 Schema
# 角色表（租户可自定义）
tenant_roles:
    id, tenant_id, name, permissions (JSONB)
    # permissions = ["kb:upload", "kb:query", "kb:delete", "kb:manage", "member:invite"]

# 成员-知识库-角色关联表（支持跨知识库不同权限）
kb_members:
    id, kb_id, user_id, role_id, granted_by, granted_at, expires_at
    # 示例：用户 A 在知识库 1 是 editor，在知识库 2 是 viewer
```

**中间件层权限校验**：

```python
class KBPermissionMiddleware:
    """所有 kb_* 端点统一校验 kb_id 权限"""
    
    async def __call__(self, request, call_next):
        kb_id = request.path_params.get("kb_id")
        if kb_id:
            user_id = request.state.user_id
            required_action = self.action_from_method(request.method, request.url.path)
            # required_action 示例: "kb:query", "kb:upload", "kb:delete"
            
            if not self.has_permission(user_id, kb_id, required_action):
                raise HTTPException(403, f"您没有知识库 {kb_id} 的 {required_action} 权限")
        
        return await call_next(request)
    
    def has_permission(self, user_id, kb_id, action):
        # 1. 查询用户在目标知识库的角色
        membership = db.query(KBMember).filter_by(user_id=user_id, kb_id=kb_id).first()
        if not membership:
            return False
        
        # 2. 检查角色是否包含所需操作
        role = db.query(TenantRole).get(membership.role_id)
        return action in role.permissions
    
    def action_from_method(self, method, path):
        """HTTP 方法 → 权限动作映射"""
        mapping = {
            ("GET", "/query"):  "kb:query",
            ("POST", "/upload"): "kb:upload",
            ("DELETE", "/"):    "kb:delete",
            ("PUT", "/"):       "kb:manage",
        }
        for (m, suffix), action in mapping.items():
            if method == m and suffix in path:
                return action
        return "kb:read"
```

**前端管理后台需求**：
- 租户管理员可创建自定义角色（如「审计员」权限 = kb:query + kb:export）
- 拖拽用户到知识库卡片，选择角色 → 一键授权
- 成员列表展示每个用户在各知识库的角色标签
- 权限变更写入审计日志（谁、何时、授予/撤销了什么权限）

**V1 最小实现**：
- 知识库级别的基础 RBAC（owner/editor/viewer）
- 所有 kb_* API 端点通过中间件校验 kb_id 权限
- 前端提供成员管理页面（列表 + 角色下拉框）

**V2 完善方向**：
- 租户自定义角色（可组合权限集合）
- ABAC 增强：IP 白名单、时间段限制、MFA 要求
- 权限模板：一键应用「研发团队标准权限」「法务团队只读权限」等预置模板

---

## 三、文档处理层加固：容错编排 + 细粒度状态机

> 对应 Critique #3：文档解析流水线缺少异常处理和部分失败策略

### 3.1 文档状态机完整流转

```
                    ┌──────────┐
                    │ UPLOADED │  ← 文件上传至 MinIO
                    └────┬─────┘
                         │ Celery Task: process_document
                         ▼
                  ┌──────────────┐
                  │  PROCESSING   │  ← 解析进行中
                  └──┬───┬───┬───┘
                     │   │   │
              ┌──────┘   │   └──────┐
              ▼          ▼          ▼
        ┌─────────┐ ┌────────┐ ┌──────────┐
        │  DONE    │ │PARTIAL │ │  FAILED   │
        │(全部成功) │ │(部分成功)│ │(全部失败)  │
        └────┬────┘ └───┬────┘ └────┬─────┘
             │          │           │
             ▼          ▼           ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │  CHUNKED  │ │PARTIAL   │ │  DEAD     │
       │ (向量化完成)│ │_CHUNKED  │ │(死信队列)  │
       └────┬─────┘ └────┬─────┘ └──────────┘
            │            │
            ▼            ▼
      ┌──────────┐ ┌──────────┐
      │  READY   │ │PARTIAL   │
      │(可检索)   │ │_READY    │
      └──────────┘ │(部分可检索)│
                   └──────────┘
```

### 3.2 Celery 任务编排架构

```python
@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,      # 首次重试延迟 60s
    retry_backoff=True,           # 指数退避：60s → 120s → 240s
    retry_backoff_max=600,        # 最大退避 600s
    autoretry_for=(IOError, TimeoutError, ConnectionError),
    acks_late=True,               # 任务完成后才确认，防止 worker 崩溃丢任务
)
def process_document(self, doc_id: str):
    """文档处理主任务：解析 → 分块 → 向量化"""
    ...
```

### 3.3 容错降级策略矩阵

| 故障场景 | 降级策略 | 用户可见影响 | 恢复方式 |
|------|------|------|------|
| PDF 解析失败 | 回退 PyPDF2 → pdfplumber → 标记为不可解析 | 文档标记「解析失败」，显示原因 | 人工上传文本版本 |
| OCR 服务超时 | PaddleOCR 本地 CPU fallback；超时 30s 后跳过图片 | 文档标记「部分图片未识别」 | 重试或人工标注 |
| 表格提取置信度低 | 置信度 < 0.7 时标记为「低置信度表格」，保留原始图片引用 | LLM 回答时看到表格图片链接 | 人工校验后确认 |
| 某页解析崩溃 | try/catch 单页隔离；失败页面跳过，成功页面继续 | 文档标记 PARTIAL_READY | 重试失败页面 |
| 连续 3 次重试仍失败 | 移入死信队列（Redis DLQ），触发 Webhook 通知 | 管理员收到告警 | 人工介入排查 |

### 3.4 V1 最小实现

- Celery 配置指数退避重试 + 死信队列
- 每个 Parser 内部 try/catch 单页隔离
- 实现 `UPLOADED → PROCESSING → (DONE/PARTIAL/FAILED) → (CHUNKED/PARTIAL_CHUNKED) → (READY/PARTIAL_READY/DEAD)` 状态机
- 死信队列接入企业微信/钉钉 Webhook 通知

### 3.5 V2 完善方向

- 表格提取增加置信度评分模型（基于行列结构完整度 + OCR 字符置信度）
- OCR 服务独立部署（PaddleOCR Serving），避免本地 fallback 性能瓶颈
- 支持文档级别的「重新解析」和「单页重试」

### 3.6 落地风险：PARTIAL_READY 状态的用户体验

PARTIAL_READY 文档的部分成功 chunk 与完整文档的 chunk 在检索时会被同等对待。用户可能问到一个信息恰好位于失败页面中，系统返回「未找到相关答案」，而用户看不到任何提示说明该内容存在于未解析页面中，造成**体验割裂**。

**V1 解决方案**：

| 场景 | 处理方式 |
|------|----------|
| 前端文档列表 | PARTIAL_READY 文档显示 🟡 角标：「该文档部分内容未索引成功（3/15 页）」 |
| 检索无结果 | 追加兜底提示：「您的问题可能与以下文档相关，但这些文档的部分内容暂未成功索引：[文档A (第5页解析失败), 文档B (第12页图片未识别)]」 |
| 用户提问时 | 检查问题语义是否匹配到 PARTIAL_READY 文档的失败页面摘要；若匹配，告知用户可尝试手动查看原始文档 |

```python
# 检索无结果时的兜底提示
def augment_no_result_response(query: str, kb_id: str):
    partial_docs = get_partial_docs(kb_id)
    if not partial_docs:
        return None
    
    # 将失败页面的摘要与 query 做相似度匹配
    candidate_hints = []
    for doc in partial_docs:
        for failed_page in doc.failed_pages:
            similarity = cosine_similarity(query_emb, failed_page.summary_emb)
            if similarity > 0.6:
                candidate_hints.append({
                    "doc_name": doc.name,
                    "page": failed_page.page_num,
                    "reason": failed_page.error_reason
                })
    
    if candidate_hints:
        return {
            "type": "partial_warning",
            "message": "以下文档的部分内容暂未成功索引，可能包含您需要的信息：",
            "hints": candidate_hints[:3],
            "suggestion": "建议直接查看原始文档以获取完整信息"
        }
```

**前端展示规范**：
- PARTIAL_READY 文档在知识库列表中展示黄色警告图标
- 鼠标悬停显示 Tooltip：「3/15 页解析成功，12 页因 [OCR 超时/表格识别失败] 暂不可检索」
- 文档详情页提供**「重试失败页面」按钮**，V1 即上线（不等 V2 的完整「重新解析」功能）

**V1 重试按钮实现**：

```python
# API 端点：重试 PARTIAL_READY 文档的失败页面
@router.post("/documents/{doc_id}/retry-failed-pages")
async def retry_failed_pages(doc_id: str, tenant_id: str = Depends(get_tenant_id)):
    doc = await get_document(doc_id)
    if doc.status != "PARTIAL_READY":
        raise HTTPException(400, "仅 PARTIAL_READY 状态的文档可重试失败页面")
    
    # 状态回退为 PROCESSING，仅处理失败的页面
    await update_doc_status(doc_id, "PROCESSING")
    
    # 重新入队，仅处理失败页面
    retry_failed_pages_task.delay(
        doc_id=doc_id,
        failed_pages=doc.failed_pages,    # 只重试失败的页面
        original_chunks=doc.success_chunks # 保留已成功的 chunk
    )
    
    return {"status": "retrying", "failed_pages": len(doc.failed_pages)}
```

前端交互逻辑：
1. PARTIAL_READY 文档详情页展示「成功 3 页 / 失败 12 页」统计
2. 点击「重试失败页面」→ 按钮置灰 + loading → 状态显示 PROCESSING
3. 重试完成 → 状态更新为 READY（若全部成功）或 PARTIAL_READY（仍有失败）
4. 重试失败页面列表更新：部分恢复为成功，部分可能仍失败

---

## 四、推理层优化：Tier 分层 + 缓存 + API 兜底

> 对应 Critique #4：vLLM + Qwen2.5-72B 成本极高

### 4.1 成本量化分析

| 部署方案 | GPU 需求 | 显存占用 | 吞吐量 | 月成本估算（单卡租赁） |
|------|------|------|------|------|
| Qwen2.5-72B (FP16) | 4× A100 80GB | ~280GB | ~50 tok/s | ¥60,000-80,000 |
| Qwen2.5-72B (Q4_K_M) | 1× A100 80GB | ~42GB | ~18 tok/s | ¥15,000-20,000 |
| Qwen2.5-14B (FP16) | 1× A100 80GB | ~28GB | ~80 tok/s | ¥15,000-20,000 |
| Qwen2.5-7B (FP16) | 1× RTX 4090 24GB | ~14GB | ~120 tok/s | ¥3,000-5,000 |
| Qwen2.5-7B (Ollama 本地) | CPU / 消费级 GPU | ~5GB | ~15 tok/s | ¥0（已有硬件） |

> 数据来源：vLLM 社区基准测试、A100 云租赁市场公开报价（2025 Q2）

### 4.2 模型 Tier 策略

```
                        ┌──────────────────┐
                        │   Query Router    │
                        │  (复杂度判定 +      │
                        │   租户 Tier 匹配)   │
                        └──┬───┬───────┬───┘
                           │   │       │
              ┌────────────┘   │       └────────────┐
              ▼                ▼                    ▼
     ┌────────────┐   ┌────────────┐      ┌──────────────┐
     │  Tier 0     │   │  Tier 1     │      │  Tier 2       │
     │  Qwen2.5-7B │   │ Qwen2.5-14B│      │ Qwen2.5-72B   │
     │  Ollama 本地 │   │ vLLM 单卡   │      │ vLLM 多卡/API │
     │  免费        │   │ 基础版租户   │      │ 高级版租户     │
     └────────────┘   └────────────┘      └──────────────┘
```

| 租户套餐 | 默认模型 | 并发限制 | 语义缓存 | 包含于 |
|------|------|------|:---:|------|
| Free | Qwen2.5-7B (Ollama) | 2 QPS | ❌ | 免费试用 |
| Basic | Qwen2.5-14B (vLLM) | 10 QPS | ✅ | ¥999/月 |
| Pro | Qwen2.5-72B (vLLM) | 50 QPS | ✅ | ¥4,999/月 |
| Enterprise | 72B 专属 + 云端 API 兜底 | 自定义 | ✅ | 定制报价 |

### 4.3 GPTCache 语义缓存方案

使用 Zilliz 开源的 **GPTCache**，以 Redis 作为向量存储后端：

```python
from gptcache import Cache
from gptcache.manager import manager_factory
from gptcache.embedding import ONNX
from gptcache.similarity_evaluation import SbertCrossEncoderEvaluation

cache = Cache()
cache.init(
    pre_embedding_func=query_embedding_fn,   # 用 BGE-small 对查询做向量化
    embedding_func=ONNX(),                    # ONNX 加速推理
    data_manager=manager_factory(
        "redis,faiss",                        # Redis 存储元数据 + Faiss 索引
        vector_params={"dimension": 512},
    ),
    similarity_evaluation=SbertCrossEncoderEvaluation(),  # 交叉编码器精排
    similarity_threshold=0.85,                # 相似度 > 0.85 命中缓存
)
```

**预期效果**：典型企业知识库问答场景下（FAQ 类问题占比 30-50%），语义缓存命中率可达 **25-40%**，对应 LLM 推理成本降低同等比例。

### 4.4 云端 API 弹性兜底

当本地 vLLM 队列深度超过阈值时：

1. **溢出策略**：排队数 > N 时，新请求路由至通义千问 API / DeepSeek API
2. **故障转移**：本地 vLLM 不可用时，自动切换至云端 API
3. **成本控制**：API 用量设置每日预算上限，超出后回退到队列等待

### 4.5 V1 最小实现

- Tier 0（7B Ollama 本地）+ Tier 2（72B API 兜底）双 Tier
- GPTCache + Redis 语义缓存
- 流式返回（SSE）降低感知延迟

### 4.6 V2 完善方向

- 引入 Qwen2.5-14B 中间 Tier
- 语义缓存升级为分布式集群
- 基于历史数据训练复杂度路由器，自动判定所需模型 Tier

### 4.7 落地风险：多租户推理公平性保障

当前 Tier 设计中，Pro 租户获得 50 QPS 配额，但**未说明当多个 Pro 租户同时高负载时，如何防止单一租户占满 vLLM 推理队列**。若一个租户的大量请求导致其他同 Tier 租户排队超时，SLA 即告失效。

**V2 令牌桶限流方案**（V1 暂缓，但须在架构中预留接口）：

```
                   ┌─────────────────────────────┐
                   │      API Gateway             │
                   │  ┌─────────────────────────┐ │
                   │  │ Per-Tenant Rate Limiter  │ │
                   │  │  ┌───────┐ ┌───────┐    │ │
                   │  │  │TenantA│ │TenantB│ ...│ │
                   │  │  │Token  │ │Token  │    │ │
                   │  │  │Bucket │ │Bucket │    │ │
                   │  │  │50 QPS │ │50 QPS │    │ │
                   │  │  └───┬───┘ └───┬───┘    │ │
                   │  └──────┼─────────┼────────┘ │
                   └─────────┼─────────┼──────────┘
                             │         │
                   ┌─────────▼─────────▼──────────┐
                   │   vLLM Inference Scheduler    │
                   │  ┌─────────────────────────┐  │
                   │  │ Weighted Fair Queuing    │  │
                   │  │ 每个租户最小保证并发 ≥ 2  │  │
                   │  └─────────────────────────┘  │
                   └──────────────────────────────┘
```

```python
# 令牌桶：每租户独立限流
import time
from collections import defaultdict

class PerTenantRateLimiter:
    """每租户独立的令牌桶限流器"""
    
    def __init__(self):
        self.buckets: dict[str, TokenBucket] = {}
        self.default_limits = {
            "free":    {"rate": 2,   "burst": 3},     # Free:  2 QPS, 突发 3
            "basic":   {"rate": 10,  "burst": 15},    # Basic: 10 QPS, 突发 15
            "pro":     {"rate": 50,  "burst": 60},    # Pro:   50 QPS, 突发 60
            "enterprise": {"rate": 100, "burst": 120}  # Enterprise: 100 QPS
        }
    
    def allow_request(self, tenant_id: str, tier: str) -> bool:
        if tenant_id not in self.buckets:
            limit = self.default_limits[tier]
            self.buckets[tenant_id] = TokenBucket(
                rate=limit["rate"],
                burst=limit["burst"]
            )
        return self.buckets[tenant_id].consume()

# vLLM 调度层：加权公平队列（最小保证）
class TenantAwareScheduler:
    """确保每个租户至少获得配额的"最小保证"并发"""
    
    MIN_GUARANTEED_CONCURRENCY = {
        "free": 1,      # 至少 1 个并发槽位
        "basic": 2,
        "pro": 5,
        "enterprise": 10
    }
    
    def schedule(self, pending_requests: list) -> list:
        """按最小保证分配 vLLM 并发槽位，剩余槽位按比例分配"""
        allocated = []
        remaining_slots = vllm_max_concurrency
        
        # Round 1: 每个租户分配最小保证
        for tenant_id, reqs in group_by_tenant(pending_requests).items():
            min_slots = self.MIN_GUARANTEED_CONCURRENCY[get_tier(tenant_id)]
            allocated.extend(reqs[:min_slots])
            remaining_slots -= min(min_slots, len(reqs))
        
        # Round 2: 剩余槽位按 QPS 配额比例分配
        if remaining_slots > 0:
            total_qps = sum(get_qps_limit(t) for t in active_tenants)
            for tenant_id, reqs in group_by_tenant(remaining_requests).items():
                share = int(remaining_slots * get_qps_limit(tenant_id) / total_qps)
                allocated.extend(reqs[:share])
        
        return allocated
```

**告警指标**：
- 租户被限流次数（`tenant_rate_limit_hits`）—— 若某租户持续被限流，提示其升级 Tier
- vLLM 队列中各租户等待时间分布 —— 检测是否存在「饥饿租户」

**V1 最低实现**：
- API Gateway 层实现基础的 per-tenant QPS 计数器（无需完整令牌桶）
- 当某租户 QPS 超配额 20% 时返回 429 + `Retry-After` 头
- Prometheus 采集限流指标，Grafana 面板展示各租户 QPS 排行

### 4.8 落地风险：开发/测试环境的推理成本

§4.1 的成本量化仅计算生产环境 GPU 租赁，但若开发/测试环境同样使用 vLLM + 72B 付费 GPU，每月额外烧掉 ¥15,000-20,000 是完全不必要的。

**V1 环境分离策略**：

| 环境 | 推理引擎 | 模型 | 成本/月 | 策略 |
|------|------|------|------|------|
| **开发** | Ollama（本地） | Qwen2.5:1.5B / 7B 量化 | ¥0 | 严禁连接付费 GPU，CI 中检查环境变量 |
| **测试** | vLLM（共享） | Qwen2.5:14B | ~¥2,000 | 按需启动（K8s CronJob 夜间运行集成测试） |
| **预发布** | vLLM（独立） | Qwen2.5:72B (INT4) | ~¥5,000 | 与生产同配置，仅上线前 3 天启用 |
| **生产** | vLLM（独立） | Tier 分层 | §4.1 已估算 | 全时运行 |

**开发环境安全措施**：

```makefile
# Makefile：一键切换本地模型，防止误连生产 API
.PHONY: dev-local
dev-local:
	@echo "切换为本地开发模式（Ollama + 7B）"
	export LLM_PROVIDER=ollama
	export LLM_MODEL=qwen2.5:7b
	export EMBEDDING_MODEL=bge-small-zh-v1.5
	export MILVUS_URI=./milvus_lite.db
	@echo "已配置本地环境，不会产生云端费用"

.PHONY: check-not-production
check-not-production:
	@if [ "$$LLM_PROVIDER" = "vllm" ] && [ "$$ENV" != "production" ]; then \
		echo "❌ 错误：非生产环境禁止使用 vLLM，请先运行 make dev-local"; \
		exit 1; \
	fi
```

**CI 管道保护**：
```yaml
# .github/workflows/test.yml
- name: Verify local-only LLM
  run: |
    if [ "$LLM_PROVIDER" != "ollama" ]; then
      echo "❌ CI 环境必须使用 Ollama 本地模型"
      exit 1
    fi
```

**V2 完善方向**：
- 测试环境 vLLM Pod 通过 K8s HPA 缩容至 0（仅夜间自动扩容运行测试）
- 引入 GPU 分时租赁（如 AutoDL），进一步降低测试环境成本

---

## 五、Agent 层控制：渐进式上线策略

> 对应 Critique #5：Agentic RAG 增加复杂度和不确定性，多步推理导致错误级联

### 5.1 渐进式路线

```
V1: 简洁/专家模式路由          V2: Agentic 插件触发          V3: 智能自治
─────────────────────      ─────────────────────      ─────────────────
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ Simple Mode      │        │ + Query Rewrite  │        │ + 多步推理        │
│ (单轮 Hybrid RAG) │   →    │ + 工具调用        │   →    │ + 自主规划        │
│                  │        │ + 子问题拆解      │        │ + 自我纠错        │
│ Expert Mode      │        │ (可选插件)        │        │                  │
│ (多轮澄清+检索)   │        │                  │        │                  │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

### 5.2 V1：简洁模式 vs 专家模式路由

```python
class QueryRouter:
    """基于查询复杂度的模式路由"""
    
    SIMPLE_PATTERNS = [
        r"^(什么是|如何|怎么|为什么)",     # 知识问答
        r"^(列出|总结|概括)",               # 信息提取
    ]
    EXPERT_PATTERNS = [
        r"(对比|比较|区别)",                # 对比分析
        r"(第一步.*第二步|首先.*然后)",      # 多步指令
        r"(\d{4}年.*\d{4}年)",             # 时间跨度
    ]
    
    def route(self, query: str) -> Literal["simple", "expert"]:
        if any(re.match(p, query) for p in self.EXPERT_PATTERNS):
            return "expert"
        return "simple"
```

- **简洁模式**：单轮 Hybrid RAG（BM25 + 向量 → RRF → LLM），延迟 < 2s
- **专家模式**：多轮澄清 + 检索链，允许用户确认中间结果，延迟 < 5s

### 5.3 V2：Agentic 插件触发机制

Agentic 能力设计为**可选插件**，用户在前端手动开启：

- **Query Rewrite 插件**：对模糊查询自动改写，显示改写结果供用户确认
- **子问题拆解插件**：复杂问题拆为 2-4 个子问题，逐一检索后合并
- **工具调用插件**：允许 LLM 调用计算器、日期转换器等确定性工具

### 5.4 失败回退设计

```
Agent 推理失败时的回退链：
  Agent 多步推理
      │
      ├─ (1) 某步检索无结果 → 扩大检索范围 (top_k 20→40) → 重试
      ├─ (2) LLM 判断无需检索 → 强制至少检索一次 → 对比有无检索的答案质量
      ├─ (3) 推理超时 > 10s → 回退为单轮 Hybrid RAG → 返回结果 + 标记 degraded
      └─ (4) 所有回退失败 → 返回"暂时无法回答" + 记录完整 trace
```

### 5.5 V1 最小实现

- 规则驱动的简洁/专家模式路由（~50 行 Python）
- 所有 LLM 调用设置 10s 超时
- 超时自动回退至单轮 Hybrid RAG
- LangFuse 记录全链路 Trace

---

## 六、可观测性补全：Prometheus + LangFuse 双轨制

> 对应 Critique #6：可观测性与告警覆盖不全，缺少 Prometheus + Alertmanager

### 6.1 双轨制分工

| 维度 | Prometheus + Grafana | LangFuse |
|------|------|------|
| **监控对象** | 系统基础设施（CPU/内存/QPS/延迟） | LLM 应用层（Trace/Token/成本/质量） |
| **数据类型** | 时序指标 (Metrics) | 分布式追踪 (Traces) + 评估分数 |
| **告警** | ✅ Alertmanager 规则引擎 | ⚠️ 需额外配置 Webhook |
| **存储** | Prometheus TSDB (15d 保留) | PostgreSQL + ClickHouse |
| **典型查询** | "Milvus P99 延迟 > 100ms？" | "用户 X 的这次问答为什么得分低？" |

### 6.2 10 项核心监控指标

| # | 指标名称 | 类型 | 告警阈值 | 说明 |
|:---:|------|:---:|------|------|
| 1 | `rag_search_latency_p99` | Histogram | > 2000ms | 检索端到端延迟 P99 |
| 2 | `milvus_search_latency_p99` | Histogram | > 100ms | Milvus 向量检索延迟 |
| 3 | `es_search_latency_p99` | Histogram | > 500ms | ES BM25 检索延迟 |
| 4 | `llm_queue_depth` | Gauge | > 20 | vLLM 推理排队深度 |
| 5 | `llm_ttft_p99` | Histogram | > 3000ms | LLM 首 Token 延迟 |
| 6 | `embedding_queue_depth` | Gauge | > 50 | Embedding 任务排队数 |
| 7 | `doc_processing_error_rate` | Counter | > 5% | 文档处理失败率 |
| 8 | `semantic_cache_hit_rate` | Gauge | < 20%（预警） | 语义缓存命中率 |
| 9 | `cross_tenant_access_alert` | Counter | > 0（立即告警） | 跨租户访问拦截次数 |
| 10 | `api_error_rate_5xx` | Counter | > 1% | API 5xx 错误率 |

### 6.3 Grafana 三屏看板设计

| 看板 | 目标受众 | 核心面板 |
|------|------|------|
| **系统健康屏** | SRE/运维 | 各组件 UP/DOWN 状态、CPU/内存/磁盘、QPS、P99 延迟趋势 |
| **RAG 质量屏** | 算法工程师 | 检索召回率、LLM 忠实度、答案相关性、缓存命中率趋势 |
| **多租户用量屏** | PM/运营 | 每租户 QPS、Token 消耗、文档量、错误率排行 |

### 6.4 V1 最小实现

- `prometheus_client` + FastAPI `/metrics` 端点
- Grafana 导入三屏看板 JSON（可参考社区 Dashboard #19275）
- Alertmanager 配置 4 条关键告警：Milvus 延迟、LLM 排队、文档处理失败率、5xx 错误率
- 告警通道：企业微信 Webhook

### 6.5 V2 完善方向

- LangFuse 深度集成，每个用户问答关联完整 Trace
- 自动告警规则生成（基于历史基线 ± 3σ）
- 多租户用量看板支持实时刷新和导出

---

## 七、计费模型完善：全因子计费

> 对应 Critique #7：计费粒度遗漏了 Embedding 和 Rerank 的 Token 成本

### 7.1 完整计费因子清单

| 计费因子 | 计量单位 | 统计方式 | Free | Basic | Pro |
|------|:---:|------|:---:|:---:|:---:|
| **文档存储** | GB/月 | MinIO bucket 用量 | 0.1 GB | 1 GB | 10 GB |
| **文档处理** | 页数 | 按成功解析页数计 | 50 页/月 | 1,000 页/月 | 10,000 页/月 |
| **向量存储** | 万向量/月 | Milvus `num_entities` | 1 万 | 50 万 | 500 万 |
| **问答次数** | 次 | API 调用计数 | 100 次/月 | 5,000 次/月 | 50,000 次/月 |
| **Embedding Token** ⭐ 新增 | 万 Token | tiktoken 精确统计 | 10 万 | 500 万 | 5,000 万 |
| **LLM 推理 Token** | 万 Token | vLLM/LangFuse 统计 | 含免费 Tier | 100 万 | 1,000 万 |
| **Rerank 调用** ⭐ 新增 | 千次 | Reranker API 计数 | 0 | 50 千次 | 500 千次 |

> ⭐ 标记为本次修订新增计费因子

### 7.2 Embedding Token 统计实现

```python
import tiktoken

class EmbeddingTokenMeter:
    """精确统计 Embedding 调用的 Token 消耗"""
    
    def __init__(self, model: str = "bge-small-zh-v1.5"):
        # BGE 系列使用 BERT tokenizer，近似统计可用 tiktoken cl100k_base
        # 更精确的方案是加载对应模型的 tokenizer
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def record(self, tenant_id: str, doc_id: str, text: str):
        tokens = self.count(text)
        # 写入 ClickHouse 或 PostgreSQL 计费表
        save_usage(
            tenant_id=tenant_id,
            metric="embedding_tokens",
            value=tokens,
            metadata={"doc_id": doc_id}
        )
```

### 7.3 Rerank 调用的计费折算

Reranker 模型（如 BGE-Reranker-v2-m3）每次调用消耗 GPU 算力。计费方案：

- **自部署 Reranker**：按 GPU 时间折算成本，约 ¥0.001/千次调用
- **云端 API Reranker**（如 Cohere Rerank）：按调用次数计费，约 ¥0.015/千次
- **建议定价**：Rerank 调用 ¥0.005/千次（自部署），¥0.02/千次（云端 API）

### 7.4 V1 最小实现

- 在 API Gateway 层增加 Token 统计中间件
- 所有 LLM/Embedding/Rerank 调用写入 `usage_logs` 表（tenant_id, metric, value, timestamp）
- ClickHouse 物化视图按小时/天/月聚合
- 管理后台展示实时用量

### 7.5 V2 完善方向

- 预付费模式：预购 Token 包，余额不足时自动降级（Pro → Basic → Free）
- 用量预警：达到 80%/90%/100% 配额时发送通知
- 自定义报价：支持大型企业按年签约

---

## 八、质量闭环：RAGAS 评估 + 在线反馈

> 对应 Critique #8：缺少 RAG 效果评估和反馈闭环

### 8.1 RAGAS 四大评估维度

| 指标 | 评估对象 | 是否需要参考答案 | 计算方法 | 目标值 |
|------|:---:|:---:|------|:---:|
| **忠实度** (Faithfulness) | 生成器 | ❌ | 被支撑的声明数 / 总声明数 | > 0.85 |
| **答案相关性** (Answer Relevancy) | 生成器 | ❌ | 反推问题与原问题的语义相似度均值 | > 0.80 |
| **上下文召回** (Context Recall) | 检索器 | ✅ 需要 | 被检索覆盖的参考答案句子数 / 总句子数 | > 0.80 |
| **上下文精确** (Context Precision) | 检索器 | ❌ | 相关文档排名的加权精确率 | > 0.75 |

### 8.2 标注测试集构建规范

```
测试集最小规模：50 条 QA 对
├── 事实型查询 (30%) ：「公司2024年营收是多少？」
├── 总结型查询 (25%)  ：「总结产品X的主要功能」
├── 对比型查询 (20%)  ：「产品A和产品B的区别是什么？」
├── 过程型查询 (15%)  ：「如何配置XX功能？请列出步骤」
└── 边界型查询 (10%)  ：「知识库中没有的内容应该如何处理？」

每对包含：
  - question: 用户问题
  - answer: 参考答案（人工标注，100-300 字）
  - contexts: 预期命中的文档片段 ID（至少 1 条）
```

### 8.3 👍/👎 反馈数据流水线

```
用户点击 👍/👎
      │
      ▼
┌─────────────────┐
│ Feedback Logger   │  ← 记录：query, answer, context, rating, user_comment
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ClickHouse        │  ← 实时聚合：每知识库的好评率、差评原因分布
│ feedback_events   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 每周自动分析      │  ← 导出低分 Case → 人工复核 → 更新测试集
│ (Cron Job)       │     高分组 Case → 加入 RAGAS 测试集正样本
└─────────────────┘
```

### 8.4 V1 最小实现

- 构建 50 条标注 QA 测试集（覆盖 5 种查询类型）
- 每周自动运行 RAGAS 评估，生成质量报告
- 前端增加 👍/👎 按钮，数据写入 ClickHouse
- 低分 Case（👎）自动推送到指定 Slack/企微频道

### 8.5 V2 完善方向

- 测试集扩展至 200+ 条
- A/B 实验平台：对比不同 Chunk Size、Embedding 模型、Reranker 的效果
- 基于反馈数据自动微调检索参数（top_k、RRF k 值等）

### 8.6 落地风险：RAGAS 对中文的适配

RAGAS 默认使用英文 LLM（如 GPT-3.5/GPT-4）作为评分模型。在 `faithfulness` 和 `answer_relevancy` 评估中，评分 LLM 需要判断中文 statement 是否可由 context 支撑、反向生成问题是否与原问题语义一致——**英文 LLM 处理中文时，这两种判断都可能失真**，导致评分不可信。

**V1 解决方案**：

```python
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from langchain_community.llms import Ollama

# 明确指定中文 LLM 作为 RAGAS 评分模型
scoring_llm = Ollama(
    model="qwen2.5:7b",        # 中文 LLM
    temperature=0.0,            # 评分场景需确定性输出
    system="你是一个严格的评估专家，请用中文评估。"
)

# 替换默认的英文评分 LLM
faithfulness.llm = LangchainLLMWrapper(scoring_llm)
answer_relevancy.llm = LangchainLLMWrapper(scoring_llm)

# 运行评估
result = evaluate(
    dataset=zh_test_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
```

**V1 Prompt 工程——投入少量时间做对比验证**：

faithfulness 和 answer_relevancy 的默认 Prompt 针对英文设计，直接翻译后其中文语义可能不够精确。V1 建议投入 **10-20 条人工标注的 case**，做一轮 Prompt 对比实验：

```python
# 对比三种 Prompt 方案的评分准确率
prompt_variants = {
    "default_en":     ragas 原生英文 Prompt（LLM 自动理解中文回答），
    "zh_direct":      将原生 Prompt 直译为中文，
    "zh_optimized":   中文场景优化 Prompt——增加显式约束和中文语义提示
}

# 评估流程
for variant_name, prompt_config in prompt_variants.items():
    scores = run_ragas_eval(test_cases_20, prompt=prompt_config)
    # 与人工标注对比
    correlation = pearsonr(scores, human_labels)
    results[variant_name] = {
        "pearson_r": correlation,
        "mae": mean_absolute_error(scores, human_labels)
    }

# 选择 Pearson r 最高的方案作为 V1 评分 Prompt
best_prompt = max(results, key=lambda k: results[k]["pearson_r"])
```

**优化的中文 Prompt 关键改进点**（相比直译）：
- 明确要求评分 LLM 使用中文进行推理，而非英文内部推理后输出中文分数
- 在 faithfulness 中，增加「逐句比对」指令：逐句检查 answer 中的每个声明是否被 context 中的原文支持
- 在 answer_relevancy 中，要求生成的反向问题必须与原问题使用**相同的语言风格和领域术语**

**V1 最低投入**：
- 人工标注 20 条 case（约 2-3 小时）
- 运行 3 种 Prompt 的对比实验（自动化，约 5 分钟）
- 选择最优方案投入使用

**人工校准流程**：
- 每次 RAGAS 自动评估后，随机抽取 **20%** 的评估结果进行人工复核
- 对比指标：人工评分 vs RAGAS 评分之间的相关系数（目标 Pearson r > 0.80）
- 若相关系数 < 0.75，说明评估 LLM 评分失准，需调整 prompt 或换用更大的中文模型
- 人工校准结果写入 `eval_calibration` 表，用于持续监控评估管道质量

**V2 完善方向**：
- 训练专用的中文 RAG 评分模型（基于人工标注数据微调 Qwen2.5-7B）
- 将评分 prompt 从通用英文模板改写为中文场景优化的评估模板

---

## 九、灾备方案：多组件备份矩阵

> 对应 Critique #9：数据备份与灾备未提及

### 9.1 多组件备份矩阵

| 组件 | 数据类型 | 备份方式 | 备份频率 | RPO | RTO | 存储位置 |
|------|------|------|------|:---:|:---:|------|
| **PostgreSQL** | 元数据/用户/计费 | `pg_dump` 逻辑备份 + WAL 连续归档 | 每日全量 + 实时 WAL | < 1min | < 30min | MinIO / S3 异地 |
| **Milvus** | 向量数据 | `milvus-backup` 工具 + COS/S3 | 每日增量 + 每周全量 | < 24h | < 2h | S3 兼容存储 |
| **MinIO** | 原始文档 | Bucket 版本控制 + Mirror 到异地 MinIO | 实时同步 | < 1min | < 5min | 异地 MinIO 实例 |
| **Redis** | 缓存/会话/队列 | RDB 快照 + AOF 持久化 | 每小时 RDB + 实时 AOF | < 1h | < 15min | 本地 + S3 |
| **Elasticsearch** | BM25 索引 | Snapshot API → S3 | 每日增量 | < 24h | < 1h | S3 兼容存储 |

### 9.2 RPO/RTO 目标分级

| 等级 | RPO | RTO | 适用场景 | 月成本增量 |
|:---:|:---:|:---:|------|:---:|
| **基础**（V1） | < 24h | < 4h | 非关键业务，单区域部署 | ~¥500 |
| **标准**（V2） | < 1h | < 1h | 企业级 SLA，主备切换 | ~¥3,000 |
| **高级**（V3） | < 1min | < 5min | 金融/医疗等高合规行业，异地多活 | ~¥15,000+ |

### 9.3 低成本起步方案（V1 最小实现）

```bash
# PostgreSQL：每日 pg_dump + 上传至 MinIO
0 2 * * * pg_dump -Fc rag_platform | mc pipe minio/backups/pg/$(date +\%Y\%m\%d).dump

# Milvus：每周全量备份至 MinIO
0 3 * * 0 milvus-backup create -n weekly_backup_$(date +\%Y\%m\%d)

# MinIO：开启 Bucket 版本控制
mc version enable minio/rag-documents

# Redis：启用 AOF + 每小时 RDB
# redis.conf: save 3600 1, appendonly yes
```

### 9.4 V2 完善方向

- PostgreSQL 升级为流复制（Streaming Replication）主备架构
- Milvus 备份引入增量备份，减少全量备份窗口
- 定期灾备演练（每季度一次），验证 RTO 达标

---

## 十、合规加固：分级脱敏 + PIPL 适配

> 对应 Critique #10：合规性与审计日志隐私 —— 审计日志记录完整问答可能涉及敏感信息

### 10.1 三级脱敏策略

| 级别 | 适用数据 | 脱敏方式 | 示例 |
|:---:|------|------|------|
| **L1 可逆脱敏** | 审计日志（需可追溯） | AES-256 加密 + 访问控制 | 手机号 `138****1234` → `AES_ENC(13812341234)` |
| **L2 不可逆脱敏** | 展示层/分析报表 | 哈希 + 掩码 | 身份证 `110101****1234` → `sha256(110101199001011234)` |
| **L3 完全删除** | 不必要存储的 PII | 正则匹配 → 替换为 `<REDACTED>` | 「我叫张三」→「我叫 `<PERSON>`」 |

### 10.2 Presidio + 自定义规则的混合方案

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class AuditLogDesensitizer:
    """审计日志脱敏管道"""
    
    def __init__(self):
        # Presidio 引擎：识别通用 PII（人名、手机、邮箱、身份证）
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # 自定义规则：行业特定敏感信息
        self.custom_patterns = [
            (r'\b\d{6}\b', '<EMPLOYEE_ID>'),       # 工号
            (r'\b(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{4}\b', '<ORG_CODE>'),  # 组织机构代码
        ]
    
    def desensitize(self, text: str, level: str = "L2") -> str:
        # Step 1: Presidio 自动检测
        results = self.analyzer.analyze(
            text=text,
            language="zh",
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "ID_NUMBER"]
        )
        text = self.anonymizer.anonymize(text, results).text
        
        # Step 2: 自定义规则补充
        for pattern, replacement in self.custom_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
```

### 10.3 PIPL 合规清单

| # | 合规要求 | 实现方式 | 状态 |
|:---:|------|------|:---:|
| 1 | 告知-同意 | 注册页明确告知数据收集范围、目的、期限 | 需前端配合 |
| 2 | 最小必要原则 | 审计日志仅记录问答 ID 和脱敏摘要，完整原文加密存储 | V1 实现 |
| 3 | 数据主体权利 | 提供 API 支持用户查询/更正/删除其个人数据 | V2 实现 |
| 4 | 数据出境评估 | 默认所有数据存储于境内节点，跨境传输需单独审批 | 架构保证 |
| 5 | 安全事件报告 | 数据泄露 72 小时内通知主管部门和受影响用户 | 需建立流程 |
| 6 | 合规审计 | 每年度第三方合规审计，保留审计报告 | 运营层面 |
| 7 | 数据保护官 | 指定 DPO 联系人并在隐私政策中公示 | 组织层面 |

### 10.4 V1 最小实现

- 审计日志写入前强制经过 `AuditLogDesensitizer` 脱敏管道
- L2 级别：用户问答原文经 Presidio 脱敏后存储
- 敏感字段（手机号、身份证、邮箱）AES-256 加密 + 访问权限控制
- 隐私政策页面 + 注册时同意勾选框

### 10.5 V2 完善方向

- 支持租户自定义脱敏规则（如特定行业的自定义 PII 模式）
- 数据主体权利 API：查询我的数据、导出我的数据、删除我的数据
- 自动生成 PIPL 合规报告

### 10.6 落地风险：可逆脱敏的密钥管理

L1 可逆脱敏使用 AES-256 加密敏感字段。若加密密钥与应用代码一起存储在配置文件中并进入 Git 仓库，**一旦代码仓库或配置文件泄露，脱敏即完全失效**——加密退化为编码，任何人都可以解密还原原始数据。

**V1 最低安全标准**：

```python
import os
from cryptography.fernet import Fernet

class KeyManager:
    """密钥管理——V1 最低安全标准"""
    
    def __init__(self):
        # 密钥仅从环境变量读取，严禁写在代码或配置文件中
        self.encryption_key = os.environ.get("AUDIT_LOG_ENCRYPTION_KEY")
        if not self.encryption_key:
            raise RuntimeError("AUDIT_LOG_ENCRYPTION_KEY 环境变量未设置")
        
        # .gitignore 中排除任何包含密钥的文件
        # .env 文件符号链接到 Kubernetes Secret 或 Docker Secret
        
    def encrypt(self, plaintext: str) -> bytes:
        f = Fernet(self.encryption_key)
        return f.encrypt(plaintext.encode())
    
    def decrypt(self, ciphertext: bytes) -> str:
        f = Fernet(self.encryption_key)
        return f.decrypt(ciphertext).decode()
```

**V1 强制规范**：

| 规范 | 说明 |
|------|------|
| 密钥不入 Git | `.env` / `secrets.yaml` / 任何含密钥的文件加入 `.gitignore` |
| 密钥分离存储 | 生产环境密钥通过 Kubernetes Secret / Docker Secret 注入 |
| 访问审计 | 密钥访问操作写入独立审计日志（不依赖应用日志） |
| 定期轮换 | 每 90 天轮换一次加密密钥；旧密钥保留 30 天用于历史数据解密 |
| 代码仓库扫描 | CI 管道集成 `git-secrets` 或 `truffleHog`，阻止密钥被误提交 |

**密钥轮换对历史数据的影响——重加密机制**：

90 天轮换一次密钥、旧密钥仅保留 30 天意味着：轮换后第 31 天起，30 天前用旧密钥加密的历史数据将**无法解密**。L1 可逆脱敏的历史审计日志同样面临此问题。

**V1 重加密策略**：

```python
class KeyRotationManager:
    """密钥轮换 + 历史数据重加密"""
    
    KEY_LIFETIME_DAYS = 90
    OLD_KEY_RETENTION_DAYS = 30
    
    def rotate_keys(self):
        """90 天密钥轮换流程"""
        new_key = Fernet.generate_key()
        old_key = current_encryption_key
        
        # 1. 新密钥立即生效（新数据用新密钥加密）
        set_current_key(new_key)
        
        # 2. 后台任务：将最近 30 天的历史数据用新密钥重加密
        reencrypt_historical_data.delay(
            since=datetime.utcnow() - timedelta(days=30),
            old_key=old_key,
            new_key=new_key
        )
        
        # 3. 30 天后废弃旧密钥（此时所有历史数据已完成重加密）
        schedule_key_destruction(old_key_id, delay_days=30)
    
    def decrypt_with_fallback(self, ciphertext: bytes) -> str:
        """解密时自动尝试当前密钥 → 旧密钥 → 错误"""
        for key_version in [self.current_key, self.old_key]:
            try:
                f = Fernet(key_version)
                return f.decrypt(ciphertext).decode()
            except Exception:
                continue
        raise DecryptionError("数据加密时间超过保留期限，无法解密")

# Celery 任务：历史数据重加密
@app.task
def reencrypt_historical_data(since, old_key, new_key):
    """重加密最近 30 天内的审计日志"""
    batch_size = 1000
    offset = 0
    while True:
        records = AuditLog.query.filter(
            AuditLog.created_at >= since,
            AuditLog.encryption_key_version == old_key.fingerprint()
        ).limit(batch_size).offset(offset).all()
        
        if not records:
            break
        
        for record in records:
            # 旧密钥解密 → 新密钥加密
            plaintext = decrypt_with_key(record.encrypted_field, old_key)
            record.encrypted_field = encrypt_with_key(plaintext, new_key)
            record.encryption_key_version = new_key.fingerprint()
            record.reencrypted_at = datetime.utcnow()
        
        db.session.commit()
        offset += batch_size
```

**数据保留策略明确化**：

| 数据类型 | 加密状态 | 保留期限 | 到期处理 |
|------|------|------|------|
| 审计日志（L1 可逆） | AES-256 加密 | 90 天（默认，可配置） | 到期自动删除；需延长则重加密 |
| 脱敏后日志（L2） | SHA-256 哈希 | 365 天 | 用于分析报表，到期删除原始数据 |
| 密钥历史 | - | 轮换后保留 30 天 | 到期安全销毁（密钥覆写删除） |

**V2 完善方向**：
- 引入密钥管理服务（**HashiCorp Vault** 或阿里云 KMS / 腾讯云 KMS）
- KMS 自动轮换密钥，应用无需重启即可获取新密钥
- 密钥访问需经审批流程，所有操作留审计 Trail
- 敏感数据分级加密：不同级别的数据使用不同密钥（如 PII 数据 vs 一般业务数据）

---

## 十一、API 治理：版本化与兼容性策略

> 新增议题：SaaS 平台 API 一旦上线即不可随意变更，需要明确的版本化和兼容性策略。

### 11.1 URL 版本化方案

采用 **URL 前缀版本化**（`/api/v1/...`），这是 REST API 最主流、最直观的方案：

```
# 版本化后的端点示例
POST   /api/v1/kbs/{kb_id}/query        # 问答
POST   /api/v1/kbs/{kb_id}/documents    # 文档上传
GET    /api/v1/kbs/{kb_id}/documents    # 文档列表
GET    /api/v1/kbs/{kb_id}/documents/{doc_id}/status  # 文档状态
POST   /api/v1/kbs/{kb_id}/documents/{doc_id}/retry   # 重试失败页面
GET    /api/v1/tenants/{tenant_id}/usage # 用量查询
```

### 11.2 API Gateway 灰度发布

在 API Gateway（Kong / APISIX）层配置路由规则：

```yaml
# APISIX 路由配置示例：10% 流量灰度至 v2
routes:
  - uri: /api/v1/kbs/*
    upstream: v1-backend-service
    weight: 90
  - uri: /api/v2/kbs/*
    upstream: v2-backend-service
    weight: 10
```

### 11.3 向后兼容策略

| 变更类型 | 兼容性 | 处理方式 |
|------|:---:|------|
| 新增字段（response） | ✅ 兼容 | 直接添加，旧客户端忽略未知字段 |
| 新增可选参数（request） | ✅ 兼容 | 默认值兜底 |
| 废弃字段 | ⚠️ 需过渡 | 保留 6 个月 + `Deprecation: true` 响应头 + 文档标注 |
| 删除字段 | ❌ 破坏性 | 仅在新主版本（/v2/）中删除 |
| 修改字段类型 | ❌ 破坏性 | 新版本新建字段，保留旧字段至 EOL |

**响应头通知机制**：
```python
# FastAPI 中间件：自动为废弃字段添加 Deprecation 响应头
@app.middleware("http")
async def deprecation_header(request, call_next):
    response = await call_next(request)
    deprecated_fields = get_deprecated_fields(request.url.path)
    if deprecated_fields:
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = "Sat, 01 Dec 2025 00:00:00 GMT"  # EOL 日期
        response.headers["Link"] = '</api/v2/docs#migration>; rel="deprecation"'
    return response
```

### 11.4 OpenAPI 规范 + CI 同步检查

```python
# FastAPI 原生支持自动生成 OpenAPI 文档
app = FastAPI(
    title="Enterprise RAG API",
    version="1.0.0",
    docs_url="/api/docs",        # Swagger UI
    redoc_url="/api/redoc",      # ReDoc
    openapi_url="/api/openapi.json"
)
```

```yaml
# CI 检查：确保 OpenAPI 文档与代码实现同步
# .github/workflows/api-check.yml
- name: Verify OpenAPI spec
  run: |
    python -c "from app.main import app; import json; json.dumps(app.openapi())" > /tmp/current_spec.json
    diff <(jq -S . docs/openapi.json) <(jq -S . /tmp/current_spec.json)
    if [ $? -ne 0 ]; then
      echo "❌ OpenAPI 文档过期，请运行 'make generate-openapi' 更新"
      exit 1
    fi
```

### 11.5 V1 最小实现

- 所有端点使用 `/api/v1/` 前缀
- FastAPI 自动生成 OpenAPI 文档，部署到 `/api/docs`
- CI 管道添加 OpenAPI spec 同步检查
- API 变更记录在 `CHANGELOG.md`，遵循 Semantic Versioning

---

## 十二、质量保障：系统级测试策略

> 新增议题：多组件协同的 RAG 系统需要完整的测试金字塔。

### 12.1 测试金字塔

```
         ╱  E2E ╲           手动探索 + 混沌测试
        ╱ 集成测试 ╲         Docker Compose 全链路
       ╱  单元测试  ╲        pytest 覆盖率 >80%
      ─────────────────
```

### 12.2 单元测试（L0）

```python
# tests/unit/test_parsers.py
class TestPDFParser:
    def test_parse_normal_pdf(self):
        """正常 PDF 解析 → 返回文本 + 表格"""
        result = PDFParser().parse("fixtures/sample.pdf")
        assert len(result.pages) == 5
        assert result.pages[0].text.startswith("第一季度报告")
    
    def test_parse_empty_pdf(self):
        """空 PDF → 返回空列表，不抛异常"""
        result = PDFParser().parse("fixtures/empty.pdf")
        assert result.pages == []
    
    def test_parse_corrupted_pdf(self):
        """损坏 PDF → 抛出 DocumentParseError"""
        with pytest.raises(DocumentParseError):
            PDFParser().parse("fixtures/corrupted.pdf")

# tests/unit/test_chunking.py
class TestChunkingService:
    def test_chunk_id_deterministic(self):
        """相同输入 → chunk_id 可复现"""
        id1 = generate_chunk_id("doc_001", 3)
        id2 = generate_chunk_id("doc_001", 3)
        assert id1 == id2
    
    def test_chunk_size_respects_limit(self):
        """chunk 不超过 max_tokens"""
        chunks = ChunkingService().chunk(long_text, max_tokens=512)
        assert all(c.token_count <= 512 for c in chunks)
```

**覆盖率要求**：核心模块（parsers, chunking, embedding, hybrid_retriever）≥ 80%。

### 12.3 集成测试（L1）—— Docker Compose 全链路

```yaml
# docker-compose.test.yml
services:
  test-runner:
    build: .
    environment:
      - ES_HOST=elasticsearch
      - MILVUS_HOST=milvus
      - MINIO_ENDPOINT=minio:9000
      - REDIS_HOST=redis
    depends_on:
      - elasticsearch
      - milvus
      - minio
      - redis
    command: pytest tests/integration/ -v
```

```python
# tests/integration/test_upload_to_search_e2e.py
class TestUploadToSearchE2E:
    """端到端：文档上传 → 分块 → 双写 → 检索 → LLM 生成"""
    
    def test_full_pipeline(self, kb_id, test_pdf_path):
        # 1. 上传文档
        doc = upload_document(kb_id, test_pdf_path)
        assert doc.status == "PROCESSING"
        
        # 2. 等待处理完成
        doc = wait_for_status(doc.id, "READY", timeout=120)
        assert doc.chunk_count > 0
        
        # 3. 验证 ES 和 Milvus 双写一致
        es_count = es_client.count_chunks(kb_id, doc.id)
        mv_count = milvus_client.count_entities(kb_id, doc.id)
        assert es_count == mv_count, f"ES({es_count}) ≠ Milvus({mv_count})"
        
        # 4. 执行查询
        result = query_kb(kb_id, "第一季度营收是多少？")
        assert result.answer is not None
        assert len(result.sources) > 0
        
        # 5. 验证回答忠实度
        assert result.faithfulness_score > 0.7
```

### 12.4 RAG 质量回归测试（CI 阻断）

```yaml
# CI 管道：每次 PR 自动跑 RAGAS 评估
- name: RAG Quality Regression Test
  run: |
    python -m pytest tests/regression/test_rag_quality.py --ragas-dataset=fixtures/eval_dataset.json
  # 评分下降 >5% → 阻断合并
```

```python
# tests/regression/test_rag_quality.py
def test_rag_quality_no_regression():
    """RAGAS 评分 vs 基线：下降 >5% → FAIL"""
    baseline = load_baseline("fixtures/ragas_baseline.json")  # 上次发布时的评分
    current = run_ragas_eval("fixtures/eval_dataset.json")
    
    for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
        delta = (current[metric] - baseline[metric]) / baseline[metric]
        assert delta > -0.05, (
            f"{metric} 下降 {abs(delta)*100:.1f}%（基线 {baseline[metric]:.3f} → 当前 {current[metric]:.3f}）"
        )
```

### 12.5 性能基准测试

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class RAGUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def search_knowledge_base(self):
        self.client.post("/api/v1/kbs/test-kb/query", json={
            "question": "2024年第三季度营收增长率是多少？"
        })
```

```bash
# 运行压测
locust -f tests/performance/locustfile.py \
  --headless \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --host http://localhost:8000
```

**关注指标**：`rag_search_latency_p99` < 2000ms，`api_error_rate_5xx` < 1%。

### 12.6 混沌测试（V2+）

```yaml
# 使用 Chaos Mesh 注入故障
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: kill-milvus-test
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces: [rag-platform]
    labelSelectors:
      app: milvus
  duration: "60s"
  scheduler:
    cron: "@every 7d"
```

故障注入场景：
| 故障 | 预期行为 | 验证方式 |
|------|------|------|
| Kill Milvus Pod | 纯 ES BM25 检索降级 | 检查响应中 `degraded: true` |
| 断开 Redis 网络 | 语义缓存 miss → 直连 LLM | 检查 cache_hit_rate 下降但无 5xx |
| vLLM OOM Kill | 自动切换云端 API 兜底 | 检查 `fallback_to_cloud_api` 指标 |

### 12.7 V1 最小实现

- pytest 骨架 + CI 集成（GitHub Actions）
- 核心 Parser + Chunking 单元测试覆盖率 >80%
- 1 条集成测试：上传 PDF → 等待 READY → 查询 → 验证答案
- Locust 基准压测脚本（记录基线延迟）

---

## 十三、国际化前瞻：多语言支持的低成本架构预留

> 新增议题：当前为中文场景，但 SaaS 未来可能服务外资企业或出海客户。

### 13.1 知识库语言标记

```sql
-- knowledge_bases 表增加 language 字段
ALTER TABLE knowledge_bases ADD COLUMN language VARCHAR(10) DEFAULT 'auto';
-- 可选值: 'zh', 'en', 'ja', 'ko', 'auto'
```

| language 值 | Embedding 模型 | ES 分词器 | Reranker |
|:---:|------|------|------|
| `zh` | BGE-small-zh-v1.5 | `ik_smart` | BGE-Reranker-v2-m3 |
| `en` | all-MiniLM-L6-v2 | `standard` | BGE-Reranker-v2-m3 (多语言) |
| `ja` | multilingual-e5-base | `kuromoji` | BGE-Reranker-v2-m3 |
| `auto` | 文本语言检测 → 动态路由 | 同上 | 同上 |

### 13.2 按语言动态路由

```python
class LanguageAwareEmbeddingRouter:
    """根据知识库语言选择 Embedding 模型"""
    
    MODELS = {
        "zh": "BAAI/bge-small-zh-v1.5",
        "en": "sentence-transformers/all-MiniLM-L6-v2",
        "multilingual": "intfloat/multilingual-e5-base",
    }
    
    def get_model(self, kb_language: str):
        if kb_language in self.MODELS:
            return self.load_or_cache(self.MODELS[kb_language])
        return self.load_or_cache(self.MODELS["multilingual"])
    
    def detect_language(self, text: str) -> str:
        """轻量语言检测（不依赖外部 API）"""
        from langdetect import detect
        return detect(text)
```

### 13.3 ES 索引模板按语言配置

```json
{
  "index_patterns": ["chunks-*"],
  "settings": {
    "analysis": {
      "analyzer": {
        "zh_analyzer": { "type": "custom", "tokenizer": "ik_smart" },
        "en_analyzer": { "type": "custom", "tokenizer": "standard" },
        "default_analyzer": { "type": "custom", "tokenizer": "standard" }
      }
    }
  },
  "mappings": {
    "properties": {
      "language": { "type": "keyword" },
      "text_content": {
        "type": "text",
        "analyzer": "default_analyzer",
        "fields": {
          "zh": { "type": "text", "analyzer": "zh_analyzer" },
          "en": { "type": "text", "analyzer": "en_analyzer" }
        }
      }
    }
  }
}
```

### 13.4 RAGAS 多语言测试集

```
fixtures/eval_datasets/
├── zh_eval.json      # 50 条中文测试集
├── en_eval.json      # 50 条英文测试集（V2）
└── ja_eval.json      # 30 条日文测试集（V2）
```

### 13.5 前端 i18n 框架

```javascript
// 使用 react-i18next，V1 仅做框架预留
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

i18n.use(initReactI18next).init({
    resources: {
        zh: { translation: { /* 中文文案 */ } },
        en: { translation: { /* English copy */ } },
    },
    lng: 'zh',           // 默认中文
    fallbackLng: 'zh',
});
```

### 13.6 V1 最小实现

- `knowledge_bases` 表添加 `language` 字段（默认 `auto`）
- Embedding 路由预留多模型接口（当前仅加载 BGE 中文模型）
- 前端引入 `react-i18next` 框架，当前仅中文文案，但结构支持多语言扩展
- RAGAS 测试集目录结构预留多语言位置

**关键原则**：V1 只做**架构预留**，不实现实际的多语言功能。但 schema、路由接口、前端框架的选择都考虑了未来的多语言扩展，避免后期大规模重构。

### 修订前（Plan B v1）

```
┌──────────────────────────────────────────────────┐
│                   API Gateway                      │
├──────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ 文档解析  │ │ 问答服务  │ │ 多租户管理 (简易)  │  │
│  └────┬─────┘ └────┬─────┘ └────────┬─────────┘  │
│       │             │               │             │
│  ┌────┴─────────────┴───────────────┴──────────┐  │
│  │                数据层                         │  │
│  │  PostgreSQL │ Milvus(kb_id分区) │ MinIO │ Redis │  │
│  │  Kafka │ ClickHouse (无 ES、无 Prometheus)    │  │
│  └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
❌ BM25 实现不明  ❌ 代码漏写可致数据泄露  ❌ 无监控告警
```

### 修订后（Plan B v2）

```
┌──────────────────────────────────────────────────────────┐
│                      API Gateway                          │
│              + TenantContextMiddleware                    │
├──────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌───────────┐ │
│  │ 文档解析  │ │ 问答服务  │ │ 多租户管理  │ │ 计费引擎   │ │
│  │+ Celery  │ │+ Tier路由 │ │+ RBAC      │ │+ 全因子   │ │
│  │+ 状态机  │ │+ GPTCache│ │+ 双重校验  │ │+ Token统计│ │
│  └────┬─────┘ └────┬─────┘ └─────┬──────┘ └─────┬─────┘ │
│       │             │             │               │       │
│  ┌────┴─────────────┴─────────────┴───────────────┴─────┐ │
│  │                    数据层                             │ │
│  │  PostgreSQL │ Milvus(Partition Key) │ MinIO(版本控制)  │ │
│  │  Redis │ Kafka │ ClickHouse │ Elasticsearch ⭐        │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              可观测性层 ⭐                             │ │
│  │  Prometheus + Grafana + Alertmanager + LangFuse       │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              质量与合规层 ⭐                           │ │
│  │  RAGAS 评估 │ 👍/👎 反馈 │ Presidio 脱敏 │ 灾备      │ │
│  └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 附录 B：V1 → V2 → V3 演进路线图

```
V1 筑牢基础（当前-2个月）          V2 能力增强（3-6个月）         V3 智能自治（6-12个月）
─────────────────────────      ───────────────────────      ───────────────────────
P0 交付：                       P1 深化：                      P2 成熟：
✅ ES BM25 混合检索              ✅ Collection per Tenant       ✅ Agentic RAG 自治
✅ Partition Key 隔离             ✅ 14B 中间 Tier               ✅ A/B 实验平台
✅ Prometheus + 告警              ✅ 云端 API 弹性兜底            ✅ 异地多活灾备
✅ 全因子计费                     ✅ RAGAS 持续评估              ✅ 检索参数自动优化
P1 起步：                       ✅ 灾备升级至标准级              ✅ PIPL 合规审计自动化
✅ Celery 容错                    ✅ 细粒度脱敏策略
✅ 7B/72B 双 Tier
✅ 👍/👎 反馈收集
✅ PG 逻辑备份
✅ Presidio L2 脱敏

里程碑：10 个付费租户             里程碑：100 个付费租户          里程碑：500+ 租户，99.9% SLA
```

---

## 附录 C：技术栈选型速查表

| 领域 | 组件 | 版本 | 用途 | 替代方案 |
|------|------|------|------|------|
| **全文检索** | Elasticsearch | 8.15+ | BM25 关键词检索 | Milvus 2.5 原生 BM25 (V2) |
| **向量数据库** | Milvus | 2.4+ | 向量语义检索 + Partition Key 隔离 | Qdrant, Weaviate |
| **关系数据库** | PostgreSQL | 16 | 元数据/用户/计费 | MySQL 8.0 |
| **对象存储** | MinIO | RELEASE.2025-* | 原始文档存储 | AWS S3, 阿里云 OSS |
| **缓存** | Redis | 7.2+ | 语义缓存 + 会话 + 队列 Broker | KeyDB |
| **消息队列** | Kafka | 3.7+ | 事件驱动 + 异步解耦 | Redis Streams |
| **分析数据库** | ClickHouse | 24.x | 用量统计 + 审计日志聚合 | StarRocks |
| **LLM 推理** | vLLM + Ollama | latest | 模型推理引擎 (7B/14B/72B) | TGI, SGLang |
| **Embedding** | BGE-small-zh-v1.5 | - | 768d 中文向量化 | BGE-large-zh, m3e-base |
| **Reranker** | BGE-Reranker-v2-m3 | - | Cross-Encoder 精排 | Cohere Rerank API |
| **语义缓存** | GPTCache | 0.1.x | LLM 语义去重缓存 | LangChain Cache |
| **任务队列** | Celery | 5.4+ | 文档处理异步编排 | Dramatiq, Huey |
| **监控** | Prometheus + Grafana | 2.50+ / 10.x | 系统指标采集与可视化 | VictoriaMetrics |
| **Trace** | LangFuse | 2.x | LLM 调用链追踪与评估 | MLflow, Weights & Biases |
| **告警** | Alertmanager | 0.27+ | 告警规则 + 通知路由 | Grafana Alerting |
| **评估** | RAGAS | 0.2+ | RAG 质量离线评估 | DeepEval, TruLens |
| **脱敏** | Microsoft Presidio | 2.2+ | PII 自动检测与匿名化 | 自研正则规则 |
| **灾备** | milvus-backup | latest | Milvus 向量备份 | Velero (K8s 场景) |

---

> **文档版本**：v2.3
> **修订历史**：
> - v2.0（2025-06-04）：10 项 critique 逐条回应，首发
> - v2.1（同日）：增补 5 项落地细则
> - v2.2（同日）：深化 5 项操作细则
> - v2.3（同日）：新增 3 章系统工程议题 + 2 项章节内深化（API 版本化、测试金字塔、多语言预留、环境成本、RBAC 细粒度）
> **审核状态**：待用户刘泽文审核
> **下一步**：逐章确认技术选型，进入 V1 实施排期阶段
