# Enterprise RAG SaaS Platform — 产品说明文档

> 版本：V3.0 | 最后更新：2026-06-08 | 部署地址：47.98.106.182

---

## 一、产品定位

**Enterprise RAG SaaS** 是一个面向企业级客户的多租户 RAG（Retrieval-Augmented Generation）知识库平台。用户上传企业文档（PDF、Markdown、TXT、CSV），系统自动分块、索引、向量化，形成可检索的知识库；用户通过自然语言提问，系统混合检索最相关片段，结合 LLM 生成带有源引用的精准答案。

**一句话描述**：把你的企业文档变成可对话的 AI 知识库。

---

## 二、核心能力

### 2.1 检索增强生成（RAG）

| 能力 | 说明 |
|------|------|
| 混合检索 | BM25 全文检索 + 向量语义检索 + RRF 融合 |
| 多级重排 | BGE-Reranker / Cohere / Jina 多后端 |
| SSE 流式返回 | Server-Sent Events 实时输出生成内容 |
| 源引用追溯 | 每条答案附带来源 chunk 和置信度 |
| 查询改写 | Agentic 自动改写用户模糊查询 |

### 2.2 多租户安全隔离

| 能力 | 说明 |
|------|------|
| Collection per Tenant | 每个租户独立 Milvus 向量空间 |
| BM25 租户隔离 | 索引键 `{tenant_id}:{kb_id}` 复合分区 |
| JWT + API Key 双认证 | 身份验证 + 过期/篡改检测 |
| KB 级 RBAC | owner/editor/viewer 三级权限 |
| ResponseValidator | 中间件双重校验，拦截跨租户数据泄露 |
| PII 自动脱敏 | 姓名/手机/身份证 Presidio 自动识别掩码 |
| 审计日志 | L1-L3 三级脱敏存储 |

### 2.3 文档处理管线

| 阶段 | 能力 |
|------|------|
| 上传 | 多格式支持：PDF / Markdown / TXT / CSV |
| 解析 | Unstructured 引擎 + PyPDF2 降级 |
| 分块 | 可配置 chunk_size / overlap + 4 策略对比 |
| 状态机 | 6 状态：UPLOADED → PROCESSING → CHUNKED → READY / PARTIAL_READY / FAILED |
| 容错 | PARTIAL_READY 模式 + 失败页面重试 |
| 异步 | Celery 后台任务 + 死信队列 |

### 2.4 质量保障体系

| 能力 | 说明 |
|------|------|
| RAGAS 自动评分 | faithfulness / relevancy / context_recall / harmfulness |
| 用户反馈闭环 | 👍/👎 评分 → 写回阴影模式 → 驱动优化 |
| 影子模式 | 生产流量双写 logging，离线评估不阻塞用户 |
| A/B 实验平台 | 多实验并行 + 粘性分流 + t-检验 + 自动推广 |
| DSPy 自动优化 | 离线 prompt 优化 + few-shot 示例挖掘 |
| Hard Negative 挖掘 | 反馈数据自动生成边界型拒答训练样本 |

### 2.5 可观测性

| 能力 | 说明 |
|------|------|
| Prometheus 指标 | 10 项核心 RAG 指标 + 自定义业务指标 |
| 结构化日志 | JSON 格式 + request_id 追踪 |
| 健康探针 | /health（服务状态）+ /ready（K8s readiness） |
| Celery Flower | 任务队列实时监控（:5555） |

---

## 三、技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        客户端层                              │
│  React SPA / REST API / SSE Streaming / WebSocket           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      API 网关层                              │
│  FastAPI + Uvicorn / Rate Limiting / JWT Auth / CORS         │
└───┬───────────┬───────────┬───────────┬─────────────────────┘
    │           │           │           │
┌───▼───┐ ┌────▼────┐ ┌───▼───┐ ┌────▼────┐
│ 检索   │ │ LLM    │ │ 文档  │ │ 评估    │
│ 引擎   │ │ 推理   │ │ 处理  │ │ 实验    │
│        │ │        │ │        │ │         │
│ Milvus │ │ DeepS  │ │ Celer  │ │ RAGAS   │
│ 向量   │ │ eek    │ │ y 异步 │ │ 评分    │
│ BM25   │ │ API    │ │        │ │         │
│ Rerank │ │        │ │        │ │         │
└───┬───┘ └────┬────┘ └───┬───┘ └────┬────┘
    │          │          │          │
┌───▼──────────▼──────────▼──────────▼─────────────────────────┐
│                       数据 & 存储层                           │
│  PostgreSQL(主库) │ Redis(缓存) │ Milvus Lite(向量) │ SQLite(影子) │
└──────────────────────────────────────────────────────────────┘
```

### 技术栈明细

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 框架 | FastAPI + Uvicorn | 0.110+ | API 服务 |
| 向量数据库 | Milvus Lite | 2.4+ | 向量存储与检索 |
| 全文检索 | rank-bm25 + jieba | - | 本地 BM25 引擎 |
| 关系数据库 | PostgreSQL | 16 | 元数据/审计/用户 |
| 缓存 | Redis | 7 | 语义缓存 + 会话 |
| 异步任务 | Celery | 5.3+ | 文档处理/评估 |
| 监控 | Celery Flower | 2.0+ | 任务队列监控 |
| LLM | DeepSeek API | deepseek-chat | 云端推理 |
| 重排序 | BGE-Reranker / Cohere | - | 检索结果精排 |
| 评估 | RAGAS | 0.1+ | 自动质量评分 |
| 实验 | scipy + pyyaml | - | A/B 实验统计 |
| 容器化 | Docker + Compose | - | 部署编排 |

---

## 四、版本演进

### V1 — 功能闭环（已完成 ✅）

- 多租户隔离（Partition Key）
- 混合检索（BM25 + 向量 + RRF）
- 文档处理（上传/分块/状态机/PARTIAL_READY）
- 可观测性（Prometheus/Grafana/Health）
- 全因子计费（LLM/Embedding/API/Storage）
- 锦上添花7项（异常处理/限流/日志/配置校验/API文档/CI/灾备）
- **25+ 源文件 | 10 测试 | 6 API 端点**

### V2 — 生产级升级（已完成 ✅）

- PostgreSQL Schema（6 表完整建模）
- Redis 语义缓存（MD5 + cosine > 0.85 两层）
- Celery 异步管线（文档处理/评估/备份）
- Tier 路由（7B/14B/72B + 云端 API 弹性兜底）
- SSE 流式生成
- Reranker 多后端（BGE/Cohere/Jina）
- Collection per Tenant 安全升级
- PII 脱敏 + 审计日志
- RAGAS 评估 + 用户反馈闭环
- Agentic 查询改写
- **50+ 源文件 | 18 测试 | 20 API 端点**

### V3 — 智能自治（已完成 ✅）

- 影子模式数据积累（798 条高质量标注）
- RAGAS LLM 评分双模（本地 Ollama + 云端 GPT-4o-mini）
- Cohere Reranker API 集成
- 4 策略 Chunking 对比实验
- A/B 实验平台（粘性分流/t-检验/自动推广）
- DSPy 离线优化（8 few-shot + 2810 char prompt）
- Hard Negative 高价值负样本挖掘
- ECS 容器化部署上线
- **54 源文件 | 18 测试 | 20 API 端点**

---

## 五、API 端点一览

### 认证 & 多租户

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/tenants/auth/token` | JWT 登录获取 token |
| POST | `/api/v1/tenants/auth/apikey` | 创建 API Key |
| GET | `/api/v1/tenants/{id}/usage` | 租户用量查询 |

### 知识库 & 文档

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/kbs` | 创建知识库 |
| GET | `/api/v1/kbs` | 列出知识库 |
| POST | `/api/v1/kbs/{kb_id}/documents/upload` | 上传文档 |
| GET | `/api/v1/kbs/{kb_id}/documents` | 文档列表 |
| GET | `/api/v1/kbs/{kb_id}/documents/{doc_id}/status` | 文档处理状态 |
| POST | `/api/v1/kbs/{kb_id}/documents/{doc_id}/retry` | 重试失败文档 |

### 检索 & 问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/kbs/{kb_id}/query` | 标准问答 |
| POST | `/api/v1/kbs/{kb_id}/query/stream` | SSE 流式问答 |
| GET | `/api/v1/kbs/{kb_id}/cache/stats` | 缓存命中率 |
| POST | `/api/v1/kbs/{kb_id}/feedback` | 提交 👍/👎 反馈 |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务健康检查 |
| GET | `/ready` | K8s 就绪检查 |
| GET | `/metrics` | Prometheus 指标 |

---

## 六、部署信息

### 当前环境

| 项目 | 值 |
|------|------|
| 服务器 | 阿里云 ECS, 47.98.106.182 |
| 配置 | 2C8G, Ubuntu 22.04, 40G SSD |
| 数据库 | PostgreSQL 16 (容器内) |
| 缓存 | Redis 7 (容器内) |
| LLM | DeepSeek API (deepseek-chat) |
| 部署方式 | Docker Compose |

### 部署架构

```
ECS 2C8G (47.98.106.182)
├── rag-api (uvicorn:8000)       ← FastAPI 主服务
├── rag-worker (celery)          ← 异步任务
├── rag-flower (:5555)           ← 任务监控
├── rag-postgres (:5432)         ← 元数据库
├── rag-redis (:6379)            ← 缓存层
└── data/                        ← 持久化数据卷
    ├── milvus.db                ← 向量数据库
    └── shadow.db                ← 影子模式日志
```

---

## 七、定价模型

| 等级 | 月费 | QPS | 知识库数 | 文档数 | 缓存 | 模型 |
|------|------|-----|---------|--------|------|------|
| Free | ¥0 | 2 | 1 | 10 | ❌ | 7B |
| Basic | ¥299 | 10 | 3 | 100 | ✅ | 14B |
| Pro | ¥999 | 50 | 10 | 500 | ✅ | 72B |
| Enterprise | ¥定制 | ∞ | ∞ | ∞ | ✅ | 定制 |

**计费维度**：LLM Token / Embedding Token / Rerank 调用 / API 请求数 / 存储量

---

## 八、安全合规

| 机制 | 说明 |
|------|------|
| JWT + API Key | 双因素认证 |
| Collection per Tenant | 向量级物理隔离 |
| ResponseValidator | 响应二次校验防泄露 |
| PII 脱敏 | 姓名/手机/身份证自动掩码 |
| 审计日志 | 三级脱敏 + 数据保留策略 |
| 限流 | Free 2qps / Pro 50qps |
| RBAC | owner/editor/viewer 三级 |

---

## 九、下一步建议

1. **前端开发** — 基于 React + MUI 构建管理控制台和问答界面
2. **Nginx 反向代理** — 配置 HTTPS + 域名 + 静态资源缓存
3. **日志持久化** — 接入 ELK / Loki 集中日志管理
4. **CI/CD** — GitHub Actions 自动构建推送 + ECS 自动部署
5. **性能优化** — 引入 GPTCache 语义缓存 / vLLM 本地推理
