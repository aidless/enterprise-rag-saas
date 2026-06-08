"""
Enterprise RAG SaaS — 数据库 Schema (SQLAlchemy ORM, V2.0)
============================================================
定义五张核心表的 ORM 模型，支持 SQLite(开发) / PostgreSQL(生产) 双引擎。

======================================================================
表结构设计哲学
======================================================================

五表关系:
  tenants ──1:N──▶ users ──1:N──▶ kb_members ◀──N:1── knowledge_bases
                                    (多对多关联表)          │
                                                            │ 1:N
                                                            ▼
                                                       documents
  tenants ─────────────────────────────────────▶ usage_logs

设计决策:
  1. 为什么 tenant_id 在 documents 表冗余存储？
     → 允许跨表查询时不用 JOIN knowledge_bases 表
     → 索引 idx_docs_tenant 直接加速按租户的分页查询

  2. 为什么 preferred_model 放在 knowledge_bases 而非 tenant 表？
     → 同一个租户的不同知识库可能需要不同的模型
     → 粒度更细：法律KB用72B，客服KB用7B

  3. 为什么用 SQLAlchemy 而非原生 SQL？
     → 开发阶段用 SQLite（零配置），生产切 PostgreSQL（一行配置）
     → ORM 自动做方言适配，不用写两套 DDL
     → relationship() 提供对象图导航，减少重复 JOIN 代码

  4. metadata_json 为什么用别名 "metadata"？
     → SQLAlchemy Base 类自带 metadata 属性，会和列名冲突
     → 用 Column("metadata", ...) + metadata_json 属性名解决

======================================================================
索引策略
======================================================================
  - kb_members: UNIQUE(kb_id, user_id) — 防止成员重复添加
  - users: UNIQUE(tenant_id, email) — 租户内邮箱唯一
  - documents: idx_docs_kb + idx_docs_tenant + idx_docs_status
  - usage_logs: idx_usage_tenant + idx_usage_metric + idx_usage_created

======================================================================
迁移策略
======================================================================
  V1: SQLite (DATABASE_URL=sqlite:///./data/enterprise.db)
  V2: PostgreSQL (DATABASE_URL=postgresql://rag:pass@host:5432/rag)
  切换: 只需改 DATABASE_URL 环境变量，代码零改动
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    ForeignKey, Text, JSON, create_engine, UniqueConstraint, Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session


class Base(DeclarativeBase):
    """SQLAlchemy ORM 基类 — 所有模型继承此类"""
    pass


# ============================================================================
# 1. Tenants 表 — 租户主表
# ============================================================================
class Tenant(Base):
    """
    租户主表

    字段设计:
      tier: free|basic|pro|enterprise → 决定 LLM tier、缓存开关、QPS 上限
      api_key: 非 JWT 方式认证时使用（仅开发环境）
      max_kbs/max_users: 软限制，前端校验（非数据库强制，便于升级套餐时调整）
    """
    __tablename__ = "tenants"

    id          = Column(String(64), primary_key=True)
    name        = Column(String(128), nullable=False)
    tier        = Column(String(20), default="free")
    status      = Column(String(20), default="active")     # active|suspended|deleted
    api_key     = Column(String(64), unique=True)
    max_kbs     = Column(Integer, default=5)
    max_users   = Column(Integer, default=10)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                          onupdate=lambda: datetime.now(timezone.utc))

    # 级联关系: 删除租户 → 自动删除其所有 users 和 kbs
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    kbs   = relationship("KnowledgeBase", back_populates="tenant", cascade="all, delete-orphan")


# ============================================================================
# 2. Users 表 — 租户内用户
# ============================================================================
class User(Base):
    """
    租户内用户表
    约束: (tenant_id, email) 联合唯一 → 同一租户内邮箱不能重复，跨租户可以
    """
    __tablename__ = "users"

    id          = Column(String(64), primary_key=True)
    tenant_id   = Column(String(64), ForeignKey("tenants.id"), nullable=False)
    email       = Column(String(128), nullable=False)
    name        = Column(String(64))
    role        = Column(String(20), default="member")       # admin|member
    status      = Column(String(20), default="active")
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_tenant_email"),
    )

    tenant         = relationship("Tenant", back_populates="users")
    kb_memberships = relationship("KBMember", back_populates="user", cascade="all, delete-orphan")


# ============================================================================
# 3. KnowledgeBases 表 — 知识库
# ============================================================================
class KnowledgeBase(Base):
    """
    知识库主表
    preferred_model: 租户可为每个 KB 指定不同的 LLM 模型（覆盖 tier 默认值）
    language: zh|en|auto → 影响 ES 分词器和 LLM Prompt 语言设置
    """
    __tablename__ = "knowledge_bases"

    id              = Column(String(64), primary_key=True)
    tenant_id       = Column(String(64), ForeignKey("tenants.id"), nullable=False)
    name            = Column(String(128), nullable=False)
    description     = Column(String(512))
    language        = Column(String(10), default="zh")
    chunk_size      = Column(Integer, default=500)
    preferred_model = Column(String(64), nullable=True)   # 租户自定义模型
    doc_count       = Column(Integer, default=0)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    tenant    = relationship("Tenant", back_populates="kbs")
    members   = relationship("KBMember", back_populates="kb", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="kb", cascade="all, delete-orphan")


# ============================================================================
# 4. KB_Members 表 — 知识库级 RBAC（多对多关联）
# ============================================================================
class KBMember(Base):
    """
    知识库成员关系表（多对多关联）
    角色: owner(全权限) > editor(上传+查询+导出) > viewer(仅查询)
    为什么不是 user 表加 kb_id 字段？→ 一个用户属于多个 KB，需要中间表
    """
    __tablename__ = "kb_members"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    kb_id      = Column(String(64), ForeignKey("knowledge_bases.id"), nullable=False)
    user_id    = Column(String(64), ForeignKey("users.id"), nullable=False)
    role       = Column(String(20), default="viewer")
    granted_by = Column(String(64))    # 授权人
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("kb_id", "user_id", name="uq_kb_user"),  # 防止重复添加
    )

    kb   = relationship("KnowledgeBase", back_populates="members")
    user = relationship("User", back_populates="kb_memberships")


# ============================================================================
# 5. Documents 表 — 文档处理状态
# ============================================================================
class Document(Base):
    """
    文档处理状态表

    状态机: uploaded → processing → ready | partial_ready | failed
    failed_pages 使用 JSON 数组存储: [{"page_num": 3, "error": "页面无文本"}]
    为什么用 JSON 而非关联表？→ 单文档失败页面数通常 < 20，JSON 比关联表查询更简单
    """
    __tablename__ = "documents"

    id            = Column(String(64), primary_key=True)
    kb_id         = Column(String(64), ForeignKey("knowledge_bases.id"), nullable=False)
    tenant_id     = Column(String(64), nullable=False)   # 冗余字段，加速按租户过滤
    filename      = Column(String(256), nullable=False)
    file_path     = Column(String(512))
    status        = Column(String(20), default="uploaded")
    total_pages   = Column(Integer, default=0)
    success_pages = Column(Integer, default=0)
    failed_pages  = Column(JSON, default=list)
    chunk_count   = Column(Integer, default=0)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    kb = relationship("KnowledgeBase", back_populates="documents")

    __table_args__ = (
        Index("idx_docs_kb", "kb_id"),        # 按知识库查询
        Index("idx_docs_tenant", "tenant_id"), # 按租户过滤（冗余字段的价值）
        Index("idx_docs_status", "status"),    # 按状态筛选
    )


# ============================================================================
# 6. Usage Logs 表 — 计费数据
# ============================================================================
class UsageLog(Base):
    """
    用量日志表（全因子计费）
    metric: llm_input_tokens|llm_output_tokens|embedding_tokens|api_calls|storage_mb
    设计: 按事件记录而非按时间聚合 → 粒度细，可按需聚合到小时/天/月
    """
    __tablename__ = "usage_logs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id     = Column(String(64), nullable=False)
    metric        = Column(String(32), nullable=False)   # 指标类型
    value         = Column(Float, nullable=False)        # 用量数值
    metadata_json = Column("metadata", JSON, default=dict)  # SQLAlchemy 避开 Base.metadata
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_usage_tenant", "tenant_id"),
        Index("idx_usage_metric", "metric"),
        Index("idx_usage_created", "created_at"),
    )


# ============================================================================
# 数据库初始化工具
# ============================================================================
def init_db(database_url: str = "sqlite:///./data/enterprise.db", echo: bool = False):
    """
    创建所有表并返回 SQLAlchemy Engine

    用法:
      # 开发环境 (SQLite)
      engine = init_db()

      # 生产环境 (PostgreSQL)
      engine = init_db("postgresql://rag:pass@postgres:5432/enterprise_rag")

    为什么默认 SQLite？
      → 零配置启动，新人 clone 代码就能跑
      → 生产环境通过环境变量覆盖，代码零改动
    """
    engine = create_engine(database_url, echo=echo)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine) -> Session:
    """获取数据库会话 — 调用方负责 close()"""
    return Session(engine)
