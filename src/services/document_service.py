"""
Enterprise RAG SaaS — 文档处理服务 (V3.0)
===========================================
管理文档从上传到索引的完整生命周期，实现容错状态机和多格式解析器。

======================================================================
状态机设计哲学
======================================================================

六状态流转:
  UPLOADED → PROCESSING → {CHUNKED → READY}          (全成功)
                         → {PARTIAL_CHUNKED → PARTIAL_READY}  (部分成功)
                         → FAILED                      (全失败)

为什么要有 PARTIAL_READY 状态？
  问题: PDF 有扫描页 + 文本页的混合文档
  方案: 成功页索引，失败页记录 → 用户可检索已成功部分 + 手动重试失败页
  对比: 全有全无 → 用户上传一个 200 页 PDF，1 页扫描件导致全部不可检索
  结论: PARTIAL_READY 比 "全部失败" 的用户体验好 100 倍

为什么要有 DEAD 状态？
  → 重试 3 次后仍失败 → 标记 DEAD，不再自动重试
  → 避免无限重试浪费资源

为什么用内存存储 (self._records) 而非数据库？
  V1 开发阶段：快速迭代，避免每次改 schema 都要 migration
  V2 生产阶段：切换到 PostgreSQL (Document 表)，通过 PostgresKBMemberStore

======================================================================
解析器路由设计
======================================================================
按文件扩展名分发到对应解析器:
  .txt / .md → _parse_txt()  → 按 2000 字符分页
  .pdf       → _parse_pdf()  → PyPDF2 逐页提取文本
  .csv       → _parse_csv()  → 每 50 行一页

为什么用 PyPDF2 而非 Unstructured？
  → PyPDF2 纯 Python，零编译依赖
  → 对于文本 PDF 足够用；复杂文档(表格/图片)需要 Unstructured
  → 渐进升级策略: V1 PyPDF2 → V2 Unstructured

======================================================================
分块策略
======================================================================
简单滑动窗口: start=0 → end=chunk_size → next_start=end-overlap
为什么要 overlap？
  → 避免在句子中间切断语义
  → 例如: "第三季度营收增长" 在 chunk A 末尾，chunk B 开头 → 两个 chunk 语义都不完整
  → overlap 保证关键信息至少在两个相邻 chunk 中都出现
"""

import os, uuid, logging
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class DocStatus(str, Enum):
    """
    文档处理状态枚举 — 状态机节点

    状态转换规则:
      uploaded → processing (开始处理)
      processing → ready (全部成功)
      processing → partial_ready (部分成功)
      processing → failed (全部失败)
      partial_ready → processing (用户点击重试)
      failed → processing (用户重新上传)
    """
    UPLOADED        = "uploaded"        # 已上传，等待处理
    PROCESSING      = "processing"      # 正在解析/分块/索引
    DONE            = "done"            # 解析完成（V1内部状态）
    PARTIAL         = "partial"         # 部分解析成功（V1内部状态）
    FAILED          = "failed"          # 解析失败
    CHUNKED         = "chunked"         # 分块完成（内部状态）
    PARTIAL_CHUNKED = "partial_chunked" # 部分分块（内部状态）
    READY           = "ready"           # 已就绪，可检索 ✅
    PARTIAL_READY   = "partial_ready"   # 部分就绪，已成功的可检索 ⚠️
    DEAD            = "dead"            # 重试耗尽，不再自动处理 ❌


@dataclass
class PageResult:
    """单页解析结果"""
    page_num: int
    success: bool
    text: str = ""
    error: str = ""
    confidence: float = 1.0  # 解析置信度（PDF 文本提取 ≈ 0.9，OCR ≈ 0.7）


@dataclass
class DocumentRecord:
    """文档元数据 + 处理状态"""
    doc_id: str
    filename: str
    kb_id: str
    tenant_id: str
    status: DocStatus = DocStatus.UPLOADED
    total_pages: int = 0
    success_pages: int = 0
    failed_pages: List[dict] = field(default_factory=list)  # [{"page_num":1,"error":"..."}]
    chunk_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()


class DocumentService:
    """
    文档处理服务 — 解析 + 分块 + 索引的完整管线

    处理流程:
      1. 验证文件格式
      2. 按扩展名路由解析器
      3. 统计成功/失败页
      4. 成功页 → 分块 → 向量化 → Milvus + BM25
      5. 更新文档状态
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}

    def __init__(self, settings: Settings, rag_service=None):
        self.settings = settings
        self.rag_service = rag_service  # 注入 RAG 服务用于索引
        self._records: Dict[str, DocumentRecord] = {}  # V1 内存，V2 换 PostgreSQL

    # ===================================================================
    # upload() — 同步上传（API 直接调用）
    # ===================================================================
    def upload(self, file_path: str, filename: str, kb_id: str = "default",
               tenant_id: str = "default") -> DocumentRecord:
        """文档上传入口 — 同步处理整个管线"""
        doc_id = str(uuid.uuid4())[:12]
        ext = os.path.splitext(filename)[1].lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {self.SUPPORTED_EXTENSIONS}")

        record = DocumentRecord(doc_id=doc_id, filename=filename,
                                kb_id=kb_id, tenant_id=tenant_id,
                                status=DocStatus.PROCESSING)
        self._records[doc_id] = record

        try:
            # 1. 解析文档
            pages = self._parse(file_path, ext)
            if not pages:
                record.status = DocStatus.FAILED
                return record

            record.total_pages = len(pages)

            # 2. 统计成功/失败页
            success_pages = [p for p in pages if p.success]
            failed_pages  = [p for p in pages if not p.success]
            record.success_pages = len(success_pages)
            record.failed_pages = [{"page_num": p.page_num, "error": p.error} for p in failed_pages]

            # 3. 分块
            chunks = self._chunk(success_pages, filename)
            if not chunks:
                record.status = DocStatus.FAILED
                return record

            # 4. 向量化 + 索引（如果有 RAG 服务）
            if self.rag_service:
                self.rag_service.index_document(
                    chunks=chunks, doc_name=filename, kb_id=kb_id, tenant_id=tenant_id)

            record.chunk_count = len(chunks)

            # 5. 更新状态
            record.status = DocStatus.PARTIAL_READY if failed_pages else DocStatus.READY

        except Exception as e:
            logger.error(f"文档处理失败: {filename} - {e}")
            record.status = DocStatus.FAILED
            record.failed_pages.append({"page_num": 0, "error": str(e)})

        record.updated_at = datetime.utcnow().isoformat()
        return record

    # ===================================================================
    # upload_sync() — Celery Worker 调用版本
    # ===================================================================
    def upload_sync(self, file_path, filename, kb_id="default", tenant_id="default", doc_id=None):
        """Celery 异步任务入口 — 与 upload() 逻辑相同但支持指定 doc_id"""
        doc_id = doc_id or str(uuid.uuid4())[:12]
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {ext}")

        record = DocumentRecord(doc_id=doc_id, filename=filename,
                                kb_id=kb_id, tenant_id=tenant_id, status=DocStatus.PROCESSING)
        self._records[doc_id] = record

        try:
            pages = self._parse(file_path, ext)
            if not pages:
                record.status = DocStatus.FAILED; return record

            record.total_pages = len(pages)
            success_pages = [p for p in pages if p.success]
            failed_pages  = [p for p in pages if not p.success]
            record.success_pages = len(success_pages)
            record.failed_pages = [{"page_num": p.page_num, "error": p.error} for p in failed_pages]

            chunks = self._chunk(success_pages, filename)
            if not chunks:
                record.status = DocStatus.FAILED; return record

            if self.rag_service:
                self.rag_service.index_document(chunks=chunks, doc_name=filename,
                                                 kb_id=kb_id, tenant_id=tenant_id)
            record.chunk_count = len(chunks)
            record.status = DocStatus.PARTIAL_READY if failed_pages else DocStatus.READY
        except Exception as e:
            logger.error(f"文档处理失败: {filename} - {e}")
            record.status = DocStatus.FAILED
            record.failed_pages.append({"page_num": 0, "error": str(e)})

        record.updated_at = datetime.utcnow().isoformat()
        return record

    # ===================================================================
    # retry_failed_pages() — PARTIAL_READY 文档的补救机制
    # ===================================================================
    def retry_failed_pages(self, doc_id: str, file_path: str) -> DocumentRecord:
        """
        重试 PARTIAL_READY 文档的失败页面
        为什么要保留这个功能？
          → 扫描件 PDF 在首次处理时 PyPDF2 提取不到文字
          → 用户可能后续升级到了支持 OCR 的解析器
          → 点击重试 → 用新解析器处理之前失败的页面
        """
        record = self._records.get(doc_id)
        if not record:
            raise ValueError(f"文档不存在: {doc_id}")
        if record.status != DocStatus.PARTIAL_READY:
            raise ValueError(f"仅 PARTIAL_READY 状态可重试，当前: {record.status}")

        record.status = DocStatus.PROCESSING
        ext = os.path.splitext(record.filename)[1].lower()

        try:
            pages = self._parse(file_path, ext)
            failed_page_nums = {p["page_num"] for p in record.failed_pages}
            retry_pages = [p for p in pages if p.page_num in failed_page_nums]
            still_failed, recovered = [], []

            for p in retry_pages:
                (recovered if p.success else still_failed).append(
                    p if p.success else {"page_num": p.page_num, "error": p.error})

            # 恢复的页面重新分块+索引
            if recovered:
                chunks = self._chunk(recovered, record.filename)
                if self.rag_service:
                    self.rag_service.index_document(chunks=chunks, doc_name=record.filename,
                                                     kb_id=record.kb_id, tenant_id=record.tenant_id)
                record.chunk_count += len(chunks)
                record.success_pages += len(recovered)

            record.failed_pages = still_failed
            record.status = DocStatus.PARTIAL_READY if still_failed else DocStatus.READY
        except Exception as e:
            logger.error(f"重试失败: {doc_id} - {e}")
            record.status = DocStatus.PARTIAL_READY

        record.updated_at = datetime.utcnow().isoformat()
        return record

    # ===================================================================
    # 查询接口
    # ===================================================================
    def get_status(self, doc_id: str) -> Optional[DocumentRecord]:
        return self._records.get(doc_id)

    def list_documents(self, kb_id: str, tenant_id: str = "default") -> List[DocumentRecord]:
        return [r for r in self._records.values()
                if r.kb_id == kb_id and r.tenant_id == tenant_id]

    # ===================================================================
    # 解析器路由 + 各格式实现
    # ===================================================================
    def _parse(self, file_path: str, ext: str) -> List[PageResult]:
        """按扩展名路由到对应解析器"""
        parsers = {".txt": self._parse_txt, ".md": self._parse_txt,
                   ".pdf": self._parse_pdf, ".csv": self._parse_csv}
        return parsers.get(ext, lambda f: [])(file_path)

    def _parse_txt(self, file_path: str) -> List[PageResult]:
        """TXT/Markdown 解析 — 每 2000 字符一页"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if not text.strip():
                return []
            pages = []
            for i in range(0, len(text), 2000):
                chunk = text[i:i+2000].strip()
                if chunk:
                    pages.append(PageResult(page_num=len(pages)+1, success=True, text=chunk))
            return pages
        except Exception as e:
            logger.error(f"TXT 解析失败: {e}")
            return []

    def _parse_pdf(self, file_path: str) -> List[PageResult]:
        """PDF 解析 — PyPDF2 逐页文本提取"""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append(PageResult(page_num=i+1, success=True, text=text.strip()))
                    else:
                        pages.append(PageResult(page_num=i+1, success=False,
                                                 error="页面无文本内容（可能是扫描件，需 OCR）"))
                except Exception as e:
                    pages.append(PageResult(page_num=i+1, success=False,
                                             error=f"页面解析异常: {str(e)[:100]}"))
            return pages
        except Exception as e:
            logger.error(f"PDF 解析失败: {e}")
            return [PageResult(page_num=1, success=False, error=f"PDF 文件解析失败: {str(e)[:100]}")]

    def _parse_csv(self, file_path: str) -> List[PageResult]:
        """CSV 解析 — 每 50 行一页（跳过 header）"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            if len(lines) <= 1:
                return []
            pages = []
            for i in range(1, len(lines), 50):
                chunk = "".join(lines[i:i+50]).strip()
                if chunk:
                    pages.append(PageResult(page_num=len(pages)+1, success=True, text=chunk))
            return pages
        except Exception as e:
            logger.error(f"CSV 解析失败: {e}")
            return [PageResult(page_num=1, success=False, error=str(e)[:100])]

    def _chunk(self, pages: List[PageResult], filename: str) -> List[Dict]:
        """
        滑动窗口分块算法
        参数: CHUNK_SIZE=500, CHUNK_OVERLAP=50
        效果: 每块 500 字符，相邻块重叠 50 字符
        为什么 overlap=50？→ 保证关键名词（通常 2-4 个中文字符）能出现在相邻块中
        """
        chunks = []
        chunk_idx = 0
        for page in pages:
            text, start = page.text, 0
            while start < len(text):
                end = min(start + self.settings.CHUNK_SIZE, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "chunk_id": f"{filename}_p{page.page_num}_c{chunk_idx}",
                        "content": chunk_text, "doc_name": filename,
                        "page_num": page.page_num,
                    })
                    chunk_idx += 1
                start = end - self.settings.CHUNK_OVERLAP if end < len(text) else len(text)
        return chunks
