"""
Enterprise RAG SaaS — Embedding 向量嵌入服务 (V2.0)
====================================================
将文本转换为数值向量，供 Milvus 检索使用。

模型方案:
  AI-ModelScope/bge-small-zh-v1.5
  - 输出维度: 512 (可控，与 MILVUS_DIM 一致)
  - 语言: 中英文
  - 来源: ModelScope (国内镜像，下载更快)

Mock 模式:
  当 EMBEDDING_USE_MOCK=true 或模型加载失败时，使用 MD5 哈希生成伪向量
  用于 CI/CD 测试环境，无需下载 500MB+ 模型文件
"""

import numpy as np
from typing import List

from src.config.settings import Settings


class EmbeddingService:
    """
    文本向量化服务 — 封装 BGE 中文模型
    提供 encode (批量) 和 encode_query (单条) 两个接口
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.dim = settings.MILVUS_DIM  # 向量维度
        self._model = None

        # 非 mock 模式 → 加载真实模型
        if not settings.EMBEDDING_USE_MOCK:
            self._init_model()

    def _init_model(self):
        """
        初始化 BGE Embedding 模型
        通过 ModelScope 下载（国内镜像，速度快）
        """
        try:
            from modelscope import snapshot_download
            from sentence_transformers import SentenceTransformer

            # 下载模型到本地缓存目录
            model_dir = snapshot_download(
                self.settings.EMBEDDING_MODEL,
                cache_dir="./data/models",
            )
            self._model = SentenceTransformer(model_dir)
            self.dim = self._model.get_sentence_embedding_dimension()
            print(f"[Embedding] BGE 模型加载完成，维度: {self.dim}")
        except Exception as e:
            print(f"[Embedding] 模型加载失败: {e}，使用 mock 模式")
            self._model = None

    # ====== 批量编码 ======

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本 → 向量
        自动 L2 归一化，返回 float 列表
        """
        if self._model is None:
            return self._mock_encode(texts)

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,   # L2 归一化 (内积 = cosine)
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """单条查询文本 → 向量"""
        return self.encode([query])[0]

    # ====== Mock 编码（测试用） ======

    def _mock_encode(self, texts: List[str]) -> List[List[float]]:
        """
        Mock 编码 — MD5 哈希 + 归一化
        用于无模型时的功能测试，不可用于生产检索

        算法: md5(text) → byte[16] → float[16] → L2 normalize
        """
        import hashlib
        results = []
        for text in texts:
            h = hashlib.md5(text.encode()).digest()
            vec = [(b / 255.0) for b in h[: self.dim]]
            norm = np.linalg.norm(vec)
            vec = [v / norm for v in vec]
            results.append(vec)
        return results
