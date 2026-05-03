"""
RAG 检索模块（编排层）。

职责：
1. 加载 ``knowledge/docs/`` 下的 Markdown 知识文件。
2. 将每篇文档切分成带有重叠的短段。
3. 通过 ``Embedder`` 接口生成向量（本地 sentence-transformers / 远程 DashScope 等可切换）。
4. 通过 ``VectorStore`` 接口持久化与检索（FAISS / Qdrant 可切换）。
5. 根据用户查询返回 top-k 最相关的片段（含得分、来源、分类），查询向量带 LRU 缓存。
6. 用 ``embedder.identifier + 文件 mtime + 切分配置`` 计算指纹，让 store 自行判断是否需要重建。

阶段进度：
- P1：抽 ``Embedder`` 接口，支持本地 / DashScope；查询缓存。
- P2：抽 ``VectorStore`` 接口，支持 FAISS / Qdrant；payload 携带 category 元数据。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from vector_stores import (
    ChunkRecord,
    StoredHit,
    VectorStore,
    build_vector_store,
    infer_category,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class RetrievedChunk:
    """检索命中的单个知识片段（对外契约保持稳定）。"""

    source: str
    title: str
    content: str
    score: float
    category: str = "general"


# ---------------------------------------------------------------------------- Embedder 抽象


@runtime_checkable
class Embedder(Protocol):
    """统一的向量化接口。

    ``identifier`` 必须能唯一标识"模型 + 关键参数"，参与缓存指纹计算；
    一旦 identifier 变化，索引会自动重建。
    """

    identifier: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """返回 shape=(len(texts), dim)、已 L2 归一化的 float32 向量。"""
        ...


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class LocalSentenceTransformerEmbedder:
    """基于本地 sentence-transformers 的 embedder。

    支持：
    - 普通 BGE / M3E 等：``query_prompt=None``，document/query 走同一 ``encode``。
    - Qwen3-Embedding：建议 ``query_prompt_name="query"``，doc 端不传 prompt。
      （配套使用 ``model_kwargs={"torch_dtype": "auto"}`` 可让 GPU/CPU 自动选择 dtype。）
    """

    def __init__(
        self,
        model_name: str,
        *,
        query_prompt_name: Optional[str] = None,
        query_prompt: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.query_prompt_name = query_prompt_name
        self.query_prompt = query_prompt
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = max(1, int(batch_size))
        self._model = None  # 懒加载
        self._dim: Optional[int] = None

    @property
    def identifier(self) -> str:
        # 把 prompt 配置纳入指纹：切换 prompt 需要重建索引（doc embedding 不变，但缓存判定避免歧义）
        suffix = ""
        if self.query_prompt_name:
            suffix = f"::qpn={self.query_prompt_name}"
        elif self.query_prompt:
            suffix = f"::qp={hashlib.md5(self.query_prompt.encode('utf-8')).hexdigest()[:8]}"
        return f"local::{self.model_name}{suffix}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load_model()
        return int(self._dim or 0)

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers 未安装，无法使用本地 embedder。"
                "请执行 `pip install -r requirements.txt` 或切换到远程 embedder。"
            ) from exc
        logger.info("正在加载本地 embedding 模型：%s", self.model_name)

        kwargs: dict = {}
        if self.device:
            kwargs["device"] = self.device
        model_kwargs: dict = {}
        if self.torch_dtype:
            model_kwargs["torch_dtype"] = self.torch_dtype
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        try:
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except TypeError:
            # 兼容低版本 sentence-transformers（不支持 model_kwargs 参数）
            kwargs.pop("model_kwargs", None)
            self._model = SentenceTransformer(self.model_name, **kwargs)

        # sentence-transformers 5.x 把方法名改成 get_embedding_dimension；做一次平滑兼容
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or getattr(self._model, "get_sentence_embedding_dimension", None)
        )
        try:
            self._dim = int(get_dim()) if callable(get_dim) else None
        except Exception:
            self._dim = None

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([query], is_query=True)

    def _encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        if self._model is None:
            self._load_model()
        assert self._model is not None

        # 文档批量较大时启用进度条，避免用户看到日志静止以为程序卡死
        show_progress = (not is_query) and len(texts) >= 32

        encode_kwargs = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": show_progress,
        }
        if is_query:
            if self.query_prompt_name:
                encode_kwargs["prompt_name"] = self.query_prompt_name
            elif self.query_prompt:
                encode_kwargs["prompt"] = self.query_prompt

        vectors = self._model.encode(texts, **encode_kwargs)
        return np.asarray(vectors, dtype=np.float32)


class DashScopeEmbedder:
    """阿里云 DashScope embedding（OpenAI 兼容模式）。

    - 复用项目已经依赖的 ``openai`` SDK，不引入新依赖。
    - DashScope ``text-embedding-v3`` 默认 1024 维，单批最多 25 条文本。
    - 出于 cos 相似度的兼容性考虑，本地再做一次 L2 归一化。
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "text-embedding-v3"
    DEFAULT_DIM = 1024
    BATCH_SIZE = 25  # DashScope 单次请求上限

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "DashScope embedder 需要配置 DASHSCOPE_API_KEY，请在 .env 中设置。"
            )
        from openai import OpenAI

        self.model = model
        self._dim = int(dim)
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @property
    def identifier(self) -> str:
        return f"dashscope::{self.model}::dim{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = [t if t.strip() else " " for t in texts[start : start + self.BATCH_SIZE]]
            kwargs = {"model": self.model, "input": batch, "encoding_format": "float"}
            if self._dim and self._dim != self.DEFAULT_DIM:
                kwargs["dimensions"] = self._dim
            try:
                resp = self.client.embeddings.create(**kwargs)
            except Exception as exc:
                raise RuntimeError(f"DashScope embedding 调用失败：{exc}") from exc
            all_vectors.extend(item.embedding for item in resp.data)

        arr = np.asarray(all_vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"DashScope 返回向量形状异常：{arr.shape}")
        # 实际维度可能与配置不一致（例如服务端不支持 dimensions 时返回默认维度），以实际为准
        self._dim = arr.shape[1]
        return _l2_normalize(arr)


# ---------------------------------------------------------------------------- 检索器


class KnowledgeRetriever:
    """编排层：负责切分、嵌入、按指纹决定是否重建、查询缓存。

    线程安全：状态更新在 ``_lock`` 保护下进行；查询缓存独立加锁，避免阻塞构建。
    """

    def __init__(
        self,
        docs_dir: Path,
        embedder: Embedder,
        store: VectorStore,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        query_cache_size: int = 256,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.embedder = embedder
        self.store = store
        self.chunk_size = max(120, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size // 2))

        self._lock = threading.Lock()
        self._initialized = False

        self._query_cache_size = max(0, int(query_cache_size))
        self._query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._query_cache_lock = threading.Lock()

    # ------------------------------------------------------------------ 对外 API

    def ensure_index(self, force_rebuild: bool = False) -> int:
        """确保 store 中有可用索引，返回片段数量。"""
        with self._lock:
            fingerprint = self._fingerprint()
            if force_rebuild or self.store.needs_rebuild(fingerprint):
                self._rebuild_unlocked(fingerprint)
                logger.info(
                    "RAG 索引已重建：store=%s，共 %d 段",
                    self.store.describe(),
                    self.store.chunk_count(),
                )
            elif not self._initialized:
                logger.info(
                    "RAG 索引复用已有 store：%s，共 %d 段",
                    self.store.describe(),
                    self.store.chunk_count(),
                )
            self._initialized = True
            return self.store.chunk_count()

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
        category: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """返回 top-k 个与 query 最相关的片段。"""
        if not query or not query.strip():
            return []

        self.ensure_index()
        if self.store.chunk_count() == 0:
            return []

        query_vec = self._embed_query(query.strip())
        top_k = max(1, int(top_k))
        raw_hits = self.store.search(query_vec, top_k=top_k, category=category)

        results: list[RetrievedChunk] = []
        for hit in raw_hits:
            if hit.score < min_score:
                continue
            results.append(
                RetrievedChunk(
                    source=hit.record.source,
                    title=hit.record.title,
                    content=hit.record.content,
                    score=float(hit.score),
                    category=hit.record.category,
                )
            )
        return results

    # --------------------------------------------------------------- 查询缓存

    def _embed_query(self, query: str) -> np.ndarray:
        if self._query_cache_size <= 0:
            return self._call_query_embedder(query)

        with self._query_cache_lock:
            cached = self._query_cache.get(query)
            if cached is not None:
                self._query_cache.move_to_end(query)
                return cached

        vec = self._call_query_embedder(query)

        with self._query_cache_lock:
            self._query_cache[query] = vec
            self._query_cache.move_to_end(query)
            while len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)
        return vec

    def _call_query_embedder(self, query: str) -> np.ndarray:
        # 优先使用 embedder 自带的 ``embed_query``（例如 Qwen3-Embedding 用 query prompt）
        embed_query = getattr(self.embedder, "embed_query", None)
        if callable(embed_query):
            return embed_query(query)
        return self.embedder.embed([query])

    def _invalidate_query_cache(self) -> None:
        with self._query_cache_lock:
            self._query_cache.clear()

    # --------------------------------------------------------------- 内部工具

    def _rebuild_unlocked(self, fingerprint: str) -> None:
        records = self._load_records()
        if not records:
            logger.warning("RAG 知识库目录为空：%s", self.docs_dir)
            self.store.rebuild([], np.zeros((0, max(self.embedder.dim, 1)), dtype=np.float32), fingerprint)
            self._invalidate_query_cache()
            return

        texts = [self._format_for_embedding(rec) for rec in records]
        logger.info(
            "RAG 开始嵌入 %d 段知识（embedder=%s，CPU 上首次可能耗时数分钟，请耐心等待）",
            len(texts),
            self.embedder.identifier,
        )
        import time
        t0 = time.perf_counter()
        vectors = self.embedder.embed(texts)
        logger.info(
            "RAG 嵌入完成：%d 段，向量维度=%d，耗时 %.1fs",
            len(texts),
            int(vectors.shape[1]) if vectors.size else 0,
            time.perf_counter() - t0,
        )

        logger.info("RAG 正在写入向量库：%s", self.store.describe())
        t0 = time.perf_counter()
        self.store.rebuild(records, vectors, fingerprint)
        logger.info("RAG 向量库写入完成：耗时 %.1fs", time.perf_counter() - t0)
        self._invalidate_query_cache()

    def _load_records(self) -> list[ChunkRecord]:
        if not self.docs_dir.exists():
            return []
        records: list[ChunkRecord] = []
        files = sorted(self.docs_dir.glob("*.md"))
        for path in files:
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("读取知识文件失败：%s (%s)", path, exc)
                continue
            title = self._extract_title(raw, fallback=path.stem)
            category = infer_category(path.name)
            for piece in self._chunk_text(raw):
                records.append(
                    ChunkRecord(
                        source=path.name,
                        title=title,
                        content=piece,
                        category=category,
                    )
                )
        return records

    @staticmethod
    def _extract_title(text: str, fallback: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or fallback
            if stripped:
                return stripped[:40]
        return fallback

    def _chunk_text(self, text: str) -> list[str]:
        """基于段落 + 滑窗的简易中文友好切分。"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        buffer = ""
        for para in paragraphs:
            if not buffer:
                buffer = para
                continue
            if len(buffer) + len(para) + 1 <= self.chunk_size:
                buffer = f"{buffer}\n{para}"
                continue
            chunks.append(buffer)
            if self.chunk_overlap > 0 and len(buffer) > self.chunk_overlap:
                buffer = buffer[-self.chunk_overlap :] + "\n" + para
            else:
                buffer = para
        if buffer:
            chunks.append(buffer)

        final: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size * 1.5:
                final.append(chunk)
                continue
            start = 0
            step = self.chunk_size - self.chunk_overlap
            while start < len(chunk):
                final.append(chunk[start : start + self.chunk_size])
                start += step
        return final

    @staticmethod
    def _format_for_embedding(rec: ChunkRecord) -> str:
        return f"{rec.title}\n{rec.content}"

    # --------------------------------------------------------------- 缓存指纹

    def _fingerprint(self) -> str:
        payload = {
            "embedder": self.embedder.identifier,
            "store": self.store.backend_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files": [],
        }
        if self.docs_dir.exists():
            for path in sorted(self.docs_dir.glob("*.md")):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                payload["files"].append(
                    {
                        "name": path.name,
                        "size": stat.st_size,
                        "mtime": int(stat.st_mtime),
                    }
                )
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------- Embedder 工厂


def build_embedder(
    kind: str,
    *,
    local_model_name: str,
    dashscope_api_key: str,
    dashscope_model: str,
    dashscope_dim: int,
    dashscope_base_url: str,
    dashscope_timeout: float,
    qwen_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    qwen_query_prompt_name: Optional[str] = "query",
    qwen_device: Optional[str] = None,
    qwen_torch_dtype: Optional[str] = "auto",
    qwen_batch_size: int = 8,
) -> Embedder:
    """根据配置项构造 Embedder 实例。

    支持 kind：
    - ``dashscope``：调用阿里云 DashScope 远程 embedding
    - ``qwen-local``：本地千问 Qwen3-Embedding 系列（推荐毕设演示用）
    - ``local``：通用 sentence-transformers 加载（可指向 BGE / M3E / 任意 HF 模型）
    - 未知值回退到 ``local``
    """
    normalized = (kind or "local").strip().lower()
    if normalized == "dashscope":
        logger.info(
            "使用远程 DashScope embedder：model=%s, dim=%d", dashscope_model, dashscope_dim
        )
        return DashScopeEmbedder(
            api_key=dashscope_api_key,
            model=dashscope_model,
            dim=dashscope_dim,
            base_url=dashscope_base_url,
            timeout=dashscope_timeout,
        )
    if normalized in ("qwen", "qwen-local", "qwen_local"):
        logger.info(
            "使用本地千问 embedder：model=%s, dtype=%s, device=%s, query_prompt_name=%s",
            qwen_model_name,
            qwen_torch_dtype,
            qwen_device or "auto",
            qwen_query_prompt_name,
        )
        return LocalSentenceTransformerEmbedder(
            qwen_model_name,
            query_prompt_name=qwen_query_prompt_name,
            device=qwen_device,
            torch_dtype=qwen_torch_dtype,
            batch_size=qwen_batch_size,
        )
    if normalized != "local":
        logger.warning("未知的 RAG_EMBEDDER=%s，已回退到本地 embedder", kind)
    logger.info("使用本地 sentence-transformers embedder：%s", local_model_name)
    return LocalSentenceTransformerEmbedder(local_model_name)


# 向外重新导出 build_vector_store，保持 main.py 的导入面更小。
__all__ = [
    "Embedder",
    "KnowledgeRetriever",
    "RetrievedChunk",
    "build_embedder",
    "build_vector_store",
    "build_chat_system_prompt",
]


# ---------------------------------------------------------------------------- Prompt 拼接


_SNIPPET_MAX_CHARS = 700


def build_chat_system_prompt(base_prompt: str, hits: list[RetrievedChunk]) -> str:
    """在基础 system prompt 末尾拼接检索到的参考资料。

    设计点：
    - 片段不强制截断到很短（保持 ≤ ``_SNIPPET_MAX_CHARS`` 接近 chunk_size），让 LLM 拿到尽量
      完整的方法步骤，避免出现 "知道方法名但不知道怎么做" 的截断信息缺失。
    - 指令鼓励 "整合 / 展开" 而不是 "不要罗列"——后者会让 LLM 不敢系统化讲解具体方法，
      导致回答短且空泛。这里改成允许把多段知识融合成可执行步骤，但仍要求自然口语化。
    """
    if not hits:
        return base_prompt

    lines = [
        base_prompt.strip(),
        "",
        "以下是检索到的与用户当前话题相关的心理健康参考资料（仅供你作为知识背景，不是诊断结论）：",
    ]
    for idx, hit in enumerate(hits, start=1):
        snippet = hit.content.strip().replace("\n", " ")
        if len(snippet) > _SNIPPET_MAX_CHARS:
            snippet = snippet[:_SNIPPET_MAX_CHARS] + "…"
        lines.append(f"[{idx}] 《{hit.title}》（来源：{hit.source}）：{snippet}")
    lines.append("")
    lines.append(
        "请把上述资料中的方法、技巧、原理融合到你的回答里，用自然的口语表达，"
        "可以分段展开（先共情倾听 → 帮用户理解可能的原因或心理机制 → 给出 2~4 条"
        "具体可操作的步骤 / 练习），不要直接复述原文也不要引用编号，"
        "但要让用户真的能从你的回答里学到具体的方法。"
    )
    return "\n".join(lines)
