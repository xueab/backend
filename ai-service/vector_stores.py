"""
向量存储抽象层（P2 引入）。

将"如何持久化向量"与"如何检索"封装成 ``VectorStore`` 接口，
`KnowledgeRetriever` 只负责编排（chunking + embedding + 调用 store）。

当前提供两种实现：
- ``FaissVectorStore``：沿用 P1 之前的本地 FAISS + numpy 兜底逻辑，零外部依赖。
- ``QdrantVectorStore``：调用 Qdrant（远程或嵌入式），支持元数据过滤。

切换由 ``RAG_VECTOR_STORE`` 环境变量控制，缺省 / 失败时统一兜底到 FAISS。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


def _ensure_local_url_bypasses_proxy(url: Optional[str]) -> None:
    """把本地 Qdrant 主机追加到 NO_PROXY，防止系统代理（如 Clash/V2Ray）拦截本机请求。

    背景：在 Windows 上启用了系统代理时，httpx（qdrant-client 底层）会通过
    ``urllib.request.getproxies()`` 读取注册表里的 ``ProxyServer`` 设置，把发往
    ``127.0.0.1:6333`` 的请求也丢给代理，代理无法路由本地地址，遂返回空 body 的
    ``502 Bad Gateway``。该函数在客户端构造前补充 ``NO_PROXY``，让 httpx 跳过代理。

    仅对本地 host 生效；远程 Qdrant URL 不会被修改，避免影响生产环境的代理配置。
    """
    if not url:
        return
    try:
        parsed = urlparse(url)
    except ValueError:
        return
    host = (parsed.hostname or "").lower()
    if not host or host not in _LOCAL_HOSTS:
        return

    extras = ["127.0.0.1", "localhost", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        items = [s.strip() for s in existing.split(",") if s.strip()]
        merged = list(items)
        for extra in extras:
            if extra not in merged:
                merged.append(extra)
        os.environ[key] = ",".join(merged)


# ---------------------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class ChunkRecord:
    """对外暴露的知识片段记录（与 rag.RetrievedChunk 是同构的，不含 score）。

    放在本模块里是为了让 store 不反向依赖 rag.py。
    """

    source: str
    title: str
    content: str
    category: str = "general"


@dataclass(frozen=True)
class StoredHit:
    """store.search 的返回项，包含原始记录与分数。"""

    record: ChunkRecord
    score: float


# ---------------------------------------------------------------------------- 知识分类推断


# slug → category 的静态映射（与 RAG说明.md 中的分组保持一致）。
_CATEGORY_MAP: dict[str, str] = {
    # 基础理论与方法
    "cbt-basics": "foundations",
    "acceptance-act-basics": "foundations",
    "dbt-skills-overview": "foundations",
    "mindfulness-meditation": "foundations",
    "self-compassion": "foundations",
    "positive-psychology": "foundations",
    "meaning-and-purpose": "foundations",
    # 情绪与症状自助
    "anxiety-coping": "emotions",
    "panic-attacks": "emotions",
    "depression-self-help": "emotions",
    "seasonal-mood": "emotions",
    "emotion-regulation": "emotions",
    "anger-management": "emotions",
    "shame-and-guilt": "emotions",
    "grief-and-loss": "emotions",
    "loneliness-coping": "emotions",
    "social-anxiety": "emotions",
    "health-anxiety": "emotions",
    "ocd-tendencies": "emotions",
    "news-cycle-anxiety": "emotions",
    # 行为与习惯
    "breathing-relaxation": "habits",
    "sleep-hygiene": "habits",
    "nightmares": "habits",
    "morning-routine": "habits",
    "exercise-and-mood": "habits",
    "nutrition-and-mood": "habits",
    "mood-journaling": "habits",
    "procrastination": "habits",
    "perfectionism": "habits",
    "screen-balance": "habits",
    "addictive-behaviors": "habits",
    "adhd-self-awareness": "habits",
    # 关系与沟通
    "relationship-communication": "relationships",
    "romantic-relationship-health": "relationships",
    "breakup-recovery": "relationships",
    "family-conflict": "relationships",
    "holiday-and-family-gathering": "relationships",
    "friendship-quality": "relationships",
    "saying-no-skills": "relationships",
    "assertiveness": "relationships",
    "gaslighting-recognition": "relationships",
    "forgiveness-and-letting-go": "relationships",
    "helping-friend": "relationships",
    "caregiver-burden": "relationships",
    # 学业 / 工作 / 财务
    "stress-management": "work-life",
    "burnout-recovery": "work-life",
    "academic-stress": "work-life",
    "workplace-mental-health": "work-life",
    "workplace-bullying": "work-life",
    "financial-anxiety": "work-life",
    "impostor-syndrome": "work-life",
    "comparison-trap": "work-life",
    "life-transitions": "work-life",
    # 自我与身份
    "self-esteem": "self-identity",
    "body-image-acceptance": "self-identity",
    "highly-sensitive-person": "self-identity",
    "inner-child-and-childhood": "self-identity",
    # 创伤、慢病与专业资源
    "trauma-recovery-basics": "trauma-care",
    "chronic-illness-and-mood": "trauma-care",
    "therapy-first-session": "trauma-care",
    "psych-medication-basics": "trauma-care",
    "crisis-resources": "trauma-care",
}


def infer_category(source_filename: str) -> str:
    """根据 source 文件名（不含扩展名匹配）推断知识分类。未知则归为 ``general``。"""
    if not source_filename:
        return "general"
    stem = source_filename.rsplit(".", 1)[0].strip().lower()
    return _CATEGORY_MAP.get(stem, "general")


# ---------------------------------------------------------------------------- 接口


@runtime_checkable
class VectorStore(Protocol):
    """统一的向量库接口。"""

    backend_name: str

    def describe(self) -> str:
        """用于日志的人类可读描述。"""
        ...

    def chunk_count(self) -> int:
        """当前已加载 / 已写入的片段总数。"""
        ...

    def needs_rebuild(self, fingerprint: str) -> bool:
        """根据 fingerprint 判断是否需要重建索引。"""
        ...

    def rebuild(
        self,
        records: list[ChunkRecord],
        vectors: np.ndarray,
        fingerprint: str,
    ) -> None:
        """全量重建：清空旧数据 → 写入 records / vectors → 落盘指纹。"""
        ...

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        category: Optional[str] = None,
    ) -> list[StoredHit]:
        """向量检索；query_vec shape=(1, dim)。"""
        ...


# ---------------------------------------------------------------------------- FAISS 实现


try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - 仅在缺失 faiss 时触发
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


class FaissVectorStore:
    """FAISS + numpy 兜底实现。

    持久化文件位于 ``cache_dir`` 下：``meta.json`` / ``vectors.npy`` / ``index.faiss``。
    元数据中保存 records（含 category）和 fingerprint。
    """

    backend_name = "faiss"
    _INDEX_FILE = "index.faiss"
    _VECTOR_FILE = "vectors.npy"
    _META_FILE = "meta.json"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._lock = threading.Lock()
        self._records: list[ChunkRecord] = []
        self._vectors: Optional[np.ndarray] = None
        self._faiss_index = None
        self._fingerprint: Optional[str] = None
        # 启动时尝试加载缓存（不强制要求 fingerprint 匹配；matcher 在 needs_rebuild 里判断）
        self._try_load()

    # ----- 接口实现

    def describe(self) -> str:
        backend = "faiss" if _FAISS_AVAILABLE and self._faiss_index is not None else "numpy-fallback"
        return f"faiss(cache_dir={self.cache_dir}, backend={backend})"

    def chunk_count(self) -> int:
        return len(self._records)

    def needs_rebuild(self, fingerprint: str) -> bool:
        return self._fingerprint != fingerprint or not self._records or self._vectors is None

    def rebuild(
        self,
        records: list[ChunkRecord],
        vectors: np.ndarray,
        fingerprint: str,
    ) -> None:
        with self._lock:
            self._records = list(records)
            self._vectors = vectors.astype(np.float32) if vectors.size else None
            self._faiss_index = self._build_faiss(vectors) if vectors.size else None
            self._fingerprint = fingerprint
            self._save()

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        category: Optional[str] = None,
    ) -> list[StoredHit]:
        with self._lock:
            records = list(self._records)
            vectors = self._vectors
            faiss_index = self._faiss_index

        if not records or vectors is None:
            return []

        # FAISS 不内置过滤；为保证过滤场景下也能返回 top_k 个结果，需要扩大候选再回截
        oversample = top_k if category is None else max(top_k * 4, top_k + 10)

        if faiss_index is not None:
            scores, indices = faiss_index.search(query_vec.astype(np.float32), min(oversample, len(records)))
            score_row = scores[0]
            index_row = indices[0]
        else:
            sims = (vectors @ query_vec[0]).astype(np.float32)
            order = np.argsort(-sims)[: min(oversample, len(records))]
            score_row = sims[order]
            index_row = order

        hits: list[StoredHit] = []
        for raw_idx, raw_score in zip(index_row, score_row):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(records):
                continue
            record = records[idx]
            if category and record.category != category:
                continue
            hits.append(StoredHit(record=record, score=float(raw_score)))
            if len(hits) >= top_k:
                break
        return hits

    # ----- 私有

    @staticmethod
    def _build_faiss(vectors: np.ndarray):
        if not _FAISS_AVAILABLE or vectors.size == 0:
            return None
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
        index.add(vectors.astype(np.float32))
        return index

    def _save(self) -> None:
        if self._vectors is None or not self._records:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / self._VECTOR_FILE, self._vectors)
        meta = {
            "fingerprint": self._fingerprint,
            "records": [
                {
                    "source": r.source,
                    "title": r.title,
                    "content": r.content,
                    "category": r.category,
                }
                for r in self._records
            ],
        }
        (self.cache_dir / self._META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        if _FAISS_AVAILABLE and self._faiss_index is not None:
            faiss.write_index(  # type: ignore[attr-defined]
                self._faiss_index, str(self.cache_dir / self._INDEX_FILE)
            )

    def _try_load(self) -> None:
        meta_path = self.cache_dir / self._META_FILE
        vec_path = self.cache_dir / self._VECTOR_FILE
        if not meta_path.exists() or not vec_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        try:
            vectors = np.load(vec_path)
        except (OSError, ValueError):
            return
        records = [
            ChunkRecord(
                source=item.get("source", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                category=item.get("category") or infer_category(item.get("source", "")),
            )
            for item in meta.get("records", [])
        ]
        if len(records) != len(vectors):
            return

        self._records = records
        self._vectors = vectors.astype(np.float32)
        self._fingerprint = meta.get("fingerprint")
        self._faiss_index = None
        if _FAISS_AVAILABLE:
            index_path = self.cache_dir / self._INDEX_FILE
            if index_path.exists():
                try:
                    self._faiss_index = faiss.read_index(str(index_path))  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - 损坏时回退到内存重建
                    self._faiss_index = self._build_faiss(self._vectors)
            else:
                self._faiss_index = self._build_faiss(self._vectors)


# ---------------------------------------------------------------------------- Qdrant 实现


class QdrantVectorStore:
    """Qdrant 实现（远程 HTTP / 嵌入式 path 两种部署形态）。

    指纹 + 已写入片段数保存在 ``cache_dir/qdrant_meta.json``，
    用于跨进程判断是否需要重建（避免每次启动都重新嵌入）。
    """

    backend_name = "qdrant"
    _META_FILE = "qdrant_meta.json"
    _BATCH_SIZE = 128

    def __init__(
        self,
        collection: str,
        cache_dir: Path,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: float = 30.0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - 仅在未装依赖时触发
            raise RuntimeError(
                "未安装 qdrant-client。请执行 `pip install qdrant-client`，"
                "或在 .env 中将 RAG_VECTOR_STORE 设回 faiss。"
            ) from exc

        self._QdrantClient = QdrantClient
        self.collection = collection
        self.cache_dir = Path(cache_dir)
        self.url = url
        self.api_key = api_key or None
        self.path = path
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        self._lock = threading.Lock()

        # 嵌入式（path）和远程（url）二选一；同时给了 path 则优先嵌入式
        if path:
            self.client = QdrantClient(path=path, prefer_grpc=prefer_grpc, timeout=timeout)
            self._endpoint_label = f"path={path}"
        else:
            effective_url = url or "http://localhost:6333"
            _ensure_local_url_bypasses_proxy(effective_url)
            self.client = QdrantClient(
                url=effective_url,
                api_key=self.api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )
            self._endpoint_label = f"url={effective_url}"

        # 启动时只读取本地 meta；真正的 collection 是否存在留到 needs_rebuild 时一起判断
        self._meta = self._load_meta()

    # ----- 接口实现

    def describe(self) -> str:
        return f"qdrant(collection={self.collection}, {self._endpoint_label})"

    def chunk_count(self) -> int:
        return int(self._meta.get("chunk_count", 0))

    def needs_rebuild(self, fingerprint: str) -> bool:
        if self._meta.get("fingerprint") != fingerprint:
            return True
        if int(self._meta.get("chunk_count", 0)) <= 0:
            return True
        try:
            return not self.client.collection_exists(self.collection)
        except Exception as exc:
            logger.warning("Qdrant 连接异常，强制重建：%s", exc)
            return True

    def rebuild(
        self,
        records: list[ChunkRecord],
        vectors: np.ndarray,
        fingerprint: str,
    ) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not records or vectors.size == 0:
            with self._lock:
                self._purge_collection()
                self._meta = {"fingerprint": fingerprint, "chunk_count": 0, "dim": 0}
                self._save_meta()
            return

        dim = int(vectors.shape[1])

        with self._lock:
            self._purge_collection()
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

            points: list = []
            for idx, (record, vector) in enumerate(zip(records, vectors)):
                points.append(
                    PointStruct(
                        id=idx,
                        vector=vector.astype(np.float32).tolist(),
                        payload={
                            "source": record.source,
                            "title": record.title,
                            "content": record.content,
                            "category": record.category,
                        },
                    )
                )
                if len(points) >= self._BATCH_SIZE:
                    self.client.upsert(collection_name=self.collection, points=points)
                    points = []
            if points:
                self.client.upsert(collection_name=self.collection, points=points)

            self._meta = {
                "fingerprint": fingerprint,
                "chunk_count": len(records),
                "dim": dim,
                "collection": self.collection,
                "endpoint": self._endpoint_label,
            }
            self._save_meta()

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        category: Optional[str] = None,
    ) -> list[StoredHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if query_vec.size == 0:
            return []

        flt = None
        if category:
            flt = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        try:
            res = self.client.query_points(
                collection_name=self.collection,
                query=query_vec[0].astype(np.float32).tolist(),
                limit=top_k,
                query_filter=flt,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Qdrant 查询失败：%s", exc)
            return []

        hits: list[StoredHit] = []
        for point in res.points:
            payload = point.payload or {}
            record = ChunkRecord(
                source=payload.get("source", ""),
                title=payload.get("title", ""),
                content=payload.get("content", ""),
                category=payload.get("category") or infer_category(payload.get("source", "")),
            )
            hits.append(StoredHit(record=record, score=float(point.score)))
        return hits

    # ----- 私有

    def _purge_collection(self) -> None:
        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
        except Exception as exc:
            logger.warning("Qdrant 删除旧 collection 失败（继续创建新集合）：%s", exc)

    def _load_meta(self) -> dict:
        path = self.cache_dir / self._META_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_meta(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / self._META_FILE).write_text(
            json.dumps(self._meta, ensure_ascii=False), encoding="utf-8"
        )


# ---------------------------------------------------------------------------- 工厂


def build_vector_store(
    kind: str,
    *,
    cache_dir: Path,
    qdrant_collection: str,
    qdrant_url: str,
    qdrant_api_key: str,
    qdrant_path: str,
    qdrant_prefer_grpc: bool,
    qdrant_timeout: float,
) -> VectorStore:
    """根据配置构造 VectorStore；失败时回退到 FaissVectorStore。"""
    normalized = (kind or "faiss").strip().lower()
    if normalized == "qdrant":
        try:
            store = QdrantVectorStore(
                collection=qdrant_collection,
                cache_dir=cache_dir,
                url=qdrant_url or None,
                api_key=qdrant_api_key or None,
                path=qdrant_path or None,
                prefer_grpc=qdrant_prefer_grpc,
                timeout=qdrant_timeout,
            )
            logger.info("使用 Qdrant 向量库：%s", store.describe())
            return store
        except Exception as exc:
            msg = str(exc)
            logger.warning("Qdrant 初始化失败，回退到 FAISS：%s", exc)
            if "already accessed by another instance" in msg or "Storage folder" in msg:
                logger.warning(
                    "嵌入式 Qdrant（QDRANT_PATH）同一目录只能被一个进程独占。"
                    "若使用 uvicorn --reload，会同时存在监视进程与工作进程，易触发此错误。"
                    "处理方式任选其一：① 注释掉 QDRANT_PATH，改用 Docker/Qdrant 服务端 "
                    "（docker compose up -d qdrant）并设置 QDRANT_URL=http://localhost:6333；"
                    "② 本地调试嵌入式时请 uvicorn 不加 --reload；③ 确认无第二个 uvicorn / Python "
                    "在占用同一目录。"
                )
    elif normalized != "faiss":
        logger.warning("未知的 RAG_VECTOR_STORE=%s，已回退到 FAISS", kind)

    store = FaissVectorStore(cache_dir=cache_dir)
    logger.info("使用 FAISS 向量库：%s", store.describe())
    return store
