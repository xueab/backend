"""
BM25 稀疏检索（P3 引入）。

为什么需要：dense embedding 在中文短查询、专有名词（"PUA"、"MBTI"、"SSRI"）和
精确字面匹配上经常漂移。BM25 + 中文分词作为补充召回路，与 dense 路用 RRF 融合，
对召回率提升非常直观。

设计要点：
- 用 ``jieba`` 做分词（轻量、纯 Python，无额外模型下载）。
- 用 ``rank_bm25`` 的 ``BM25Okapi`` 做打分（经典实现，足够 < 10w 量级）。
- 索引随知识库一起持久化到 ``cache_dir/bm25_index.pkl``，按 fingerprint 复用。
- 与 ``vector_stores.ChunkRecord`` 同构，避免反向依赖 ``rag.py``。
"""

from __future__ import annotations

import logging
import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vector_stores import ChunkRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseHit:
    record: ChunkRecord
    score: float
    index: int


# 简单的中英文混合分词预处理：把英文 / 数字按空白和符号切，剩下交给 jieba。
_NON_WORD = re.compile(r"[\s,.;:!?，。；：！？、（）()【】\[\]\"'`~!@#$%^&*+=<>/\\|{}]+")


class _Tokenizer:
    """惰性持有 jieba 实例，并对常见心理健康词条做一次词典加固。"""

    _DOMAIN_WORDS = (
        "正念",
        "自我关怀",
        "认知行为疗法",
        "辩证行为疗法",
        "接纳承诺疗法",
        "睡眠卫生",
        "惊恐发作",
        "PUA",
        "gaslighting",
        "失眠",
        "焦虑",
        "抑郁",
        "孤独感",
        "强迫",
        "倦怠",
        "拖延",
        "完美主义",
        "高敏感",
        "自我价值",
        "情绪调节",
        "深呼吸",
        "心理咨询",
        "危机干预",
    )

    def __init__(self) -> None:
        self._jieba = None
        self._lock = threading.Lock()

    def _load(self):
        if self._jieba is not None:
            return self._jieba
        with self._lock:
            if self._jieba is not None:
                return self._jieba
            try:
                import jieba
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "jieba 未安装，无法构建 BM25 索引。请执行 `pip install jieba`。"
                ) from exc
            for w in self._DOMAIN_WORDS:
                jieba.add_word(w)
            self._jieba = jieba
            return jieba

    def tokenize(self, text: str) -> list[str]:
        """中文 BM25 友好分词：

        - 主分词走 ``jieba.cut_for_search``（能把长词切成多种粒度）。
        - 对 2 字以上中文词，额外把单字也加入 token 流，缓解 jieba 对 2 字词不再细分
          的问题（例如"睡眠"切出后查询"睡不着"也能命中"睡"）。
        - ASCII 段（英文 / 数字）保留原 token 并补一份小写形式。
        """
        if not text:
            return []
        jieba = self._load()
        normalized = _NON_WORD.sub(" ", text)
        tokens: list[str] = []
        for piece in normalized.split():
            if not piece:
                continue
            if piece.isascii():
                tokens.append(piece.lower())
                continue
            for t in jieba.cut_for_search(piece):
                t = t.strip()
                if not t:
                    continue
                tokens.append(t)
                if len(t) >= 2 and not t.isascii():
                    for ch in t:
                        if ch.strip():
                            tokens.append(ch)
        return tokens


_DEFAULT_TOKENIZER = _Tokenizer()


class BM25Retriever:
    """BM25 稀疏检索器，与 ``KnowledgeRetriever`` 同样按 fingerprint 复用缓存。"""

    _CACHE_FILE = "bm25_index.pkl"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._lock = threading.Lock()
        self._records: list[ChunkRecord] = []
        self._tokenized: list[list[str]] = []
        self._bm25 = None
        self._fingerprint: Optional[str] = None
        self._tokenizer = _DEFAULT_TOKENIZER
        self._try_load()

    # ------------------------------------------------------------------ 对外 API

    def chunk_count(self) -> int:
        return len(self._records)

    def needs_rebuild(self, fingerprint: str) -> bool:
        return self._fingerprint != fingerprint or self._bm25 is None or not self._records

    def rebuild(self, records: list[ChunkRecord], fingerprint: str) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "rank-bm25 未安装，无法构建 BM25 索引。请执行 `pip install rank-bm25`。"
            ) from exc

        with self._lock:
            self._records = list(records)
            self._tokenized = [
                self._tokenizer.tokenize(f"{r.title}\n{r.content}") for r in records
            ]
            self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
            self._fingerprint = fingerprint
            self._save()
        logger.info("BM25 索引已重建：共 %d 段", len(records))

    def search(
        self,
        query: str,
        top_k: int,
        category: Optional[str] = None,
    ) -> list[SparseHit]:
        if not query or not query.strip() or self._bm25 is None or not self._records:
            return []

        tokens = self._tokenizer.tokenize(query)
        if not tokens:
            return []

        with self._lock:
            scores = self._bm25.get_scores(tokens)
            records = self._records

        # 过滤 + 取 top_k：先 oversample 让分类过滤后仍能凑够 top_k
        oversample = top_k if category is None else max(top_k * 4, top_k + 10)
        # argsort 降序
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: min(oversample, len(scores))]

        hits: list[SparseHit] = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            record = records[idx]
            if category and record.category != category:
                continue
            hits.append(SparseHit(record=record, score=float(scores[idx]), index=idx))
            if len(hits) >= top_k:
                break
        return hits

    # ------------------------------------------------------------------ 持久化

    def _save(self) -> None:
        if self._bm25 is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / self._CACHE_FILE
        try:
            with open(path, "wb") as f:
                pickle.dump(
                    {
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
                        "tokenized": self._tokenized,
                    },
                    f,
                )
        except OSError as exc:
            logger.warning("BM25 索引持久化失败：%s", exc)

    def _try_load(self) -> None:
        path = self.cache_dir / self._CACHE_FILE
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            logger.warning("BM25 缓存读取失败，将忽略：%s", exc)
            return

        records = [
            ChunkRecord(
                source=item.get("source", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                category=item.get("category", "general"),
            )
            for item in payload.get("records", [])
        ]
        tokenized = payload.get("tokenized") or []
        if len(records) != len(tokenized):
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return  # 没装就保持空，后续 needs_rebuild 会触发重建

        self._records = records
        self._tokenized = tokenized
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._fingerprint = payload.get("fingerprint")
