"""
RAG 检索流水线（编排层）。

把分散的能力组合成一条线：

```
用户消息 → 危机词检测 ┐
                       └→ multi-query / HyDE 改写
                              │
                              ├─► Dense 召回（KnowledgeRetriever 拿走 query_vec → store.search）
                              └─► Sparse 召回（BM25Retriever）
                                       │
                                       ▼
                              RRF 融合 → 候选 N
                                       │
                                       ▼
                                  Reranker → top_k
```

实现原则：
- 所有子模块都可独立失败 / 跳过（reranker 是 Noop、bm25 不可用、改写超时 …），主路径仍能返回结果。
- 与 ``KnowledgeRetriever`` / ``main.py`` 解耦：依赖注入，不再做单例魔法。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from bm25_index import BM25Retriever
from query_rewriter import EnrichedQuery, LLMQueryRewriter, detect_crisis
from rag import KnowledgeRetriever, RetrievedChunk
from rerankers import Reranker
from vector_stores import ChunkRecord, StoredHit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- RRF 融合


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """对多路排序列表做 RRF 融合。

    每路输入是 (key, score) 列表（score 仅用于稳定排序，不参与融合权重）；
    返回按融合分数降序的 (key, fused_score) 列表。

    Args:
        rankings: 多路检索结果。每条排序里 key 必须是同一空间下的可比标识（推荐用 record uid）。
        k: RRF 平滑常量，经验取 60。

    Returns:
        融合后的 (key, fused_score) 列表，按分数降序。
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (key, _score) in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


# ---------------------------------------------------------------------------- 危机兜底


_CRISIS_SOURCES = ("crisis-resources.md",)


# ---------------------------------------------------------------------------- Pipeline


@dataclass
class PipelineConfig:
    top_k: int = 3
    min_score: float = 0.0  # 仅在 /rag/search 等需要硬过滤的场景生效；详见 search() 的注释
    rerank_min_score: float = 0.0  # rerank 后的最低 yes-prob；默认 0.0 = 信任 reranker 的 top_n
    candidate_pool: int = 20
    rrf_k: int = 60
    enable_bm25: bool = True
    enable_rerank: bool = True
    rerank_input_size: int = 20
    enable_query_rewrite: bool = True
    crisis_top_k: int = 2
    crisis_force_inject: bool = True
    bm25_dense_weight_split: tuple[int, int] = (1, 1)  # 仅信息性，未来可扩展加权


class RagPipeline:
    """对外的检索入口。"""

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        bm25: Optional[BM25Retriever],
        reranker: Optional[Reranker],
        rewriter: Optional[LLMQueryRewriter],
        config: PipelineConfig,
    ) -> None:
        self.retriever = retriever
        self.bm25 = bm25
        self.reranker = reranker
        self.rewriter = rewriter
        self.config = config
        self._lock = threading.Lock()
        self._initialized = False

    # ----- 初始化 / 预热

    def ensure_ready(self, force_rebuild: bool = False) -> int:
        """确保 dense + sparse 索引都已就绪；返回 dense 索引片段数。"""
        with self._lock:
            count = self.retriever.ensure_index(force_rebuild=force_rebuild)
            if self.bm25 is not None:
                fp = self.retriever._fingerprint()  # noqa: SLF001 - pipeline 与 retriever 强耦合，复用同一指纹
                if force_rebuild or self.bm25.needs_rebuild(fp):
                    logger.info("BM25 索引需要重建，开始构建（jieba 分词 + BM25Okapi）...")
                    records = self._dense_records()
                    self.bm25.rebuild(records, fp)
                else:
                    logger.info("BM25 索引复用缓存：%d 段", self.bm25.chunk_count())
            self._initialized = True
            return count

    def rebuild(self) -> int:
        """强制全量重建（供 /rag/reindex 调用）。"""
        return self.ensure_ready(force_rebuild=True)

    # ----- 主入口

    def search(
        self,
        query: str,
        *,
        history: Optional[list] = None,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        enable_query_rewrite: Optional[bool] = None,
    ) -> list[RetrievedChunk]:
        """完整流水线检索。允许调用方覆盖部分配置，便于 /rag/search 等场景。"""
        if not query or not query.strip():
            return []

        self.ensure_ready()
        cfg = self.config
        effective_top_k = int(top_k) if top_k else cfg.top_k
        effective_min_score = float(min_score) if min_score is not None else cfg.min_score
        do_rewrite = (
            cfg.enable_query_rewrite
            if enable_query_rewrite is None
            else bool(enable_query_rewrite)
        )
        text = query.strip()

        # 1. 危机词兜底：直接回退到 crisis-resources（如有）
        is_crisis = detect_crisis(text)
        if is_crisis and cfg.crisis_force_inject:
            crisis_hits = self._crisis_hits(top_k=cfg.crisis_top_k)
            if crisis_hits:
                logger.info("命中危机词，强制注入 crisis-resources：%d 段", len(crisis_hits))
                return crisis_hits
            # 没有 crisis-resources 时退回正常流程，但让 LLM 自己谨慎处理

        # 2. Query 改写（multi-query + HyDE）
        enriched: EnrichedQuery
        if self.rewriter and do_rewrite:
            try:
                enriched = self.rewriter.enrich(text, history=history)
            except Exception as exc:
                logger.warning("查询改写失败，使用原始查询：%s", exc)
                enriched = EnrichedQuery(raw=text, queries=[text])
        else:
            enriched = EnrichedQuery(raw=text, queries=[text])

        if not enriched.queries:
            enriched.queries = [text]

        if enriched.hyde:
            enriched.queries.append(enriched.hyde)

        # 3. 多 query × 多路召回 → RRF 融合
        candidates = self._multi_route_recall(enriched.queries, category=category)
        if not candidates:
            return []

        # 4. Rerank（候选只取前 rerank_input_size）
        # ⚠ min_score 的尺度是为原始余弦相似度（dense 召回）设计的（0~1，典型 0.3~0.7）。
        # 而 Qwen reranker 输出是 yes-token 的 softmax 概率（典型 0.05~0.99，中等相关 0.1~0.3），
        # RRF 融合分数是 Σ 1/(k+rank)（k=60 时典型 0.005~0.05）。
        # 三个尺度完全不同，混用同一阈值会导致融合 / 重排路径把"中等相关"全部砍光。
        # 这里采用各自独立的阈值：rerank_min_score 默认极低，本质上信任 reranker 的 top_n 选择；
        # 融合 fallback 路径则不再用 min_score 过滤，纯靠 top_k 截断。
        # 用户传入的 min_score 仅作用于"无 rerank 且想要硬阈值"的边缘场景，故仍保留覆盖能力。
        candidates = candidates[: cfg.rerank_input_size]
        if self.reranker and cfg.enable_rerank and len(candidates) > 1:
            try:
                docs = [self._format_for_rerank(c.record) for c in candidates]
                ranked = self.reranker.rerank(text, docs, top_n=effective_top_k)
                effective_rerank_floor = (
                    float(min_score) if min_score is not None else cfg.rerank_min_score
                )
                results: list[RetrievedChunk] = []
                for r in ranked:
                    if r.score < effective_rerank_floor:
                        continue
                    cand = candidates[r.index]
                    results.append(self._to_chunk(cand.record, r.score))
                if results:
                    logger.info(
                        "RAG 命中 %d 段（rerank=%s, queries=%d）：%s",
                        len(results),
                        self.reranker.backend_name,
                        len(enriched.queries),
                        [r.source for r in results],
                    )
                return results
            except Exception as exc:
                logger.warning("Rerank 失败，回退到融合分数排序：%s", exc)

        # 5. 没有 rerank（或 rerank 失败）→ 直接用融合分数；不再按 min_score 过滤（RRF 尺度不匹配）
        results: list[RetrievedChunk] = []
        for cand in candidates[:effective_top_k]:
            results.append(self._to_chunk(cand.record, cand.score))
        if results:
            logger.info(
                "RAG 命中 %d 段（无 rerank，queries=%d）：%s",
                len(results),
                len(enriched.queries),
                [r.source for r in results],
            )
        return results

    # ----- 内部

    def _multi_route_recall(
        self,
        queries: list[str],
        category: Optional[str],
    ) -> list["_Candidate"]:
        cfg = self.config
        # 按 record_key 去重，保留首次出现的 record（payload 一致即可）
        record_index: dict[str, _Candidate] = {}
        # 收集每个 query 的 dense / sparse 排序
        rankings: list[list[tuple[str, float]]] = []

        for q in queries:
            if not q:
                continue
            try:
                dense_hits = self._dense_recall(q, category=category, top_k=cfg.candidate_pool)
            except Exception as exc:
                logger.warning("Dense 召回失败 query=%r：%s", q, exc)
                dense_hits = []
            dense_ranking: list[tuple[str, float]] = []
            for hit in dense_hits:
                key = self._record_key(hit.record)
                dense_ranking.append((key, hit.score))
                record_index.setdefault(key, _Candidate(record=hit.record, score=hit.score))
            rankings.append(dense_ranking)

            if cfg.enable_bm25 and self.bm25 is not None:
                try:
                    sparse_hits = self.bm25.search(q, top_k=cfg.candidate_pool, category=category)
                except Exception as exc:
                    logger.warning("BM25 召回失败 query=%r：%s", q, exc)
                    sparse_hits = []
                sparse_ranking: list[tuple[str, float]] = []
                for hit in sparse_hits:
                    key = self._record_key(hit.record)
                    sparse_ranking.append((key, hit.score))
                    record_index.setdefault(key, _Candidate(record=hit.record, score=hit.score))
                rankings.append(sparse_ranking)

        if not rankings or not record_index:
            return []

        fused = reciprocal_rank_fusion(rankings, k=cfg.rrf_k)
        candidates: list[_Candidate] = []
        for key, fused_score in fused:
            cand = record_index.get(key)
            if cand is None:
                continue
            candidates.append(_Candidate(record=cand.record, score=fused_score))
        return candidates

    def _dense_recall(
        self,
        query: str,
        category: Optional[str],
        top_k: int,
    ) -> list[StoredHit]:
        # 复用 KnowledgeRetriever 的查询向量缓存与 store 检索
        query_vec = self.retriever._embed_query(query)  # noqa: SLF001
        return self.retriever.store.search(query_vec, top_k=top_k, category=category)

    def _crisis_hits(self, top_k: int) -> list[RetrievedChunk]:
        """从已加载的 records 中筛出 crisis-resources，避免再做一次向量检索。"""
        records = self._dense_records()
        crisis = [r for r in records if r.source.lower() in {s.lower() for s in _CRISIS_SOURCES}]
        if not crisis:
            return []
        # 取前 top_k 段（同一文档可能切了多段）
        return [
            self._to_chunk(rec, score=1.0)
            for rec in crisis[:top_k]
        ]

    def _dense_records(self) -> list[ChunkRecord]:
        """从 store 复原 records。FAISS store 是直接持有；Qdrant store 没暴露，则重新从 docs 加载。"""
        store_records = getattr(self.retriever.store, "_records", None)  # noqa: SLF001
        if store_records:
            return list(store_records)
        # Qdrant 等不持有 records 的 store，回退到从 docs 重新加载
        return self.retriever._load_records()  # noqa: SLF001

    @staticmethod
    def _record_key(record: ChunkRecord) -> str:
        # 用 source + 内容前 32 字符 hash 作为去重键
        head = (record.content or "").strip()[:32]
        return f"{record.source}::{record.title}::{head}"

    @staticmethod
    def _format_for_rerank(record: ChunkRecord) -> str:
        return f"《{record.title}》\n{record.content}"

    @staticmethod
    def _to_chunk(record: ChunkRecord, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            source=record.source,
            title=record.title,
            content=record.content,
            score=float(score),
            category=record.category,
        )


@dataclass
class _Candidate:
    record: ChunkRecord
    score: float
