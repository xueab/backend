"""
RAG 离线评估脚本（P6）。

用法：
    py eval_rag.py
    py eval_rag.py --no-rerank --no-rewrite       # 关闭 rerank / 改写做消融
    py eval_rag.py --no-bm25                       # 仅 dense
    py eval_rag.py --dataset eval/qa_dataset.json --top-k 5
    py eval_rag.py --output eval/results.json      # 保存逐条结果便于复盘

指标：
    hit@k：top-k 命中（命中标注 expected 中任意一个 source 即算 hit）
    MRR ：Mean Reciprocal Rank（命中位置的倒数均值；未命中按 0 计）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 让脚本可以直接运行：把 ai-service 目录加入 sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import Settings, _build_pipeline  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    query: str
    expected: list[str]
    topic: str = ""
    note: str = ""


@dataclass
class EvalResult:
    query: str
    expected: list[str]
    hits: list[dict]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    rank: Optional[int]
    elapsed_ms: float


def _load_dataset(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items") if isinstance(raw, dict) else raw
    if not items:
        raise RuntimeError(f"评估集为空：{path}")
    cases: list[EvalCase] = []
    for it in items:
        cases.append(
            EvalCase(
                query=it["query"],
                expected=[e.lower() for e in it.get("expected", [])],
                topic=it.get("topic", ""),
                note=it.get("note", ""),
            )
        )
    return cases


def _eval_case(pipeline, case: EvalCase, top_k: int) -> EvalResult:
    started = time.perf_counter()
    hits = pipeline.search(case.query, top_k=top_k, min_score=0.0, enable_query_rewrite=None)
    elapsed = (time.perf_counter() - started) * 1000.0

    hit_sources = [(h.source or "").lower() for h in hits]
    expected_set = set(case.expected)

    rank: Optional[int] = None
    for i, src in enumerate(hit_sources, start=1):
        if src in expected_set:
            rank = i
            break

    return EvalResult(
        query=case.query,
        expected=case.expected,
        hits=[
            {
                "rank": i + 1,
                "source": h.source,
                "title": h.title,
                "score": round(float(h.score), 4),
            }
            for i, h in enumerate(hits)
        ],
        hit_at_1=rank is not None and rank <= 1,
        hit_at_3=rank is not None and rank <= 3,
        hit_at_5=rank is not None and rank <= 5,
        rank=rank,
        elapsed_ms=round(elapsed, 1),
    )


def _summarize(results: list[EvalResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"count": 0}
    hit1 = sum(1 for r in results if r.hit_at_1) / n
    hit3 = sum(1 for r in results if r.hit_at_3) / n
    hit5 = sum(1 for r in results if r.hit_at_5) / n
    mrr = sum((1.0 / r.rank) if r.rank else 0.0 for r in results) / n
    avg_ms = sum(r.elapsed_ms for r in results) / n
    return {
        "count": n,
        "hit@1": round(hit1, 4),
        "hit@3": round(hit3, 4),
        "hit@5": round(hit5, 4),
        "mrr": round(mrr, 4),
        "avg_ms": round(avg_ms, 1),
    }


def _override_settings(args) -> Settings:
    # Pydantic-Settings 会从 .env 读取，再用代码覆盖部分开关
    s = Settings()
    if args.no_bm25:
        s.rag_enable_bm25 = False
    if args.no_rerank:
        s.rag_enable_rerank = False
    if args.no_rewrite:
        s.rag_enable_query_rewrite = False
    if args.embedder:
        s.rag_embedder = args.embedder
    if args.reranker:
        s.rag_reranker = args.reranker
    return s


def _print_table(results: list[EvalResult]) -> None:
    print(f"{'rank':>5}  {'h@3':>4}  {'ms':>6}  query")
    print("-" * 80)
    for r in results:
        rank_str = str(r.rank) if r.rank else "-"
        mark = "✓" if r.hit_at_3 else "·"
        q = r.query if len(r.query) <= 56 else r.query[:55] + "…"
        print(f"{rank_str:>5}  {mark:>4}  {r.elapsed_ms:>6.1f}  {q}")


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="RAG 离线评估")
    parser.add_argument("--dataset", default="eval/qa_dataset.json", help="评估集路径")
    parser.add_argument("--top-k", type=int, default=5, help="检索 top-k")
    parser.add_argument("--no-bm25", action="store_true", help="关闭稀疏召回")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 rerank")
    parser.add_argument("--no-rewrite", action="store_true", help="关闭 query 改写")
    parser.add_argument("--embedder", default="", help="覆盖 RAG_EMBEDDER")
    parser.add_argument("--reranker", default="", help="覆盖 RAG_RERANKER")
    parser.add_argument("--output", default="", help="可选：把逐条结果写到 JSON 文件")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（调试用）")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    cases = _load_dataset(dataset_path)
    if args.limit > 0:
        cases = cases[: args.limit]

    settings = _override_settings(args)

    print("=" * 80)
    print("配置：")
    print(f"  embedder       = {settings.rag_embedder}")
    print(f"  vector_store   = {settings.rag_vector_store}")
    print(f"  reranker       = {settings.rag_reranker if settings.rag_enable_rerank else 'noop'}")
    print(f"  enable_bm25    = {settings.rag_enable_bm25}")
    print(f"  enable_rerank  = {settings.rag_enable_rerank}")
    print(f"  enable_rewrite = {settings.rag_enable_query_rewrite}")
    print(f"  top_k          = {args.top_k}")
    print(f"  dataset        = {dataset_path} ({len(cases)} 条)")
    print("=" * 80)

    pipeline = _build_pipeline(settings)
    if pipeline is None:
        print("ERROR：pipeline 初始化失败，请检查 .env 配置 / Qdrant 是否启动 / 模型是否能加载")
        return 2

    results: list[EvalResult] = []
    for case in cases:
        try:
            results.append(_eval_case(pipeline, case, top_k=args.top_k))
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # 单条失败不阻塞总评估
            logger.warning("评估失败 query=%r：%s", case.query, exc)
            results.append(
                EvalResult(
                    query=case.query, expected=case.expected, hits=[],
                    hit_at_1=False, hit_at_3=False, hit_at_5=False, rank=None, elapsed_ms=0.0,
                )
            )

    print()
    _print_table(results)

    summary = _summarize(results)
    print()
    print("=" * 80)
    print("总体指标：")
    for k, v in summary.items():
        print(f"  {k:<8} = {v}")
    print("=" * 80)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "config": {
                        "embedder": settings.rag_embedder,
                        "vector_store": settings.rag_vector_store,
                        "reranker": settings.rag_reranker,
                        "enable_bm25": settings.rag_enable_bm25,
                        "enable_rerank": settings.rag_enable_rerank,
                        "enable_rewrite": settings.rag_enable_query_rewrite,
                        "top_k": args.top_k,
                    },
                    "summary": summary,
                    "results": [
                        {
                            "query": r.query,
                            "expected": r.expected,
                            "rank": r.rank,
                            "hit@1": r.hit_at_1,
                            "hit@3": r.hit_at_3,
                            "hit@5": r.hit_at_5,
                            "elapsed_ms": r.elapsed_ms,
                            "hits": r.hits,
                        }
                        for r in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n结果已写入：{out_path}")

    # 按 hit@3 是否 >= 0.7 给出退出码（CI 友好）
    return 0 if summary.get("hit@3", 0) >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
