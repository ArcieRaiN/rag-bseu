from __future__ import annotations

"""Metadata/search_context ablations on the v4 vector store.

These ablations reuse the same FAISS index and change in-memory chunk metadata
before BM25/metadata fusion. They are intended to quantify retrieval-stage
contributions without paying the cost of rebuilding embeddings for every variant.
"""

import argparse
import json
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.query import QueryPipeline
from src.retrieval.hybrid_search import HybridSearcher
from tests.retrieval_eval import TestCase, evaluate_single, load_tests


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "tests" / "results" / "v4"


def reset_hybrid(pipeline: QueryPipeline) -> None:
    pipeline._hybrid = HybridSearcher(  # noqa: SLF001 - experiment harness
        semantic_searcher=pipeline._semantic,  # noqa: SLF001
        config=pipeline._retrieval_config,  # noqa: SLF001
    )


def run(name: str, mutate: Callable[[QueryPipeline], None], tests: List[TestCase], vector_store: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    pipeline = QueryPipeline(base_dir=BASE_DIR, vector_store_dir=vector_store)
    mutate(pipeline)
    reset_hybrid(pipeline)

    results = []
    for idx, test in enumerate(tests, 1):
        results.append(evaluate_single(pipeline, test))
        if idx % 100 == 0:
            print(f"[{name}] {idx}/{len(tests)}")

    total = len(results)
    elapsed_values = [r.elapsed_s for r in results]
    return {
        "name": name,
        "queries": total,
        "hit_at_1": sum(int(r.hit_at_1) for r in results) / max(total, 1),
        "hit_at_3": sum(int(r.hit_at_3) for r in results) / max(total, 1),
        "hit_at_5": sum(int(r.hit_at_5) for r in results) / max(total, 1),
        "mrr": sum(r.reciprocal_rank for r in results) / max(total, 1),
        "avg_time_s": statistics.mean(elapsed_values) if elapsed_values else None,
        "wall_time_s": time.perf_counter() - started,
    }


def keep_only_geo(pipeline: QueryPipeline) -> None:
    for chunk in pipeline._semantic.get_all_chunks():  # noqa: SLF001
        chunk.metrics = None
        chunk.years = []


def keep_years_metrics_units(pipeline: QueryPipeline) -> None:
    for chunk in pipeline._semantic.get_all_chunks():  # noqa: SLF001
        chunk.geo = None


def search_context_only(pipeline: QueryPipeline) -> None:
    for chunk in pipeline._semantic.get_all_chunks():  # noqa: SLF001
        chunk.geo = None
        chunk.metrics = None
        chunk.years = []
        chunk.units = None


def final_full(pipeline: QueryPipeline) -> None:
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v4 ablation experiments")
    parser.add_argument("--benchmark", default=str(BASE_DIR / "tests" / "benchmarks" / "benchmark_v4.json"))
    parser.add_argument("--vector-store", default=str(BASE_DIR / "usage" / "vector_store"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tests = load_tests(Path(args.benchmark), limit=args.limit)
    vector_store = Path(args.vector_store)
    variants = [
        ("regex_geo_only", keep_only_geo),
        ("years_metrics_units_only", keep_years_metrics_units),
        ("search_context_only", search_context_only),
        ("final_best_full", final_full),
    ]

    reports = [run(name, mutate, tests, vector_store) for name, mutate in variants]
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": args.benchmark,
        "vector_store": str(vector_store),
        "note": "Ablations reuse the FAISS index in usage/vector_store; metadata/BM25 chunk fields are modified in memory.",
        "variants": reports,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"v4_ablation_experiments_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (BASE_DIR / "reports" / "v4" / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
