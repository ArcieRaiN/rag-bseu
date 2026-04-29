from __future__ import annotations

"""Run v4 retrieval experiments against usage/vector_store (override via --variant)."""

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.query import QueryPipeline
from tests.retrieval_eval import TestCase, evaluate_single, load_tests
from tests.enrichment_quality import load_rows, summarize


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "tests" / "results" / "v4"


def run_variant(name: str, vector_store_dir: Path, tests: List[TestCase]) -> Dict[str, Any]:
    started = time.perf_counter()
    pipeline = QueryPipeline(base_dir=BASE_DIR, vector_store_dir=vector_store_dir)
    init_elapsed = time.perf_counter() - started

    results = []
    failures = []
    for idx, test in enumerate(tests, 1):
        try:
            result = evaluate_single(pipeline, test)
            results.append(result)
        except Exception as exc:  # keep long experiment running
            failures.append({"query": test.query, "error": str(exc)})
        if idx % 100 == 0:
            print(f"[{name}] {idx}/{len(tests)}")

    total = len(results) + len(failures)
    hit1 = sum(int(r.hit_at_1) for r in results)
    hit3 = sum(int(r.hit_at_3) for r in results)
    hit5 = sum(int(r.hit_at_5) for r in results)
    mrr = sum(r.reciprocal_rank for r in results) / max(total, 1)
    elapsed_values = [r.elapsed_s for r in results]
    wall = time.perf_counter() - started

    rows = load_rows(vector_store_dir / "data.json", only_documents=False)
    quality = summarize(rows, label=name, data_path=vector_store_dir / "data.json")

    return {
        "name": name,
        "vector_store_dir": str(vector_store_dir),
        "queries": total,
        "failures": failures,
        "hit_at_1": hit1 / max(total, 1),
        "hit_at_3": hit3 / max(total, 1),
        "hit_at_5": hit5 / max(total, 1),
        "mrr": mrr,
        "avg_time_s": statistics.mean(elapsed_values) if elapsed_values else None,
        "median_time_s": statistics.median(elapsed_values) if elapsed_values else None,
        "init_time_s": init_elapsed,
        "wall_time_s": wall,
        "enrichment_quality": quality,
        "results": [
            {
                "query": r.query,
                "category": r.category,
                "expected": r.expected_sources,
                "retrieved": r.retrieved_sources,
                "hit_at_1": r.hit_at_1,
                "hit_at_3": r.hit_at_3,
                "hit_at_5": r.hit_at_5,
                "reciprocal_rank": r.reciprocal_rank,
                "elapsed_s": r.elapsed_s,
            }
            for r in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v4 retrieval experiments")
    parser.add_argument("--benchmark", default=str(BASE_DIR / "tests" / "benchmarks" / "benchmark_v4.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variant",
        action="append",
        nargs=2,
        metavar=("NAME", "VECTOR_STORE_DIR"),
        help="Variant name and vector store directory. Can be repeated.",
    )
    args = parser.parse_args()

    tests = load_tests(Path(args.benchmark), limit=args.limit)
    variants = args.variant or [
        ("current", str(BASE_DIR / "usage" / "vector_store")),
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, store in variants:
        store_path = Path(store)
        if not (store_path / "data.json").exists() or not (store_path / "index.faiss").exists():
            reports.append({"name": name, "vector_store_dir": str(store_path), "status": "missing"})
            continue
        reports.append(run_variant(name, store_path, tests))

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": args.benchmark,
        "limit": args.limit,
        "variants": reports,
    }
    out = RESULTS_DIR / f"v4_experiments_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {
        "benchmark": args.benchmark,
        "limit": args.limit,
        "variants": [
            {
                "name": item.get("name"),
                "status": item.get("status", "ok"),
                "hit_at_1": item.get("hit_at_1"),
                "hit_at_3": item.get("hit_at_3"),
                "hit_at_5": item.get("hit_at_5"),
                "mrr": item.get("mrr"),
                "avg_time_s": item.get("avg_time_s"),
                "queries": item.get("queries"),
            }
            for item in reports
        ],
    }
    (BASE_DIR / "reports" / "v4" / "experiment_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
