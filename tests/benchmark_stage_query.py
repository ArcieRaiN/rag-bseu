"""
Замер времени этапа обработки запросов.

Режимы:
  1) retrieval (по умолчанию) — только QueryPipeline.run (эмбеддинг запроса + гибридный поиск),
     полный прогон по tests/test_data.json (182 вопроса), без генерации таблицы LLM.
  2) e2e — QueryPipeline + OutputPipeline для каждого запроса (медленно; по умолчанию --e2e-limit 3).

Результаты: tests/results/benchmark_query_<timestamp>.json

Usage:
    python -m tests.benchmark_stage_query
    python -m tests.benchmark_stage_query --quick
    python -m tests.benchmark_stage_query --e2e --e2e-limit 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.output import OutputPipeline
from src.pipelines.query import QueryPipeline

from tests.evaluator import load_tests, TEST_DATA_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_retrieval_benchmark(test_path: Path, limit: int | None) -> Dict[str, Any]:
    tests = load_tests(test_path, category=None, limit=limit)
    pipeline = QueryPipeline(base_dir=BASE_DIR)

    times: List[float] = []
    enrich_times: List[float] = []
    search_times: List[float] = []

    t_wall0 = time.perf_counter()
    for i, t in enumerate(tests, 1):
        t0 = time.perf_counter()
        result = pipeline.run(t.query)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if result.timings:
            enrich_times.append(result.timings.get("enrich", 0.0))
            search_times.append(result.timings.get("search", 0.0))
        if i % 50 == 0:
            print(f"  ... {i}/{len(tests)}")
    wall = time.perf_counter() - t_wall0

    return {
        "mode": "retrieval_only",
        "queries": len(tests),
        "wall_time_total_s": round(wall, 2),
        "avg_query_s": round(statistics.mean(times), 4),
        "median_query_s": round(statistics.median(times), 4),
        "stdev_query_s": round(statistics.pstdev(times), 4) if len(times) > 1 else 0.0,
        "avg_enrich_s": round(statistics.mean(enrich_times), 4) if enrich_times else None,
        "avg_search_s": round(statistics.mean(search_times), 4) if search_times else None,
    }


def run_e2e_benchmark(test_path: Path, limit: int) -> Dict[str, Any]:
    tests = load_tests(test_path, category=None, limit=limit)
    qpipe = QueryPipeline(base_dir=BASE_DIR)
    out = OutputPipeline(output_dir=BASE_DIR / "usage" / "outputs")

    times: List[float] = []
    for i, t in enumerate(tests, 1):
        print(f"[e2e {i}/{len(tests)}] {t.query[:60]}...")
        t0 = time.perf_counter()
        result = qpipe.run(t.query)
        if not result.top_chunks:
            times.append(time.perf_counter() - t0)
            continue
        out.run(result, user_query=t.query)
        times.append(time.perf_counter() - t0)

    return {
        "mode": "retrieval_plus_output_llm",
        "queries": len(tests),
        "avg_query_s": round(statistics.mean(times), 2),
        "median_query_s": round(statistics.median(times), 2),
        "min_s": round(min(times), 2),
        "max_s": round(max(times), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark query processing stage")
    parser.add_argument("--quick", action="store_true", help="First 10 queries only")
    parser.add_argument("--test-file", type=str, default=None)
    parser.add_argument("--e2e", action="store_true", help="Include OutputPipeline (LLM table)")
    parser.add_argument("--e2e-limit", type=int, default=3, help="Max queries in e2e mode")
    args = parser.parse_args()

    test_path = Path(args.test_file) if args.test_file else TEST_DATA_PATH
    limit = 10 if args.quick else None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    if args.e2e:
        report = run_e2e_benchmark(test_path, limit=args.e2e_limit)
    else:
        report = run_retrieval_benchmark(test_path, limit=limit)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    out_path = RESULTS_DIR / f"benchmark_query_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
