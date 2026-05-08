"""
Калибровка порога hybrid_score: перцентили max(hybrid_score) среди топ-3 кандидатов.

Запуск:
  python -m tests.calibrate_hybrid_threshold
  python -m tests.calibrate_hybrid_threshold --benchmark tests/benchmarks/benchmark_v4.json --limit 200

Результат по умолчанию: reports/v4/hybrid_score_percentiles.json
При смене корпуса или RRF-k пересчитайте и обновите RAG_MIN_HYBRID_SCORE / OutputPipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipelines.query import QueryPipeline
from tests.retrieval_eval import load_tests


def _percentile_linear(values: List[float], p: float) -> Optional[float]:
    """p в диапазоне 0..100; линейная интерполяция между соседними элементами."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    lo = max(0, min(lo, len(s) - 1))
    hi = max(0, min(hi, len(s) - 1))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid score percentiles for min_hybrid_score calibration")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=_ROOT / "tests" / "benchmarks" / "benchmark_v4.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "reports" / "v4" / "hybrid_score_percentiles.json",
    )
    args = parser.parse_args()

    tests = load_tests(args.benchmark, limit=args.limit)
    pipeline = QueryPipeline(base_dir=_ROOT)

    max_scores: List[float] = []
    for test in tests:
        pr = pipeline.run(test.query)
        if not pr.top_chunks:
            continue
        top3 = pr.top_chunks[:3]
        max_scores.append(max(float(sc.hybrid_score) for sc in top3))

    n = len(tests)
    m = len(max_scores)
    report = {
        "queries": n,
        "samples_with_chunks": m,
        "min": min(max_scores) if max_scores else None,
        "max": max(max_scores) if max_scores else None,
        "mean": round(statistics.mean(max_scores), 15) if max_scores else None,
        "median": round(statistics.median(max_scores), 15) if max_scores else None,
        "p5": (round(p5, 15) if (p5 := _percentile_linear(max_scores, 5)) is not None else None),
        "p10": (round(p10, 15) if (p10 := _percentile_linear(max_scores, 10)) is not None else None),
        "p25": (round(p25, 15) if (p25 := _percentile_linear(max_scores, 25)) is not None else None),
        "p75": (round(p75, 15) if (p75 := _percentile_linear(max_scores, 75)) is not None else None),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
