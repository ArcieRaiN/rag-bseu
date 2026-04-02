"""
Автоматическая оценка качества retrieval в RAG-системе.

Метрики: Hit@1, Hit@3, Hit@5, MRR, Average Query Time.
Тестовые вопросы: tests/test_data.json (v3, 182 вопроса, 11 категорий).
Результаты: tests/results/eval_YYYYMMDD_HHMMSS.json.

Usage:
    python -m tests.evaluator                      # полный прогон
    python -m tests.evaluator --quick              # первые 10 вопросов
    python -m tests.evaluator --category prices    # только категория prices
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.query_pipeline import QueryPipeline
from src.core.models import ScoredChunk

TEST_DATA_PATH = Path(__file__).resolve().parent / "test_data.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class TestCase:
    query: str
    expected_sources: List[str]
    category: str


@dataclass
class TestResult:
    query: str
    category: str
    expected_sources: List[str]
    retrieved_sources: List[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    elapsed_s: float


@dataclass
class EvalSummary:
    total: int = 0
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    mrr_sum: float = 0.0
    total_time: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def hit_rate_1(self) -> float:
        return self.hit_at_1 / max(self.total, 1)

    @property
    def hit_rate_3(self) -> float:
        return self.hit_at_3 / max(self.total, 1)

    @property
    def hit_rate_5(self) -> float:
        return self.hit_at_5 / max(self.total, 1)

    @property
    def mrr(self) -> float:
        return self.mrr_sum / max(self.total, 1)

    @property
    def avg_time(self) -> float:
        return self.total_time / max(self.total, 1)


def load_tests(path: Path, category: Optional[str] = None, limit: Optional[int] = None) -> List[TestCase]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = []
    for t in data["tests"]:
        if category and t["category"] != category:
            continue
        tests.append(TestCase(
            query=t["query"],
            expected_sources=t["expected_sources"],
            category=t["category"],
        ))

    if limit:
        tests = tests[:limit]
    return tests


def source_key(chunk: ScoredChunk) -> str:
    """Build source key matching test_data format: 'SourceName.pdf::pageN'"""
    return f"{chunk.chunk.source}::page{chunk.chunk.page}"


def evaluate_single(pipeline: QueryPipeline, test: TestCase) -> TestResult:
    t0 = time.perf_counter()
    result = pipeline.run(test.query)
    elapsed = time.perf_counter() - t0

    retrieved = [source_key(sc) for sc in result.top_chunks]
    expected_set = set(test.expected_sources)

    hit_1 = any(s in expected_set for s in retrieved[:1])
    hit_3 = any(s in expected_set for s in retrieved[:3])
    hit_5 = any(s in expected_set for s in retrieved[:5])

    rr = 0.0
    for rank, s in enumerate(retrieved, 1):
        if s in expected_set:
            rr = 1.0 / rank
            break

    return TestResult(
        query=test.query,
        category=test.category,
        expected_sources=test.expected_sources,
        retrieved_sources=retrieved,
        hit_at_1=hit_1,
        hit_at_3=hit_3,
        hit_at_5=hit_5,
        reciprocal_rank=rr,
        elapsed_s=elapsed,
    )


def run_evaluation(
    tests: List[TestCase],
    pipeline: QueryPipeline,
) -> EvalSummary:
    summary = EvalSummary()

    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Query: {test.query!r}")
        try:
            result = evaluate_single(pipeline, test)
            summary.results.append(result)
            summary.total += 1
            summary.hit_at_1 += int(result.hit_at_1)
            summary.hit_at_3 += int(result.hit_at_3)
            summary.hit_at_5 += int(result.hit_at_5)
            summary.mrr_sum += result.reciprocal_rank
            summary.total_time += result.elapsed_s

            status = "HIT" if result.hit_at_5 else "MISS"
            print(f"  {status} | RR={result.reciprocal_rank:.2f} | time={result.elapsed_s:.2f}s")
            print(f"  Expected: {test.expected_sources[:2]}")
            print(f"  Got:      {result.retrieved_sources[:3]}")
        except Exception as e:
            print(f"  ERROR: {e}")
            summary.failures.append(f"{test.query}: {e}")
            summary.total += 1

    return summary


def print_summary(summary: EvalSummary) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total queries:   {summary.total}")
    print(f"Errors:          {len(summary.failures)}")
    print(f"Hit@1:           {summary.hit_rate_1:.1%} ({summary.hit_at_1}/{summary.total})")
    print(f"Hit@3:           {summary.hit_rate_3:.1%} ({summary.hit_at_3}/{summary.total})")
    print(f"Hit@5:           {summary.hit_rate_5:.1%} ({summary.hit_at_5}/{summary.total})")
    print(f"MRR:             {summary.mrr:.3f}")
    print(f"Avg time:        {summary.avg_time:.2f}s")
    print(f"Total time:      {summary.total_time:.1f}s")

    categories: Dict[str, List[TestResult]] = {}
    for r in summary.results:
        categories.setdefault(r.category, []).append(r)

    if categories:
        print("\nPer-category Hit@5:")
        for cat, results in sorted(categories.items()):
            hits = sum(1 for r in results if r.hit_at_5)
            print(f"  {cat:20s}  {hits}/{len(results)}  ({hits/len(results):.0%})")

    if summary.failures:
        print("\nFailed queries:")
        for f in summary.failures:
            print(f"  - {f}")
    print("=" * 60)


def save_results(summary: EvalSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total": summary.total,
        "hit_at_1": summary.hit_rate_1,
        "hit_at_3": summary.hit_rate_3,
        "hit_at_5": summary.hit_rate_5,
        "mrr": summary.mrr,
        "avg_time_s": summary.avg_time,
        "results": [
            {
                "query": r.query,
                "category": r.category,
                "hit_at_5": r.hit_at_5,
                "reciprocal_rank": r.reciprocal_rank,
                "elapsed_s": r.elapsed_s,
                "expected": r.expected_sources,
                "retrieved": r.retrieved_sources,
            }
            for r in summary.results
        ],
        "failures": summary.failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluator")
    parser.add_argument("--quick", action="store_true", help="Run only first 10 tests")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--test-file", type=str, default=None, help="Path to test data JSON")
    args = parser.parse_args()

    test_path = Path(args.test_file) if args.test_file else TEST_DATA_PATH
    limit = 10 if args.quick else None

    tests = load_tests(test_path, category=args.category, limit=limit)
    print(f"Loaded {len(tests)} test cases from {test_path}")

    print("Initializing QueryPipeline...")
    pipeline = QueryPipeline(base_dir=BASE_DIR)

    summary = run_evaluation(tests, pipeline)
    print_summary(summary)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_results(summary, RESULTS_DIR / f"eval_{timestamp}.json")


if __name__ == "__main__":
    main()
