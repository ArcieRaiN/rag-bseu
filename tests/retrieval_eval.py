"""Shared retrieval evaluation helpers for benchmark JSON (v4+)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from dataclasses import dataclass
from typing import List, Optional

from src.core.models import ScoredChunk
from src.pipelines.query import QueryPipeline


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


def load_tests(path: Path, category: Optional[str] = None, limit: Optional[int] = None) -> List[TestCase]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = []
    for t in data["tests"]:
        if category and t.get("category") != category:
            continue
        tests.append(
            TestCase(
                query=t["query"],
                expected_sources=t["expected_sources"],
                category=str(t.get("category", "")),
            )
        )

    if limit:
        tests = tests[:limit]
    return tests


def source_key(chunk: ScoredChunk) -> str:
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
