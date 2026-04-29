from __future__ import annotations

"""Validate benchmark expected source keys against a vector-store data.json."""

import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark JSON")
    parser.add_argument("--benchmark", default=str(BASE_DIR / "tests" / "benchmarks" / "benchmark_v4.json"))
    parser.add_argument("--data", default=str(BASE_DIR / "usage" / "vector_store" / "data.json"))
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    valid_sources = {f"{row['source']}::page{row['page']}" for row in rows}

    errors = []
    seen_queries = set()
    for idx, item in enumerate(benchmark.get("tests", []), 1):
        query = item.get("query")
        expected = item.get("expected_sources") or []
        if not query:
            errors.append(f"{idx}: empty query")
        if query and query.lower() in seen_queries:
            errors.append(f"{idx}: duplicate query: {query}")
        seen_queries.add((query or "").lower())
        if not expected:
            errors.append(f"{idx}: empty expected_sources")
        missing = [source for source in expected if source not in valid_sources]
        if missing:
            errors.append(f"{idx}: expected source not in data.json: {missing[:3]}")

    report = {
        "benchmark": args.benchmark,
        "data": args.data,
        "tests": len(benchmark.get("tests", [])),
        "errors": errors,
        "is_valid": not errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
