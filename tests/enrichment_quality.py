from __future__ import annotations

"""
Measure metadata coverage in a vector-store data.json.

Usage:
    python -m tests.enrichment_quality
    python -m tests.enrichment_quality --data usage/vector_store/data.json --only-documents
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "tests" / "results" / "v4"


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def load_rows(path: Path, only_documents: bool) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not only_documents:
        return rows
    documents = {p.name for p in (BASE_DIR / "usage" / "documents").glob("*.pdf")}
    return [row for row in rows if row.get("source") in documents]


def summarize(rows: List[Dict[str, Any]], *, label: str, data_path: Path) -> Dict[str, Any]:
    fields = ["section", "geo", "metrics", "units", "years", "search_context"]
    coverage = {}
    for field in fields:
        missing = sum(_is_missing(row.get(field)) for row in rows)
        coverage[field] = {
            "present": len(rows) - missing,
            "missing": missing,
            "present_pct": round((len(rows) - missing) / max(len(rows), 1) * 100, 2),
            "missing_pct": round(missing / max(len(rows), 1) * 100, 2),
        }

    search_lengths = [len(row.get("search_context") or "") for row in rows]

    return {
        "label": label,
        "data_path": str(data_path),
        "chunks": len(rows),
        "sources": len({row.get("source") for row in rows}),
        "coverage": coverage,
        "avg_search_context_length": round(statistics.mean(search_lengths), 2) if search_lengths else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure enrichment metadata coverage")
    parser.add_argument("--data", default=str(BASE_DIR / "usage" / "vector_store" / "data.json"))
    parser.add_argument("--label", default="current")
    parser.add_argument("--only-documents", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    rows = load_rows(data_path, only_documents=args.only_documents)
    report = summarize(rows, label=args.label, data_path=data_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else RESULTS_DIR / f"enrichment_quality_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
