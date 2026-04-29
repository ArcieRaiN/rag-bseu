from __future__ import annotations

"""
Generate benchmark v4 from enriched vector-store chunks.

The output keeps retrieval_eval-compatible fields (`query`, `expected_sources`,
`category`) and adds provenance/type/difficulty for auditability.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = BASE_DIR / "usage" / "vector_store" / "data.json"
DEFAULT_OUTPUT = BASE_DIR / "tests" / "benchmarks" / "benchmark_v4.json"


CATEGORY_RULES = [
    ("demographics", ["населен", "демограф", "рождаем", "смерт", "миграц", "браки", "развод"]),
    ("economy", ["ввп", "валовой", "национальн", "счет", "инвестиц", "предприним"]),
    ("prices", ["цен", "тариф", "инфляц", "ипц"]),
    ("trade", ["торгов", "экспорт", "импорт", "услуг", "товарооборот"]),
    ("social", ["доход", "заработ", "образован", "здравоохран", "социаль", "жилищ"]),
    ("regions", ["област", "регион", "минск", "район"]),
    ("science_it", ["наук", "инновац", "информацион", "интернет", "икт"]),
    ("agriculture", ["сельск", "растение", "животновод", "урожай", "скот"]),
    ("transport", ["транспорт", "перевоз", "дорог"]),
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" .;:")


def category_for(row: Dict[str, Any]) -> str:
    haystack = clean(" ".join(str(row.get(k) or "") for k in ("section", "source", "search_context"))).lower()
    for category, needles in CATEGORY_RULES:
        if any(needle in haystack for needle in needles):
            return category
    return "general"


def year_repr(years: Iterable[int]) -> str:
    values = sorted({int(y) for y in years if y})
    if not values:
        return ""
    if len(values) >= 3:
        return f"{values[0]}-{values[-1]}"
    return ", ".join(str(y) for y in values)


def source_key(row: Dict[str, Any]) -> str:
    return f"{row['source']}::page{row['page']}"


def metric_values(row: Dict[str, Any]) -> List[str]:
    values = row.get("metrics") or []
    if isinstance(values, str):
        values = [values]
    extra = row.get("extra") or {}
    if extra.get("table_title"):
        values = [extra["table_title"], *values]
    result = []
    for value in values:
        value = clean(value)
        if 4 <= len(value) <= 120 and not re.fullmatch(r"[\d\s.,:;%–—-]+", value):
            result.append(value)
    return list(dict.fromkeys(result))[:6]


def row_questions(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    metrics = metric_values(row)
    if not metrics:
        return []
    geo_values = row.get("geo") or []
    if isinstance(geo_values, str):
        geo_values = [geo_values]
    units = row.get("units") or []
    if isinstance(units, str):
        units = [units]
    years = row.get("years") or []
    yr = year_repr(years)
    geo = clean(geo_values[0]) if geo_values else ""
    unit = clean(units[0]) if units else ""
    section = clean(row.get("section") or "")
    expected = [source_key(row)]
    category = category_for(row)

    questions: List[Dict[str, Any]] = []
    for metric in metrics[:3]:
        templates = [
            ("simple_metric", "Найди данные по показателю {metric}", "easy"),
            ("source_wording", "Статистика: {metric}", "easy"),
        ]
        if geo:
            templates.append(("geo_metric", "{metric} по территории {geo}", "medium"))
        if yr:
            templates.append(("metric_years", "{metric} за {years}", "medium"))
        if geo and yr:
            templates.append(("geo_years", "{metric} {geo} {years}", "hard"))
        if unit:
            templates.append(("unit_metric", "{metric}, единица измерения {unit}", "hard"))
        if section:
            templates.append(("section_metric", "{metric} в разделе {section}", "medium"))

        for qtype, template, difficulty in templates:
            query = template.format(metric=metric, geo=geo, years=yr, unit=unit, section=section)
            questions.append(
                {
                    "query": clean(query),
                    "expected_sources": expected,
                    "category": category,
                    "type": qtype,
                    "difficulty": difficulty,
                    "provenance": {
                        "chunk_id": row.get("id"),
                        "source": row.get("source"),
                        "page": row.get("page"),
                        "metric": metric,
                    },
                }
            )
    return questions


def generate(rows: List[Dict[str, Any]], *, target_size: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    candidates = [
        row
        for row in rows
        if row.get("source")
        and row.get("page")
        and len(row.get("text") or "") > 120
        and metric_values(row)
    ]
    rng.shuffle(candidates)

    tests: List[Dict[str, Any]] = []
    seen = set()
    for row in candidates:
        variants = row_questions(row)
        rng.shuffle(variants)
        for item in variants:
            key = item["query"].lower()
            if key in seen:
                continue
            seen.add(key)
            tests.append(item)
            if len(tests) >= target_size:
                break
        if len(tests) >= target_size:
            break

    return {
        "version": "4.0",
        "description": "Benchmark v4 generated from 12 Belstat PDF publications with chunk-level expected sources.",
        "seed": seed,
        "target_size": target_size,
        "actual_size": len(tests),
        "tests": tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark v4")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    benchmark = generate(rows, target_size=args.target_size, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: benchmark[k] for k in ("version", "target_size", "actual_size", "seed")}, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
