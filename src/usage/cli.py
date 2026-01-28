"""
CLI для нового пайплайна запросов (PIPELINE 2–4).

Требования:
- два режима через флаги запуска:
  * --predefined-queries  (запуск набора предопределённых запросов)
  * --user-queries       (интерактивный режим)
- НЕТ генерации ответа LLM — только Top-3 чанка.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.main.models import ScoredChunk
from src.main.query_pipeline_v2 import QueryPipelineV2


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RAG-BSEU CLI v2 (hybrid retrieval + reranking)."
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--predefined-queries",
        action="store_true",
        help="Запустить набор предопределённых запросов (по умолчанию).",
    )
    group.add_argument(
        "--user-queries",
        action="store_true",
        help="Интерактивный режим ввода запросов пользователя.",
    )

    return p


def _format_top_chunks(chunks: List[ScoredChunk]) -> str:
    if not chunks:
        return "Ничего не найдено."

    lines: List[str] = []
    for i, sc in enumerate(chunks, start=1):
        ch = sc.chunk
        lines.append(f"{i}. [source={ch.source}, page={ch.page}, id={ch.id}]")

        meta_parts = []
        if ch.geo:
            meta_parts.append(f"geo={ch.geo}")
        if ch.years:
            meta_parts.append(f"years={ch.years}")
        if ch.metrics:
            meta_parts.append(f"metrics={ch.metrics}")
        if ch.time_granularity:
            meta_parts.append(f"time={ch.time_granularity}")
        if ch.oked:
            meta_parts.append(f"oked={ch.oked}")

        if meta_parts:
            lines.append("   " + "; ".join(meta_parts))

        lines.append(f"   context: {ch.context}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = _build_argparser().parse_args()

    # по умолчанию — predefined
    run_predefined = not args.user_queries

    base_dir = Path(__file__).resolve().parents[1]
    pipeline = QueryPipelineV2(base_dir=base_dir)

    predefined_queries = [
        "Численность населения по областям Беларуси",
        "Производство молока",
        "Число учреждений здравоохранения",
        # "Добыча нефти в Беларуси",
    ]

    if run_predefined:
        for q in predefined_queries:
            print(f'Запрос: "{q}"')
            result = pipeline.run(q)
            print(_format_top_chunks(result.top_chunks))

    if args.user_queries:
        print("Интерактивный режим. Введите запрос (Ctrl+C для выхода).")

        try:
            while True:
                query = input("> ").strip()
                if not query:
                    continue

                result = pipeline.run(query)
                print(f'Запрос: "{query}"')
                print(_format_top_chunks(result.top_chunks))
        except KeyboardInterrupt:
            print("Выход из программы.")


if __name__ == "__main__":
    main()
