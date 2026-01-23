"""
CLI для нового пайплайна запросов (PIPELINE 2–4).

Требования:
- два режима через флаги запуска:
  * --predefined-queries  (запуск набора предопределённых запросов)
  * --user-queries        (интерактивный режим)
- НЕТ генерации ответа LLM — только Top‑3 чанка.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.main.models import ScoredChunk
from src.main.query_pipeline_v2 import QueryPipelineV2


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG-BSEU CLI v2 (hybrid retrieval + reranking).")
    p.add_argument(
        "--predefined-queries",
        action="store_true",
        help="Запустить набор предопределённых запросов.",
    )
    p.add_argument(
        "--user-queries",
        action="store_true",
        help="Интерактивный режим ввода запросов пользователя.",
    )
    return p


def _format_top_chunks(chunks: List[ScoredChunk]) -> str:
    """
    Форматирует вывод Top‑3 чанков для терминала.
    """
    if not chunks:
        return "Ничего не найдено."

    lines: List[str] = []
    for i, sc in enumerate(chunks, start=1):
        ch = sc.chunk
        header = f"{i}. [source={ch.source}, page={ch.page}, id={ch.id}]"
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

        lines.append(header)
        if meta_parts:
            lines.append("   " + "; ".join(meta_parts))
        # выводим только context, т.к. он более краткий и уже обогащён LLM
        lines.append(f"   context: {ch.context}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _build_argparser().parse_args()

    if not args.predefined_queries and not args.user_queries:
        print(
            "Ни один режим не включён. "
            "Используйте хотя бы один флаг: --predefined-queries или --user-queries."
        )
        return

    base_dir = Path(__file__).resolve().parents[1]
    pipeline = QueryPipelineV2(base_dir=base_dir)

    predefined_queries = [
        "Численность населения по областям Беларуси",
        "Производство молока",
        "Число учреждений здравоохранения",
        "Добыча нефти в Беларуси",
    ]

    # Режим предопределённых запросов
    if args.predefined_queries:
        for q in predefined_queries:
            print("=" * 80)
            print(f"[ОТВЕТ] на запрос: \"{q}\"")
            print("-" * 80)
            result = pipeline.run(q)
            print(_format_top_chunks(result.top_chunks))
            print()

    # Режим пользовательского ввода
    if args.user_queries:
        print("\n" + "=" * 80)
        print("Интерактивный режим. Введите запрос (Ctrl+C для выхода).")
        print("=" * 80 + "\n")

        try:
            while True:
                query = input("> ").strip()
                if not query:
                    continue
                result = pipeline.run(query)
                print()
                print(f"[ОТВЕТ] на запрос: \"{query}\"")
                print("-" * 80)
                print(_format_top_chunks(result.top_chunks))
                print()
        except KeyboardInterrupt:
            print("\nВыход из программы.")


if __name__ == "__main__":
    main()

