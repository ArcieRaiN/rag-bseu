"""
USAGE: интерактивный запуск query_pipeline.

Запускает пайплайн запросов и выводит топ-3 чанка.
"""

from pathlib import Path
from src.pipelines.query_pipeline import QueryPipeline

def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent  # rag-bseu
    pipeline = QueryPipeline(base_dir=base_dir)

    print("Интерактивный режим. Введите запрос (Ctrl+C для выхода).")
    try:
        while True:
            query = input("> ").strip()
            if not query:
                continue

            result = pipeline.run(query)

            if not result.top_chunks:
                print("❌ Ничего не найдено.")
                continue

            print(f"Топ-{min(3, len(result.top_chunks))} чанков:")
            for i, sc in enumerate(result.top_chunks[:3], 1):
                ch = sc.chunk
                meta = []
                if ch.geo: meta.append(f"geo={ch.geo}")
                if ch.years: meta.append(f"years={ch.years}")
                if ch.metrics: meta.append(f"metrics={ch.metrics}")
                if ch.time_granularity: meta.append(f"time={ch.time_granularity}")
                if ch.oked: meta.append(f"oked={ch.oked}")

                print(f"{i}. [source={ch.source}, page={ch.page}, id={ch.id}]")
                if meta:
                    print("   " + "; ".join(meta))
                print(f"   context: {ch.context}")
                print()
    except KeyboardInterrupt:
        print("\nВыход из программы.")
