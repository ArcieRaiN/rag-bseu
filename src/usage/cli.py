from pathlib import Path
from src.main.retriever import SemanticRetriever
from src.main.vectorizer import HashVectorizer

# Флаги запуска
predefined_queries_flag = True   # запустить в начале предопределённые запросы
user_queries_flag = False        # дать пользователю возможность вводить запросы

def main() -> None:
    # Базовая директория src/
    base_dir = Path(__file__).resolve().parents[1]

    # Папка с уже построенным индексом
    vector_store_dir = base_dir / "prepare_db" / "vector_store"

    vectorizer = HashVectorizer(dimension=256)

    retriever = SemanticRetriever(
        vectorizer=vectorizer,
        data_path=vector_store_dir / "data.json",
    )

    print("RAG CLI запущен.\n")

    # Список предопределённых запросов
    predefined_queries = [
        "Число учреждений дошкольного образования",
        "Сколько человек в Беларуси",
        "Сколько человек в Минске было в 2025 году"
    ]

    if not predefined_queries_flag and not user_queries_flag:
        print("Оба режима выключены. Задайте хотя бы один режим: predefined_queries=True или user_queries=True")
        return

    try:
        # Предопределённые запросы
        if predefined_queries_flag:
            for query in predefined_queries:
                print(f"> {query}")
                results = retriever.search(query, top_k=3)

                if not results:
                    print("Ничего не найдено.\n")
                    continue

                print("Топ-3 совпадения:")
                for i, r in enumerate(results, start=1):
                    print(f"\n{i}. {r['text']}")
                    print(f"   score: {r['score']:.2f}")
                    print(f"   source: {r['source']}")
                    print(f"   page: {r['page']}")
                print()

        # Пользовательский интерактив
        if user_queries_flag:
            print("Теперь можно вводить свои запросы (Ctrl+C для выхода):\n")
            while True:
                query = input("> ").strip()
                if not query:
                    continue

                results = retriever.search(query, top_k=3)
                if not results:
                    print("Ничего не найдено.\n")
                    continue

                print("Топ-3 совпадения:")
                for i, r in enumerate(results, start=1):
                    print(f"\n{i}. {r['text']}")
                    print(f"   score: {r['score']:.2f}")
                    print(f"   source: {r['source']}")
                    print(f"   page: {r['page']}")
                print()

    except KeyboardInterrupt:
        print("\nВыход из программы.")


if __name__ == "__main__":
    main()
