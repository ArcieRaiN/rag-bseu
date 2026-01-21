from pathlib import Path

from src.main.retriever import TableRetriever
from src.main.vectorizer import HashVectorizer


def main() -> None:
    # Базовая директория src/
    base_dir = Path(__file__).resolve().parents[1]

    # Папка с уже построенным индексом
    vector_store_dir = base_dir / "prepare_db" / "vector_store"

    vectorizer = HashVectorizer(dimension=32)

    retriever = TableRetriever(
        vectorizer=vectorizer,
        index_path=vector_store_dir / "index.faiss",
        metadata_path=vector_store_dir / "metadata.json",
        data_path=vector_store_dir / "data.json",
    )

    print("RAG CLI запущен.")
    print("Введите запрос (Ctrl+C для выхода)\n")

    try:
        while True:
            query = input("> ").strip()
            if not query:
                continue

            results = retriever.search(query, top_k=3)

            if not results:
                print("Ничего не найдено.\n")
                continue

            print("\nТоп-3 результата:")
            for i, r in enumerate(results, start=1):
                print(f"\n{i}. {r.get('title')}")
                print(f"   score: {r.get('score'):.4f}")
                print(f"   source: {r.get('source')}")
                print(f"   page: {r.get('page')}")

            print()

    except KeyboardInterrupt:
        print("\nВыход из программы.")


if __name__ == "__main__":
    main()
