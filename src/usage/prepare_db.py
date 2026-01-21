from pathlib import Path
from src.prepare_db.chunk_maker import ChunkMaker
from src.main.vectorizer import HashVectorizer


def main():
    # Папка src
    src_dir = Path(__file__).resolve().parent.parent  # rag-bseu/src

    # Папка с PDF
    docs_dir = src_dir / "prepare_db" / "documents"

    # Проверяем наличие PDF
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Нет PDF-файлов в {docs_dir}. Поместите PDF для индексации.")
        return
    else:
        print(f"Найдено {len(pdf_files)} PDF-файлов в {docs_dir}")

    # Векторизатор
    vectorizer = HashVectorizer(dimension=32)

    # Берём уже существующую папку vector_store
    vector_store_dir = src_dir / "prepare_db" / "vector_store"

    print("Строим индекс из PDF...")
    chunk_maker = ChunkMaker(vectorizer, documents_dir=docs_dir)
    chunk_maker.build_from_pdfs(output_dir=vector_store_dir)

    print(f"Индекс построен! Файлы сохранены в {vector_store_dir}")


if __name__ == "__main__":
    main()
