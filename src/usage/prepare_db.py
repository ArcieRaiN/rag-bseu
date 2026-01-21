from pathlib import Path
from src.prepare_db.chunk_maker import ChunkMaker
from src.main.vectorizer import HashVectorizer
import json


def main():
    # Папка src/
    src_dir = Path(__file__).resolve().parent.parent  # rag-bseu/src

    # Папка с PDF-файлами
    docs_dir = src_dir / "prepare_db" / "documents"

    # Проверка наличия PDF
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Нет PDF-файлов в {docs_dir}. Поместите PDF для индексации.")
        return
    else:
        print(f"Найдено {len(pdf_files)} PDF-файлов в {docs_dir}")

    # Векторизатор
    vectorizer = HashVectorizer(dimension=32)

    # Папка для сохранения векторного хранилища
    vector_store_dir = src_dir / "prepare_db" / "vector_store"

    print("Строим индекс из PDF...")
    chunk_maker = ChunkMaker(vectorizer, documents_dir=docs_dir)
    artifacts = chunk_maker.build_from_pdfs(output_dir=vector_store_dir)

    print(f"Индекс построен! Файлы сохранены в {vector_store_dir}")

    # Считаем количество таблиц
    with open(artifacts.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Всего таблиц обработано: {len(data)}")


if __name__ == "__main__":
    main()
