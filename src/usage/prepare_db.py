from pathlib import Path
import json

from src.prepare_db.chunk_maker import ChunkMaker
from src.main.vectorizer import HashVectorizer


def main():
    # Корень src/
    src_dir = Path(__file__).resolve().parent.parent  # rag-bseu/src

    # PDF-документы
    docs_dir = src_dir / "prepare_db" / "documents"

    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Нет PDF-файлов в {docs_dir}")
        return

    print(f"📄 Найдено PDF-файлов: {len(pdf_files)}")

    # ⚠️ УВЕЛИЧЕННАЯ РАЗМЕРНОСТЬ
    vectorizer = HashVectorizer(dimension=256)

    # vector_store
    vector_store_dir = src_dir / "prepare_db" / "vector_store"
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    print("🔧 Строим семантический индекс из PDF...")
    chunk_maker = ChunkMaker(
        vectorizer=vectorizer,
        documents_dir=docs_dir,
        min_words=20,
    )

    artifacts = chunk_maker.build_from_pdfs(output_dir=vector_store_dir)

    print("✅ Индекс построен!")
    print(f"📁 vector_store: {vector_store_dir}")

    with open(artifacts.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📊 Всего сохранено семантических чанков: {len(data)}")


if __name__ == "__main__":
    main()
