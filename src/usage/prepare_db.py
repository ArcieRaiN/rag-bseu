from pathlib import Path
import json

from src.prepare_db.chunk_maker import ChunkMaker
from src.main.vectorizer import HashVectorizer


def main() -> None:
    # Корень src/
    src_dir = Path(__file__).resolve().parent.parent  # rag-bseu/src

    # Папка с PDF-документами
    docs_dir = src_dir / "prepare_db" / "documents"

    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Нет PDF-файлов в {docs_dir}")
        return

    print(f"📄 Найдено PDF-файлов: {len(pdf_files)}")

    # ⚠️ ВАЖНО: увеличенная размерность
    vectorizer = HashVectorizer(dimension=256)

    # Папка для индекса
    vector_store_dir = src_dir / "prepare_db" / "vector_store"
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    print("🔧 Строим ТАБЛИЧНЫЙ семантический индекс из PDF...")

    # ChunkMaker теперь работает ТОЛЬКО с таблицами
    chunk_maker = ChunkMaker(
        vectorizer=vectorizer,
        documents_dir=docs_dir,

        # минимальная длина заголовка таблицы
        min_title_words=3,

        # игнорируем слишком короткие / мусорные таблицы
        min_rows=2,
        min_cols=2,
    )

    artifacts = chunk_maker.build_tables_from_pdfs(
        output_dir=vector_store_dir
    )

    print("✅ Индекс таблиц построен!")
    print(f"📁 vector_store: {vector_store_dir}")

    # Диагностика
    with open(artifacts.data_path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    print(f"📊 Всего таблиц проиндексировано: {len(tables)}")

    # полезный дебаг
    print("\n🧪 Примеры заголовков таблиц:")
    for t in tables[10:15]:
        print(" •", t["title"])


if __name__ == "__main__":
    main()
