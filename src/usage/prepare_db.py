"""
PIPELINE 1: подготовка базы знаний через новый пайплайн.

Использует:
- LlamaIndex для чанкинга PDF
- Ollama для батчевого enrichment чанков
- FAISS для векторного индекса
"""

from pathlib import Path

from src.prepare_db.knowledge_builder import KnowledgeBaseBuilder, BuildConfig


def main() -> None:
    """Entrypoint для подготовки базы знаний через новый пайплайн."""
    # Корень src/
    src_dir = Path(__file__).resolve().parent.parent  # rag-bseu/src

    # Папка с PDF-документами
    docs_dir = src_dir / "prepare_db" / "documents"

    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Нет PDF-файлов в {docs_dir}")
        return

    print(f"📄 Найдено PDF-файлов: {len(pdf_files)}")

    # Папка для индекса
    vector_store_dir = src_dir / "prepare_db" / "vector_store"
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    print("🔧 Строим базу знаний через LlamaIndex + Ollama enrichment...")
    print("   (Это может занять некоторое время из-за LLM-запросов)")

    # Конфигурация
    config = BuildConfig(
        documents_dir=docs_dir,
        output_dir=vector_store_dir,
        vector_dim=256,
    )

    # Строим базу знаний
    builder = KnowledgeBaseBuilder(config=config)
    builder.build()

    print("✅ База знаний построена!")
    print(f"📁 vector_store: {vector_store_dir}")

    # Диагностика
    import json
    data_path = vector_store_dir / "data.json"
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"📊 Всего чанков проиндексировано: {len(chunks)}")

        # Полезный дебаг: примеры context
        print("\n🧪 Примеры context (первые 10-13 чанки):")
        for ch in chunks[10:14]:
            context_preview = ch.get("context", "")[:100]
            print(f" • {context_preview}...")


if __name__ == "__main__":
    main()
