"""
PIPELINE 1: подготовка базы знаний через новый пайплайн.

Использует:
- LlamaIndex для чанкинга PDF
- Ollama для батчевого enrichment чанков
- FAISS для векторного индекса
"""

from pathlib import Path

from src.prepare_db.knowledge_builder import KnowledgeBaseBuilder, BuildConfig
from src.main.ollama_client import OllamaClient, OllamaConfig


def main() -> None:
    """Entrypoint для подготовки базы знаний через новый пайплайн."""
    # Корень src/
    src_dir = Path(__file__).resolve().parent.parent  # rag-bseu/src

    # Папка с PDF-документами
    docs_dir = src_dir / "usage" / "documents"

    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Нет PDF-файлов в {docs_dir}")
        return

    print(f"📄 Найдено PDF-файлов: {len(pdf_files)}")

    # Папка для индекса
    vector_store_dir = src_dir / "usage" / "vector_store"
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    print("🔧 Строим базу знаний через LlamaIndex + Ollama enrichment...")
    print("   (Это может занять некоторое время из-за LLM-запросов)")

    # Конфигурация
    config = BuildConfig(
        documents_dir=docs_dir,
        output_dir=vector_store_dir,
        vector_dim=256,
    )

    # Создаём Ollama клиент с моделью llama3-chatqa:latest
    ollama_config = OllamaConfig(model="llama3-chatqa:latest")
    llm_client = OllamaClient(config=ollama_config)

    # Строим базу знаний
    builder = KnowledgeBaseBuilder(config=config, llm_client=llm_client)
    builder.build()

    print("✅ База знаний построена!")
    print(f"📁 Индекс сохранён в: {vector_store_dir}")

if __name__ == "__main__":
    main()
