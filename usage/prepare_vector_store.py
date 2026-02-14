"""
USAGE: кнопка "Prepare Vector Store".

Просто парсит папку с PDF и вызывает pipeline из pipelines/knowledge_base_builder_pipeline.py.
"""

from pathlib import Path
from src.pipelines.knowledge_base_builder_pipeline import KnowledgeBaseBuilder

def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent  # rag-bseu

    documents_dir = root_dir / "usage" / "documents"
    output_dir = root_dir / "usage" / "vector_store"

    builder = KnowledgeBaseBuilder(
        documents_dir=documents_dir,
        output_dir=output_dir,
        llm_model="llama3-chatqa:latest",
        vector_dim=256,
    )

    builder.build()

if __name__ == "__main__":
    main()
