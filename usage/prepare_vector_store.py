"""
USAGE: построение векторной базы знаний.

Парсит PDF из usage/documents и строит FAISS индекс + data.json.
"""

from pathlib import Path
from src.pipelines.knowledge_base_builder_pipeline import KnowledgeBaseBuilder

def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent

    documents_dir = root_dir / "usage" / "documents"
    output_dir = root_dir / "usage" / "vector_store"

    builder = KnowledgeBaseBuilder(
        documents_dir=documents_dir,
        output_dir=output_dir,
        llm_model="llama3-chatqa:latest",
    )

    builder.build()

if __name__ == "__main__":
    main()
