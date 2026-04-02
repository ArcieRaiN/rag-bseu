"""
USAGE: построение векторной базы знаний.

Парсит PDF из usage/documents и строит FAISS индекс + data.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.pipelines.knowledge_base_builder_pipeline import KnowledgeBaseBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("prepare_vector_store")


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent

    documents_dir = root_dir / "usage" / "documents"
    output_dir = root_dir / "usage" / "vector_store"

    pdf_count = len(list(documents_dir.glob("*.pdf")))
    log.info(
        "KnowledgeBaseBuilder start documents_dir=%s output_dir=%s pdf_files=%d llm=%s",
        documents_dir,
        output_dir,
        pdf_count,
        "llama3-chatqa:latest",
    )

    builder = KnowledgeBaseBuilder(
        documents_dir=documents_dir,
        output_dir=output_dir,
        llm_model="llama3-chatqa:latest",
    )

    t0 = time.perf_counter()
    builder.build()
    wall_s = time.perf_counter() - t0

    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log.info(
            "metadata: chunks=%s model=%s dimension=%s",
            meta.get("chunks"),
            meta.get("model"),
            meta.get("dimension"),
        )
        print(
            f"[prepare_vector_store] wall_time={wall_s:.1f}s "
            f"chunks={meta.get('chunks')} dim={meta.get('dimension')}"
        )
    else:
        log.warning("metadata.json not found after build at %s", meta_path)
        print(f"[prepare_vector_store] wall_time={wall_s:.1f}s (metadata missing)")


if __name__ == "__main__":
    main()
