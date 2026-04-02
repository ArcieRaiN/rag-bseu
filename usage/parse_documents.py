"""
USAGE: кнопка "Parse Documents".

Запускает pipeline парсинга документов
и сохраняет PDF в usage/archive_documents (см. main).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.pipelines.parse_documents_pipeline import ParseDocumentsPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("parse_documents")


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent  # rag-bseu

    documents_dir = root_dir / "usage" / "archive_documents"

    pipeline = ParseDocumentsPipeline(
        output_dir=documents_dir,
        max_pages=1,
    )

    log.info(
        "Starting ParseDocumentsPipeline output_dir=%s max_pages=%s",
        documents_dir,
        pipeline.site_parser.max_pages,
    )
    t0 = time.perf_counter()
    paths = pipeline.run()
    elapsed = time.perf_counter() - t0

    total_bytes = 0
    for p in paths:
        try:
            if p.exists() and p.is_file():
                total_bytes += p.stat().st_size
        except OSError:
            pass

    log.info(
        "Finished in %.2fs | downloaded_files=%d | total_size_bytes=%d",
        elapsed,
        len(paths),
        total_bytes,
    )
    print(
        f"[parse_documents] wall_time={elapsed:.2f}s files={len(paths)} total_bytes={total_bytes}"
    )


if __name__ == "__main__":
    main()
