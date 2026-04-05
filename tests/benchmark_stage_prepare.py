"""
Замер времени подготовительного этапа: один PDF → чанки → LLM-обогащение → FAISS.

Требования: PDF в каталоге (по умолчанию копируется первый файл из usage/documents/),
работающий Ollama с моделью llama3-chatqa:latest, свободное место для выходного индекса.

Результат пишется в tests/results/benchmark_prepare_<timestamp>.json и дублируется в stdout.

Usage:
    python -m tests.benchmark_stage_prepare
    python -m tests.benchmark_stage_prepare --pdf "C:\\path\\to\\file.pdf"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.knowledge_base_builder import KnowledgeBaseBuilder

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DOCUMENTS_DIR = BASE_DIR / "usage" / "documents"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark preparatory stage (one PDF)")
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to a single PDF (otherwise first *.pdf from usage/documents/)",
    )
    args = parser.parse_args()

    if args.pdf:
        pdf_path = Path(args.pdf).resolve()
        if not pdf_path.is_file():
            print(f"ERROR: file not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        work_root = tempfile.mkdtemp(prefix="rag_benchmark_prepare_")
        tmp_docs = Path(work_root) / "docs"
        tmp_docs.mkdir(parents=True)
        dest = tmp_docs / pdf_path.name
        shutil.copy2(pdf_path, dest)
        documents_dir = tmp_docs
        cleanup = Path(work_root)
    else:
        pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
        if not pdfs:
            print(
                f"ERROR: no PDF files in {DOCUMENTS_DIR}. "
                "Add a PDF or pass --pdf path.",
                file=sys.stderr,
            )
            sys.exit(1)
        pdf_path = pdfs[0]
        work_root = tempfile.mkdtemp(prefix="rag_benchmark_prepare_")
        tmp_docs = Path(work_root) / "docs"
        tmp_docs.mkdir(parents=True)
        shutil.copy2(pdf_path, tmp_docs / pdf_path.name)
        documents_dir = tmp_docs
        cleanup = Path(work_root)

    out_root = tempfile.mkdtemp(prefix="rag_benchmark_out_")
    output_dir = Path(out_root) / "vector_store"
    output_dir.mkdir(parents=True)

    print(f"Documents dir: {documents_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"PDF:           {pdf_path.name}")

    builder = KnowledgeBaseBuilder(
        documents_dir=documents_dir,
        output_dir=output_dir,
        llm_model="llama3-chatqa:latest",
    )

    t0 = time.perf_counter()
    builder.build()
    wall_s = time.perf_counter() - t0

    data_json = output_dir / "data.json"
    n_chunks = 0
    if data_json.exists():
        n_chunks = len(json.loads(data_json.read_text(encoding="utf-8")))

    per_page = wall_s / max(n_chunks, 1)

    report = {
        "stage": "preparatory",
        "pdf": pdf_path.name,
        "chunks_pages": n_chunks,
        "wall_time_s": round(wall_s, 2),
        "avg_seconds_per_page": round(per_page, 3),
        "output_dir": str(output_dir),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"benchmark_prepare_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    if cleanup and cleanup.exists():
        shutil.rmtree(cleanup, ignore_errors=True)
    shutil.rmtree(out_root, ignore_errors=True)


if __name__ == "__main__":
    main()
