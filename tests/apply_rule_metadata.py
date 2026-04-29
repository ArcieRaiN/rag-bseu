from __future__ import annotations

"""
Apply deterministic metadata extraction to an existing data.json.

This script is used for v4 experiments where we need a reproducible,
non-LLM enrichment pass over already extracted PDF page text.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.models import Chunk
from src.enrichers.rule_metadata_extractor import RuleMetadataExtractor
from src.ingestion.section_mapper import SectionMapper
from src.utils.post_processor import EnrichmentPostProcessor
from src.vectorstore.faiss_store import FAISSStore
from src.vectorstore.vectorizer import SentenceVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent


def row_to_chunk(item: Dict[str, Any]) -> Chunk:
    return Chunk(
        id=str(item["id"]),
        search_context=item.get("search_context", ""),
        text=item.get("text", ""),
        source=item["source"],
        page=int(item["page"]),
        section=item.get("section"),
        geo=item.get("geo"),
        metrics=item.get("metrics"),
        units=item.get("units"),
        years=item.get("years") or [],
        extra=item.get("extra"),
        metadata_quality=item.get("metadata_quality"),
    )


def chunk_to_row(ch: Chunk) -> Dict[str, Any]:
    return {
        "id": ch.id,
        "search_context": ch.search_context,
        "text": ch.text,
        "source": ch.source,
        "page": ch.page,
        "section": ch.section,
        "geo": ch.geo,
        "metrics": ch.metrics,
        "units": ch.units,
        "years": ch.years,
        "extra": ch.extra,
        "metadata_quality": ch.metadata_quality,
    }


def load_chunks(data_path: Path, *, only_documents: bool) -> List[Chunk]:
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    if only_documents:
        documents = {p.name for p in (BASE_DIR / "usage" / "documents").glob("*.pdf")}
        rows = [row for row in rows if row.get("source") in documents]
    return [row_to_chunk(row) for row in rows]


def apply_rules(chunks: List[Chunk], *, overwrite: bool) -> Dict[str, Any]:
    extractor = RuleMetadataExtractor()
    post = EnrichmentPostProcessor()

    by_source: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)

    section_stats = {}
    for source, source_chunks in by_source.items():
        mapper = SectionMapper(source, source_chunks)
        mapper.apply_to_chunks(source_chunks)
        section_stats[source] = sum(1 for ch in source_chunks if ch.section)

    started = time.perf_counter()
    for chunk in chunks:
        extractor.apply_to_chunk(chunk, overwrite=overwrite)
        post.process_chunk(chunk)
    elapsed = time.perf_counter() - started

    return {
        "rule_elapsed_s": round(elapsed, 3),
        "avg_rule_seconds_per_chunk": round(elapsed / max(len(chunks), 1), 6),
        "section_mapped_by_source": section_stats,
    }


def save_vector_store(chunks: List[Chunk], output_dir: Path, *, rebuild_index: bool) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.json"
    data_path.write_text(
        json.dumps([chunk_to_row(ch) for ch in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "chunks": len(chunks),
        "sources": len({ch.source for ch in chunks}),
        "schema_version": "4.0",
        "enrichment": "rules+existing-llm",
    }

    if rebuild_index:
        vectorizer = SentenceVectorizer()
        store = FAISSStore(vectorizer=vectorizer)
        t0 = time.perf_counter()
        store.add_chunks(chunks)
        index_elapsed = time.perf_counter() - t0
        store.save(output_dir / "index.faiss")
        metadata.update(
            {
                "vectorizer": type(vectorizer).__name__,
                "model": vectorizer.model_name,
                "dimension": vectorizer.dimension,
                "index_elapsed_s": round(index_elapsed, 3),
            }
        )

    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply rule metadata extraction to data.json")
    parser.add_argument("--input", default=str(BASE_DIR / "usage" / "vector_store" / "data.json"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "usage" / "vector_store"))
    parser.add_argument("--only-documents", action="store_true", help="Use only PDFs currently present in usage/documents")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metadata instead of merging")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--report", default=str(BASE_DIR / "reports" / "v4" / "rule_metadata_report.json"))
    args = parser.parse_args()

    chunks = load_chunks(Path(args.input), only_documents=args.only_documents)
    stats = apply_rules(chunks, overwrite=args.overwrite)
    metadata = save_vector_store(chunks, Path(args.output_dir), rebuild_index=args.rebuild_index)

    report = {
        "input": args.input,
        "output_dir": args.output_dir,
        "only_documents": args.only_documents,
        "overwrite": args.overwrite,
        "rebuild_index": args.rebuild_index,
        "chunks": len(chunks),
        "stats": stats,
        "metadata": metadata,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
