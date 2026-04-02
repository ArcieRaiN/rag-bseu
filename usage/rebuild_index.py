"""
Пересборка FAISS-индекса из data.json без повторного LLM-обогащения.

Usage: python usage/rebuild_index.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.models import Chunk
from src.ingestion.section_mapper import SectionMapper
from src.vectorstore.vectorizer import SentenceVectorizer
from src.vectorstore.faiss_store import FAISSStore


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    vector_store_dir = root_dir / "usage" / "vector_store"
    data_path = vector_store_dir / "data.json"
    index_path = vector_store_dir / "index.faiss"

    print(f"Loading chunks from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    chunks = []
    for item in raw:
        chunks.append(Chunk(
            id=str(item["id"]),
            context=item.get("context", ""),
            text=item.get("text", ""),
            source=item["source"],
            page=int(item["page"]),
            section=item.get("section"),
            geo=item.get("geo"),
            metrics=item.get("metrics"),
            years=item.get("years") or [],
        ))

    print(f"Loaded {len(chunks)} chunks")

    # Apply section mapping from TOC (enrich context with section names)
    sources = set(ch.source for ch in chunks)
    for source in sources:
        source_chunks = [ch for ch in chunks if ch.source == source]
        mapper = SectionMapper(source, source_chunks)
        mapper.apply_to_chunks(source_chunks)
        applied = sum(1 for ch in source_chunks if ch.section)
        print(f"  Section mapping for {source}: {applied}/{len(source_chunks)} chunks mapped")

    # Save updated data.json with section field
    updated_data = []
    for ch in chunks:
        item = {
            "id": ch.id,
            "context": ch.context,
            "text": ch.text,
            "source": ch.source,
            "page": ch.page,
            "section": ch.section,
            "geo": ch.geo,
            "metrics": ch.metrics,
            "years": ch.years,
        }
        updated_data.append(item)

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)
    print(f"Updated data.json with section mappings")

    print("Initializing vectorizer...")
    vectorizer = SentenceVectorizer()
    print(f"Model: {vectorizer.model_name}, dimension: {vectorizer.dimension}")

    store = FAISSStore(vectorizer=vectorizer)
    print("Building FAISS index...")
    store.add_chunks(chunks)

    store.save(index_path)
    print(f"Index saved to {index_path} ({store.index.ntotal} vectors)")

    meta_path = vector_store_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "vectorizer": type(vectorizer).__name__,
            "model": vectorizer.model_name,
            "dimension": vectorizer.dimension,
            "chunks": len(chunks),
        }, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
