"""
Сравнение embedding-моделей на тестах v3.

Для каждой модели перестраивает только FAISS-индекс из существующего data.json,
затем прогоняет evaluator v3 (182 вопроса).

Usage: python -m tests.compare_embeddings
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.models import Chunk
from src.vectorstore.vectorizer import SentenceVectorizer
from tests.evaluator import (
    load_tests,
    run_evaluation,
    print_summary,
    save_results,
    TEST_DATA_PATH,
    RESULTS_DIR,
)
from src.pipelines.query import QueryPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "usage" / "vector_store"
DATA_JSON = VECTOR_STORE_DIR / "data.json"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"


@dataclass
class ModelSpec:
    name: str
    hf_name: str


MODELS: List[ModelSpec] = [
    ModelSpec(name="MiniLM-384d", hf_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ModelSpec(name="bge-m3-1024d", hf_name="BAAI/bge-m3"),
    ModelSpec(name="USER-bge-m3-1024d", hf_name="deepvk/USER-bge-m3"),
    ModelSpec(name="e5-large-1024d", hf_name="intfloat/multilingual-e5-large"),
]


def load_chunks() -> List[Chunk]:
    with open(DATA_JSON, "r", encoding="utf-8") as f:
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
    return chunks


def build_embed_text(chunk: Chunk) -> str:
    context = (chunk.context or "").strip()
    text_prefix = (chunk.text or "")[:500].strip()
    if context and text_prefix:
        return f"{context}\n{text_prefix}"
    return context or text_prefix or ""


def rebuild_index(chunks: List[Chunk], vectorizer: SentenceVectorizer) -> float:
    """Rebuild FAISS index, return elapsed seconds."""
    t0 = time.perf_counter()
    texts = [build_embed_text(ch) for ch in chunks]
    embeddings = vectorizer.embed_many(texts, is_query=False)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    elapsed = time.perf_counter() - t0
    return elapsed


def run_for_model(spec: ModelSpec, chunks: List[Chunk], tests) -> Dict:
    print(f"\n{'='*60}")
    print(f"  Model: {spec.name} ({spec.hf_name})")
    print(f"{'='*60}")

    print("  Loading model...")
    vectorizer = SentenceVectorizer(model_name=spec.hf_name)
    print(f"  Dimension: {vectorizer.dimension}")

    print("  Building index...")
    embed_time = rebuild_index(chunks, vectorizer)
    print(f"  Index built in {embed_time:.1f}s")

    print("  Running evaluator...")
    pipeline = QueryPipeline.__new__(QueryPipeline)
    pipeline._base_dir = BASE_DIR
    pipeline._vectorizer = vectorizer

    from src.retrieval.semantic_search import FaissSemanticSearcher
    from src.core.config import RetrievalConfig
    from src.retrieval.hybrid_search import HybridSearcher
    from src.core.context_enrichment import QueryContextEnricher

    pipeline._semantic = FaissSemanticSearcher(
        index_path=INDEX_PATH,
        data_path=DATA_JSON,
    )
    config = RetrievalConfig()
    pipeline._retrieval_config = config
    pipeline._hybrid = HybridSearcher(
        semantic_searcher=pipeline._semantic,
        config=config,
    )
    pipeline._enricher = QueryContextEnricher(vectorizer=vectorizer)
    pipeline._reranker = None

    summary = run_evaluation(tests, pipeline)
    print_summary(summary)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = spec.name.replace("/", "_").replace(" ", "_")
    save_results(summary, RESULTS_DIR / f"embed_{safe_name}_{timestamp}.json")

    return {
        "model": spec.name,
        "hf_name": spec.hf_name,
        "dimension": vectorizer.dimension,
        "embed_time_s": round(embed_time, 1),
        "hit_at_1": round(summary.hit_rate_1, 4),
        "hit_at_3": round(summary.hit_rate_3, 4),
        "hit_at_5": round(summary.hit_rate_5, 4),
        "mrr": round(summary.mrr, 4),
        "avg_query_s": round(summary.avg_time, 4),
    }


def main():
    print("Loading chunks from data.json...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    tests = load_tests(TEST_DATA_PATH)
    print(f"Loaded {len(tests)} tests")

    results: List[Dict] = []
    for spec in MODELS:
        try:
            r = run_for_model(spec, chunks, tests)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR for {spec.name}: {e}")
            results.append({"model": spec.name, "error": str(e)})

    print(f"\n\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Dim':>5} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7} {'Embed':>7} {'Query':>7}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<25}  ERROR: {r['error']}")
            continue
        print(
            f"{r['model']:<25} {r['dimension']:>5} "
            f"{r['hit_at_1']:>6.1%} {r['hit_at_3']:>6.1%} {r['hit_at_5']:>6.1%} "
            f"{r['mrr']:>6.3f} {r['embed_time_s']:>6.1f}s {r['avg_query_s']:>6.3f}s"
        )

    comparison_path = RESULTS_DIR / f"comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nComparison saved to {comparison_path}")

    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["hit_at_5"])
        print(f"\nBest model by Hit@5: {best['model']} ({best['hit_at_5']:.1%})")


if __name__ == "__main__":
    main()
