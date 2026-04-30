from pathlib import Path
from typing import List, Optional
import json
import time

from src.core.models import Chunk
from src.ingestion.pdf_chunker import PDFChunker
from src.ingestion.section_mapper import SectionMapper
from src.enrichers.ollama_client import OllamaClient, OllamaConfig
from src.enrichers.llm_enricher import LLMEnricher
from src.utils.post_processor import EnrichmentPostProcessor
from src.vectorstore.vectorizer import SentenceVectorizer
from src.vectorstore.faiss_store import FAISSStore


class KnowledgeBaseBuilder:
    """
    Построение базы знаний: PDF → Chunk → LLM-обогащение → FAISS.

    Каждый запуск полностью пересоздаёт индекс и data.json.
    """

    def __init__(
        self,
        documents_dir: Path,
        output_dir: Path,
        llm_model: str = "llama3-chatqa:8b",
        llm_client: Optional[OllamaClient] = None,
    ):
        self.documents_dir = documents_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if llm_client is None:
            ollama_config = OllamaConfig(model=llm_model)
            self.llm_client = OllamaClient(config=ollama_config)
        else:
            self.llm_client = llm_client

        self.pdf_chunker = PDFChunker()
        self.llm_enricher = LLMEnricher(llm_client=self.llm_client)
        self.post_processor = EnrichmentPostProcessor()
        self.vectorizer = SentenceVectorizer()
        self.faiss_indexer = FAISSStore(vectorizer=self.vectorizer)

    def build(self) -> None:
        chunk_id_counter = 0
        all_chunks: List[Chunk] = []
        build_start = time.perf_counter()

        pdf_files = sorted(self.documents_dir.glob("*.pdf"))
        for pdf_path in pdf_files:
            pdf_name = pdf_path.name
            pdf_start = time.perf_counter()

            raw_chunks = self.pdf_chunker.chunk_pdf(pdf_path)
            for ch in raw_chunks:
                ch.id = f"{pdf_name}::page{ch.page}::chunk{chunk_id_counter}"
                chunk_id_counter += 1

            section_mapper = SectionMapper(pdf_name, raw_chunks)
            raw_chunks = section_mapper.apply_to_chunks(raw_chunks)

            enriched_chunks = self.llm_enricher.enrich_chunks(pdf_name, raw_chunks, show_progress=True)
            processed_chunks = [self.post_processor.process_chunk(ch) for ch in enriched_chunks]
            all_chunks.extend(processed_chunks)

            pdf_elapsed = time.perf_counter() - pdf_start
            n_pages = len(processed_chunks)
            avg_per_page = pdf_elapsed / max(n_pages, 1)
            print(
                f"[{pdf_name}] {n_pages} страниц за {pdf_elapsed:.1f}s "
                f"(в среднем {avg_per_page:.2f}s/стр)"
            )

        t0 = time.perf_counter()
        self.faiss_indexer.add_chunks(all_chunks)
        embed_elapsed = time.perf_counter() - t0

        self._save_data_json(all_chunks)
        self.faiss_indexer.save(self.output_dir / "index.faiss")
        self._save_metadata(all_chunks)

        total_elapsed = time.perf_counter() - build_start
        print(
            f"\nБаза знаний построена: {len(all_chunks)} чанков, "
            f"dim={self.vectorizer.dimension}\n"
            f"  Embedding:  {embed_elapsed:.1f}s\n"
            f"  Всего:      {total_elapsed:.1f}s"
        )

    def _save_data_json(self, chunks: List[Chunk]) -> None:
        data_path = self.output_dir / "data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
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
                    for ch in chunks
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _save_metadata(self, chunks: List[Chunk]) -> None:
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vectorizer": type(self.vectorizer).__name__,
                    "model": self.vectorizer.model_name,
                    "dimension": self.vectorizer.dimension,
                    "chunks": len(chunks),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
