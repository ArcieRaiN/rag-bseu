from pathlib import Path
from typing import List, Optional
import json

from src.core.models import Chunk
from src.ingestion.pdf_chunker import PDFChunker
from src.ingestion.section_mapper import SectionMapper
from src.enrichers.client import OllamaClient, OllamaConfig
from src.enrichers.enrichers import LLMEnricher
from src.utils.post_processor import EnrichmentPostProcessor
from src.vectorstore.vectorizer import SentenceVectorizer
from src.vectorstore.faiss_store import FAISSStore


class KnowledgeBaseBuilder:
    """
    Pipeline для построения базы знаний (PDF -> Chunk -> LLM -> FAISS).

    Full rebuild: каждый запуск полностью пересоздает индекс.
    """

    def __init__(
        self,
        documents_dir: Path,
        output_dir: Path,
        llm_model: str = "llama3-chatqa:latest",
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

        pdf_files = sorted(self.documents_dir.glob("*.pdf"))
        for pdf_path in pdf_files:
            pdf_name = pdf_path.name

            raw_chunks = self.pdf_chunker.chunk_pdf(pdf_path)
            for ch in raw_chunks:
                ch.id = f"{pdf_name}::page{ch.page}::chunk{chunk_id_counter}"
                chunk_id_counter += 1

            # Apply section mapping from TOC
            section_mapper = SectionMapper(pdf_name, raw_chunks)
            raw_chunks = section_mapper.apply_to_chunks(raw_chunks)

            enriched_chunks = self.llm_enricher.enrich_chunks(pdf_name, raw_chunks, show_progress=True)
            processed_chunks = [self.post_processor.process_chunk(ch) for ch in enriched_chunks]
            all_chunks.extend(processed_chunks)
            print(f"Processed PDF {pdf_name}: {len(processed_chunks)} chunks")

        # Build FAISS index over all chunks at once (positional)
        self.faiss_indexer.add_chunks(all_chunks)

        self._save_data_json(all_chunks)
        self.faiss_indexer.save(self.output_dir / "index.faiss")
        self._save_metadata(all_chunks)
        print(f"Knowledge base built: {len(all_chunks)} chunks, dim={self.vectorizer.dimension}")

    def _save_data_json(self, chunks: List[Chunk]) -> None:
        data_path = self.output_dir / "data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": ch.id,
                        "context": ch.context,
                        "text": ch.text,
                        "source": ch.source,
                        "page": ch.page,
                        "section": ch.section,
                        "geo": ch.geo,
                        "metrics": ch.metrics,
                        "years": ch.years,
                        "time_granularity": ch.time_granularity,
                        "oked": ch.oked,
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
                    "dimension": self.vectorizer.dimension,
                    "chunks": len(chunks),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
