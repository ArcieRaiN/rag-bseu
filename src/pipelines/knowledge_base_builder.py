from pathlib import Path
from typing import List, Optional
import json

from src.core.models import Chunk
from src.ingestion.pdf_chunker import PDFChunker
from src.enrichers.client import OllamaClient, OllamaConfig
from src.enrichers.enrichers import LLMEnricher
from src.utils.post_processor import EnrichmentPostProcessor
from src.vectorstore.vectorizer import SentenceVectorizer
from src.vectorstore.faiss_store import FAISSStore


class KnowledgeBaseBuilder:
    """
    Pipeline для построения базы знаний (PDF → Chunk → LLM → FAISS).

    Все шаги интегрированы в одном фасаде:
    1. Чанкинг PDF через PDFChunker
    2. Enrichment через LLMEnricher
    3. Post-processing через EnrichmentPostProcessor
    4. Построение FAISS индекса через FAISSStore
    5. Сохранение данных и метаданных
    """

    def __init__(
        self,
        documents_dir: Path,
        output_dir: Path,
        llm_model: str = "llama3-chatqa:latest",
        vector_dim: int = 256,
        llm_client: Optional[OllamaClient] = None,
    ):
        self.documents_dir = documents_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LLM
        if llm_client is None:
            ollama_config = OllamaConfig(model=llm_model)
            self.llm_client = OllamaClient(config=ollama_config)
        else:
            self.llm_client = llm_client

        # Модули
        self.pdf_chunker = PDFChunker()
        self.llm_enricher = LLMEnricher(llm_client=self.llm_client)
        self.post_processor = EnrichmentPostProcessor()
        self.vectorizer = SentenceVectorizer(dimension=vector_dim)
        self.faiss_indexer = FAISSStore(vectorizer=self.vectorizer)

    def build(self) -> None:
        """Основной метод: строим базу знаний из всех PDF в папке."""
        all_chunks: List[Chunk] = []
        chunk_id_counter = 0

        pdf_files = sorted(self.documents_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ Нет PDF-файлов в {self.documents_dir}")
            return

        print(f"📄 Найдено PDF-файлов: {len(pdf_files)}")
        print("🔧 Строим базу знаний через LlamaIndex + Ollama enrichment...")

        for pdf_path in pdf_files:
            # 1. Чанкинг PDF
            raw_chunks = self.pdf_chunker.chunk_pdf(pdf_path)

            # Назначаем уникальные id
            for ch in raw_chunks:
                ch.id = f"{pdf_path.name}::page{ch.page}::chunk{chunk_id_counter}"
                chunk_id_counter += 1

            # 2. LLM enrichment
            enriched_chunks = self.llm_enricher.enrich_chunks(
                pdf_path.name,
                raw_chunks,
                show_progress=True,
            )

            # 3. Post-processing
            processed_chunks = [self.post_processor.process_chunk(ch) for ch in enriched_chunks]

            all_chunks.extend(processed_chunks)

        # 4. Сохраняем data.json
        self._save_data_json(all_chunks)

        # 5. Строим FAISS индекс
        index_path = self.output_dir / "index.faiss"
        self.faiss_indexer.build_and_save(all_chunks, index_path)

        # 6. Сохраняем метаданные
        self._save_metadata(all_chunks)

        print("✅ База знаний построена!")
        print(f"📁 Индекс и data.json сохранены в: {self.output_dir}")

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
