from __future__ import annotations

"""
PIPELINE 1: подготовка базы знаний (prepare_db).

Рефакторенная версия с разделением ответственности:
- PDFChunker: чанкинг PDF
- LLMEnricher: LLM enrichment с rolling context buffer
- FAISSIndexer: построение FAISS индекса
- ChunkValidator: валидация данных

Высокоуровневый фасад KnowledgeBaseBuilder координирует работу всех модулей.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json
import time

from src.main.models import Chunk
from src.main.ollama_client import OllamaClient
from src.main.vectorizer import HashVectorizer
from src.logs.logger import get_logger

from src.prepare_db.pdf_chunker import PDFChunker
from src.prepare_db.llm_enricher import LLMEnricher
from src.prepare_db.faiss_indexer import FAISSIndexer


@dataclass
class BuildConfig:
    """Конфигурация для построения базы знаний."""
    documents_dir: Path
    output_dir: Path
    vector_dim: int = 256


class KnowledgeBaseBuilder:
    """
    Высокоуровневый фасад для подготовки базы знаний.
    
    Координирует работу модулей:
    - PDFChunker для чанкинга PDF
    - LLMEnricher для обогащения чанков
    - FAISSIndexer для построения индекса
    """

    def __init__(self, config: BuildConfig, llm_client: OllamaClient | None = None):
        """
        Инициализация билдера.
        
        Args:
            config: Конфигурация построения базы знаний
            llm_client: Клиент Ollama (опционально)
        """
        self._config = config
        self._llm = llm_client or OllamaClient()
        self._vectorizer = HashVectorizer(dimension=config.vector_dim)
        self._logger = get_logger()
        
        # Инициализируем модули
        self._pdf_chunker = PDFChunker(
            chunk_size=1500,
            chunk_overlap=200,
            paragraph_separator="\n\n",
        )
        self._llm_enricher = LLMEnricher(
            llm_client=self._llm,
            batch_size=5,
            batch_concurrency=1,
            context_buffer_size=10,
            reset_interval=50,
        )
        self._faiss_indexer = FAISSIndexer(vectorizer=self._vectorizer)

    def build(self) -> None:
        """
        Основной entrypoint для подготовки базы знаний.
        
        Процесс:
        1. Чанкинг всех PDF через PDFChunker
        2. Обогащение чанков через LLMEnricher (с rolling context buffer)
        3. Сохранение data.json
        4. Построение FAISS индекса через FAISSIndexer
        """
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Логируем начало процесса
        total_pdfs = len(list(self._config.documents_dir.glob("*.pdf")))
        self._logger.log_prepare_db("start", total_pdfs=total_pdfs)
        build_start_time = time.time()

        all_chunks: List[Chunk] = []
        chunk_id_counter = 0

        # Обрабатываем каждый PDF
        for pdf_path in sorted(self._config.documents_dir.glob("*.pdf")):
            pdf_start_time = time.time()
            self._logger.log_prepare_db("pdf_start", pdf_name=pdf_path.name)
            
            # 1. Чанкинг PDF через PDFChunker
            raw_chunks = self._pdf_chunker.chunk_pdf(pdf_path)

            # Присваиваем id на уровне всего корпуса
            for ch in raw_chunks:
                ch.id = f"{pdf_path.name}::page{ch.page}::chunk{chunk_id_counter}"
                chunk_id_counter += 1

            # 2. LLM-enrichment через LLMEnricher (использует rolling context buffer)
            enriched_chunks = self._llm_enricher.enrich_chunks(
                pdf_path.name,
                raw_chunks,
                skip_first_pages=3,
            )
            all_chunks.extend(enriched_chunks)
            
            pdf_elapsed = time.time() - pdf_start_time
            self._logger.log_prepare_db(
                "pdf_end",
                pdf_name=pdf_path.name,
                chunks_count=len(enriched_chunks),
                elapsed_time=pdf_elapsed,
            )

        # 3. Сохранение data.json
        self._save_data_json(all_chunks)

        # 4. Построение FAISS индекса через FAISSIndexer
        self._faiss_indexer.build_index(
            all_chunks,
            self._config.output_dir / "index.faiss",
        )

        # 5. Сохранение метаданных
        self._save_metadata(all_chunks)
        
        # Логируем завершение процесса
        build_elapsed = time.time() - build_start_time
        self._logger.log_prepare_db(
            "end",
            total_chunks=len(all_chunks),
            elapsed_time=build_elapsed,
        )

    def _save_data_json(self, chunks: List[Chunk]) -> None:
        """
        Сохраняет чанки в data.json.
        
        Args:
            chunks: Список чанков для сохранения
        """
        data_json_path = self._config.output_dir / "data.json"
        with open(data_json_path, "w", encoding="utf-8") as f:
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
        """
        Сохраняет метаданные индекса.
        
        Args:
            chunks: Список чанков
        """
        meta_path = self._config.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vectorizer": type(self._vectorizer).__name__,
                    "dimension": self._vectorizer.dimension,
                    "chunks": len(chunks),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
