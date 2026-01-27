from __future__ import annotations

"""
PIPELINE 1: подготовка базы знаний (prepare_db).

Задачи (высокоуровневый skeleton):
1. Взять PDF из src/prepare_db/documents/
2. Разбить документ на чанки с помощью LlamaIndex
3. Передать ВСЕ чанки в LLM (Ollama) ОДНИМ ЗАПРОСОМ
   и получить для каждого чанка:
   - context (краткое описание на основе всего документа)
   - geo / metrics / years / time_granularity / oked
4. Сохранить чанки в data.json
5. Построить embedding ТОЛЬКО для поля context
6. Загрузить embeddings в FAISS (index.faiss)

Реализация намеренно оставлена на уровне skeleton, чтобы:
- зафиксировать интерфейсы и архитектурные границы
- не смешивать здесь сетевой/IO‑код с бизнес‑логикой
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import shutil
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import threading

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from src.main.models import Chunk
from src.main.ollama_client import OllamaClient
from src.main.vectorizer import HashVectorizer
from src.logs.logger import get_logger


@dataclass
class BuildConfig:
    documents_dir: Path
    output_dir: Path
    vector_dim: int = 256


class KnowledgeBaseBuilder:
    """
    Высокоуровневый фасад для подготовки базы знаний.

    ВАЖНО:
    - этот класс НЕ занимается скачиванием PDF (site_parser.py остаётся заглушкой)
    - LlamaIndex интеграция описана как TODO‑интеграция (скелет)
    """

    def __init__(self, config: BuildConfig, llm_client: OllamaClient | None = None):
        self._config = config
        self._llm = llm_client or OllamaClient()
        self._vectorizer = HashVectorizer(dimension=config.vector_dim)
        self._logger = get_logger()

    # -------------------- Публичный интерфейс -------------------- #

    def build(self) -> None:
        """
        Основной entrypoint для подготовки базы знаний.

        Реальная реализация должна:
        - пройти по всем PDF
        - вызвать `_chunk_pdf_with_llamaindex`
        - затем `_enrich_chunks_with_llm_batch`
        - сохранить JSON и построить FAISS‑индекс
        """
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Логируем начало процесса
        self._logger.log_prepare_db("start", total_pdfs=len(list(self._config.documents_dir.glob("*.pdf"))))
        build_start_time = time.time()

        all_chunks: List[Chunk] = []
        chunk_id_counter = 0

        for pdf_path in sorted(self._config.documents_dir.glob("*.pdf")):
            pdf_start_time = time.time()
            self._logger.log_prepare_db("pdf_start", pdf_name=pdf_path.name)
            
            # 1–2. Чанкинг PDF через LlamaIndex (skeleton)
            raw_chunks = self._chunk_pdf_with_llamaindex(pdf_path)

            # Присваиваем id на уровне всего корпуса
            for ch in raw_chunks:
                ch.id = f"{pdf_path.name}::page{ch.page}::chunk{chunk_id_counter}"
                chunk_id_counter += 1

            # 3. LLM‑enrichment для чанков с контекстом предыдущих страниц
            # Передаем all_chunks для получения контекста предыдущих страниц
            enriched_chunks = self._enrich_chunks_with_llm_batch(
                pdf_path.name,
                raw_chunks,
                all_chunks=all_chunks + raw_chunks  # Все чанки до текущего PDF + текущие
            )
            all_chunks.extend(enriched_chunks)
            
            pdf_elapsed = time.time() - pdf_start_time
            self._logger.log_prepare_db(
                "pdf_end",
                pdf_name=pdf_path.name,
                chunks_count=len(enriched_chunks),
                elapsed_time=pdf_elapsed
            )

        # 4. Сохранение data.json
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
                    for ch in all_chunks
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 5–6. Embeddings по context + FAISS
        self._build_faiss_index(all_chunks, self._config.output_dir / "index.faiss")

        # Дополнительная мета‑информация
        meta_path = self._config.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vectorizer": type(self._vectorizer).__name__,
                    "dimension": self._vectorizer.dimension,
                    "chunks": len(all_chunks),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        
        # Логируем завершение процесса
        build_elapsed = time.time() - build_start_time
        self._logger.log_prepare_db(
            "end",
            total_chunks=len(all_chunks),
            elapsed_time=build_elapsed
        )

    # -------------------- Skeleton‑методы для интеграции -------------------- #

    def _chunk_pdf_with_llamaindex(self, pdf_path: Path) -> List[Chunk]:
        """
        Интеграция с LlamaIndex для чанкинга PDF.

        Использует:
        - PDFReader для чтения PDF
        - SimpleNodeParser для разбиения на чанки (chunk_size=512, chunk_overlap=50)
        - Извлечение метаданных страницы из node.metadata

        Возвращает список Chunk с заполненными:
        - text (текст чанка)
        - source (имя PDF)
        - page (номер страницы из метаданных)
        - context и метаданные пока пустые (заполнятся в _enrich_chunks_with_llm_batch)
        """
        # Создаём временную директорию с одним PDF для LlamaIndex
        # (PDFReader работает с директориями)
        temp_dir = pdf_path.parent / f"_temp_{pdf_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        temp_pdf = temp_dir / pdf_path.name

        try:
            # Копируем PDF во временную директорию
            shutil.copy2(pdf_path, temp_pdf)

            # Читаем PDF через LlamaIndex
            pdf_reader = PDFReader()
            documents = pdf_reader.load_data(file=str(temp_pdf))

            # Парсим документы на ноды (чанки)
            # Используем разумные параметры для статистических документов
            node_parser = SentenceSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                paragraph_separator="\n\n",
            )

            chunks: List[Chunk] = []
            for doc in documents:
                nodes = node_parser.get_nodes_from_documents([doc])

                for node in nodes:
                    # Извлекаем номер страницы из метаданных
                    # LlamaIndex обычно сохраняет page_label или page_number
                    page_num = 0
                    if hasattr(node, "metadata"):
                        page_num = node.metadata.get("page_label", 0)
                        if not page_num:
                            page_num = node.metadata.get("page_number", 0)
                        if not page_num:
                            page_num = node.metadata.get("page", 0)
                        try:
                            page_num = int(page_num) if page_num else 0
                        except (ValueError, TypeError):
                            page_num = 0

                    # Создаём Chunk с пустыми полями для enrichment
                    chunk = Chunk(
                        id="",  # будет присвоен в build()
                        context="",  # заполнится в _enrich_chunks_with_llm_batch
                        text=node.text or "",
                        source=pdf_path.name,
                        page=page_num,
                        geo=None,
                        metrics=None,
                        years=None,
                        time_granularity=None,
                        oked=None,
                    )
                    chunks.append(chunk)

        finally:
            # Удаляем временную директорию
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        return chunks

    def _enrich_chunks_with_llm_batch(self, pdf_name: str, chunks: List[Chunk], all_chunks: List[Chunk] | None = None) -> List[Chunk]:
        """
        Батчевое LLM‑обогащение чанков с контекстом.

        Для каждого чанка передается:
        - 1 чанка до и 1 чанка после текущего
        - Текущий чанк для обогащения

        Обрабатывается батчами (один LLM-запрос на N чанков), чтобы
        резко сократить количество запросов к Ollama и получить скорость 1600–3200 чанков/час.
        Параллелизм применяется на уровне батчей (обычно 1–2 одновременных запроса).
        Пропускает LLM обработку для страниц 1-3 (только обложка и название).
        """
        if not chunks:
            return []
        
        
        # Разделяем чанки на те, что нужно обработать LLM и те, что можно пропустить
        chunks_to_process: List[Chunk] = []
        chunks_to_skip: List[Chunk] = []
        
        for chunk in chunks:
            if chunk.page <= 3:
                # Для первых трёх страниц просто используем text как context
                chunk.context = chunk.text[:200] if chunk.text else "нет текста"
                chunks_to_skip.append(chunk)
            else:
                chunks_to_process.append(chunk)
        
        if not chunks_to_process:
            return chunks
        
        # Настройки батчирования/параллелизма через env:
        # - RAG_ENRICH_BATCH_SIZE: сколько чанков в одном LLM-запросе (по умолчанию 5 для лучшего качества)
        # - RAG_ENRICH_CONCURRENCY: сколько батчей обрабатывать параллельно (по умолчанию 2 для ускорения)
        #
        # ВАЖНО: маленькие модели (llama3.2:3b) лучше работают с меньшими батчами (5 вместо 10)
        # Это повышает качество и снижает вероятность таймаутов
        batch_size = int(os.getenv("RAG_ENRICH_BATCH_SIZE", "5"))  # Уменьшено с 10 до 5 для лучшего качества
        batch_concurrency = int(os.getenv("RAG_ENRICH_CONCURRENCY", "1"))  # Увеличено с 1 до 2 для ускорения
        batch_size = max(1, batch_size)
        batch_concurrency = max(1, min(batch_concurrency, 8))  # Максимум 4 параллельных запроса

        total_batches = (len(chunks_to_process) + batch_size - 1) // batch_size
        print(f"   Обработка {len(chunks_to_process)} чанков (пропущено {len(chunks_to_skip)} чанков со страниц 1-3)")
        print(f"   Батчи: размер={batch_size}, батчей={total_batches}, параллельность={batch_concurrency}")

        all_enriched_chunks: List[Chunk] = chunks_to_skip.copy()

        # Параллелим только батчи (а не каждый чанк), чтобы не убить Ollama сотнями одновременных запросов
        start_time = time.time()
        
        # Счетчик для сброса контекста каждые 50 чанков
        chunks_since_reset = 0
        reset_interval = int(os.getenv("RAG_ENRICH_RESET_INTERVAL", "50"))  # Каждые 50 чанков по умолчанию
        
        # Сохраняем ссылку на исходный список всех чанков для получения контекста
        all_chunks_ref = all_chunks if all_chunks is not None else chunks
        
        with ThreadPoolExecutor(max_workers=min(batch_concurrency, total_batches)) as executor:
            futures = []
            for start in range(0, len(chunks_to_process), batch_size):
                batch = chunks_to_process[start : start + batch_size]
                # Получаем контекст предыдущих страниц для батча
                # Берем чанки до текущего батча из исходного списка all_chunks_ref
                # Это важно для получения контекста даже если предыдущие батчи еще не обработаны
                previous_chunks_for_batch = None
                if start > 0:
                    # Берем последние 2 чанка из исходного списка chunks (до текущего батча)
                    # Используем исходный список all_chunks_ref, чтобы получить контекст даже в параллельном режиме
                    prev_end = start
                    prev_start = max(0, prev_end - 2)
                    # Находим соответствующие чанки в исходном списке chunks
                    prev_chunk_indices = []
                    for i in range(prev_start, prev_end):
                        if i < len(chunks_to_process):
                            # Находим соответствующий чанк в исходном списке chunks
                            # chunks_to_process содержит чанки начиная с индекса len(chunks_to_skip)
                            chunk_idx = len(chunks_to_skip) + i
                            if chunk_idx < len(all_chunks_ref):
                                prev_chunk_indices.append(chunk_idx)
                    
                    if prev_chunk_indices:
                        previous_chunks_for_batch = [all_chunks_ref[idx] for idx in prev_chunk_indices]
                
                fut = executor.submit(
                    self._enrich_single_batch,
                    pdf_name,
                    batch,
                    previous_chunks_for_batch
                )
                # сохраняем, какой батч соответствует future + когда он был отправлен
                fut._rag_submit_ts = time.time()  # type: ignore[attr-defined]
                futures.append(fut)

            completed_batches = 0
            completed_chunks = 0
            batch_to_future = {}
            for i, fut in enumerate(futures):
                batch_start_idx = i * batch_size
                batch_end_idx = min(batch_start_idx + batch_size, len(chunks_to_process))
                batch_to_future[fut] = chunks_to_process[batch_start_idx:batch_end_idx]
            
            for fut in as_completed(futures):
                original_batch = batch_to_future[fut]
                try:
                    enriched_batch = fut.result()
                    submit_ts = getattr(fut, "_rag_submit_ts", None)
                    batch_time = (time.time() - submit_ts) if submit_ts else 0.0
                    all_enriched_chunks.extend(enriched_batch)
                    completed_batches += 1
                    completed_chunks += len(enriched_batch)
                    chunks_since_reset += len(enriched_batch)
                    
                    # Сброс контекста каждые N чанков для предотвращения деградации производительности
                    if chunks_since_reset >= reset_interval:
                        print(f"   🔄 Сброс контекста модели после {completed_chunks} чанков...")
                        self._llm.reset_context()
                        chunks_since_reset = 0
                        # Небольшая пауза после сброса
                        time.sleep(0.5)
                    
                    elapsed = time.time() - start_time
                    rate = completed_chunks / elapsed * 3600 if elapsed > 0 else 0
                    print(f"   Батч {completed_batches}/{total_batches}: {len(enriched_batch)} чанков за {batch_time:.1f}с | Всего: {completed_chunks}/{len(chunks_to_process)} | Скорость: {rate:.0f} чанков/час")
                    
                    # Логируем прогресс батча
                    self._logger.log_prepare_db(
                        "batch",
                        pdf_name=pdf_name,
                        batch_number=completed_batches,
                        total_batches=total_batches,
                        chunks_in_batch=len(enriched_batch),
                        elapsed_time=batch_time,
                        total_chunks_processed=completed_chunks,
                        rate_per_hour=rate
                    )
                except Exception as e:
                    submit_ts = getattr(fut, "_rag_submit_ts", None)
                    batch_time = (time.time() - submit_ts) if submit_ts else 0.0
                    completed_batches += 1
                    print(f"   ⚠️  Ошибка при обработке батча {completed_batches}/{total_batches} (время: {batch_time:.1f}с): {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: добавляем чанки без обогащения
                    for ch in original_batch:
                        if not ch.context:
                            ch.context = ch.text[:200] if ch.text else "нет текста"
                    all_enriched_chunks.extend(original_batch)
                    completed_chunks += len(original_batch)
                    chunks_since_reset += len(original_batch)
        
        # Сортируем по исходному порядку
        chunk_order = {ch.id: i for i, ch in enumerate(chunks)}
        all_enriched_chunks.sort(key=lambda ch: chunk_order.get(ch.id, 999999))
        
        return all_enriched_chunks
    
    
    def _enrich_single_with_context(self, pdf_name: str, chunk: Chunk, context_data: Dict[str, Any]) -> Chunk:
        """
        Обогащение одного чанка с контекстом.
        """
        system_prompt = (
            "Ты — аналитик по официальной статистике Республики Беларусь. "
            "Твоя задача — обогатить чанк документа структурированными метаданными. "
            "Верни ТОЛЬКО JSON-объект (не массив!) с полями: chunk_id, context, geo, metrics, years, time_granularity, oked."
        )
        
        prompt = (
            f"Документ: {pdf_name}\n\n"
            "КОНТЕКСТ ДО ТЕКУЩЕГО ЧАНКА (1 предыдущих чанка):\n"
            f"{json.dumps(context_data['before'], ensure_ascii=False, indent=2)}\n\n"
            "ТЕКУЩИЙ ЧАНК ДЛЯ ОБОГАЩЕНИЯ:\n"
            f"ID: {context_data['target']['chunk_id']}\n"
            f"Страница: {context_data['target']['page']}\n"
            f"Текст: {context_data['target']['text']}\n\n"
            "КОНТЕКСТ ПОСЛЕ ТЕКУЩЕГО ЧАНКА (1 следующих чанка):\n"
            f"{json.dumps(context_data['after'], ensure_ascii=False, indent=2)}\n\n"
            "Задача: опиши ТЕКУЩИЙ ЧАНК на основе его текста и контекста.\n"
            "Верни JSON-объект с полями:\n"
            "- chunk_id: точно такой же как ID выше\n"
            "- context: краткое, точное описание содержания чанка на русском языке (1-2 предложения), отражающее основную тему и данные\n"
            "- geo: географический объект (название региона, города, области) или null, если не указан\n"
            "- metrics: список названий метрик/показателей в нижнем регистре на русском языке (например: ['удой молока', 'инвестиции в основной капитал', 'доля домашних хозяйств, имеющих компьютер']) или null. "
            "Извлекай только реальные названия метрик из текста, не придумывай. Каждое название должно быть в нижнем регистре.\n"
            "- years: список годов (только целые числа, например [2023, 2024]) или null\n"
            "- time_granularity: 'year'/'quarter'/'month'/'day' или null\n"
            "- oked: код ОКЭД или null\n\n"
            "ВАЖНО: Верни ТОЛЬКО JSON-объект {}, НЕ массив!"
        )
        
        try:
            raw_response = self._llm.generate(prompt, system_prompt=system_prompt, format="json")
            enriched_data = self._parse_llm_single_enrichment(raw_response, chunk.id)
            
            if enriched_data:
                # Обновляем чанк данными от LLM
                if enriched_data.get("context"):
                    chunk.context = str(enriched_data["context"])[:200]
                elif chunk.text:
                    chunk.context = chunk.text[:200]
                else:
                    chunk.context = "нет текста"
                
                if "geo" in enriched_data:
                    chunk.geo = enriched_data["geo"]
                if "metrics" in enriched_data:
                    # Нормализуем метрики: приводим к нижнему регистру и фильтруем только русские строки
                    metrics = enriched_data["metrics"]
                    if metrics and isinstance(metrics, list):
                        normalized_metrics = []
                        for m in metrics:
                            if isinstance(m, str) and m.strip():
                                # Приводим к нижнему регистру
                                normalized = m.strip().lower()
                                # Проверяем, что это русский текст (содержит кириллицу)
                                if any('\u0400' <= char <= '\u04FF' for char in normalized):
                                    normalized_metrics.append(normalized)
                        chunk.metrics = normalized_metrics if normalized_metrics else None
                    else:
                        chunk.metrics = None
                if "years" in enriched_data:
                    chunk.years = self._normalize_years(enriched_data["years"])
                if "time_granularity" in enriched_data:
                    chunk.time_granularity = enriched_data["time_granularity"]
                if "oked" in enriched_data:
                    chunk.oked = enriched_data["oked"]
        except Exception as e:
            print(f"   ⚠️  Ошибка при обогащении чанка {chunk.id}: {e}")
            # Оставляем чанк без обогащения
            if not chunk.context:
                chunk.context = chunk.text[:200] if chunk.text else "нет текста"
        
        return chunk
    
    def _enrich_single_batch(
        self,
        pdf_name: str,
        chunks: List[Chunk],
        previous_chunks: List[Chunk] | None = None
    ) -> List[Chunk]:
        """
        Обогащение одного батча чанков с учетом контекста предыдущих страниц.
        
        Args:
            pdf_name: Имя PDF файла
            chunks: Список чанков для обогащения
            previous_chunks: Список чанков с предыдущих страниц для контекста
        """
        if not chunks:
            return []

        # Формируем промпт с описанием документа и всеми чанками
        chunks_data = []
        for i, ch in enumerate(chunks):
            chunks_data.append({
                "chunk_id": ch.id,
                # ВАЖНО: режем текст, чтобы ускорить генерацию и снизить нагрузку на контекст.
                # Для извлечения метрик/лет/гео обычно достаточно первых ~350 символов.
                "text": (ch.text or "")[:350],
                "page": ch.page,
            })
        
        # Подготавливаем контекст предыдущих страниц
        previous_context = None
        if previous_chunks:
            # Берем последние 1-2 чанка с предыдущих страниц для контекста
            prev_texts = []
            for prev_ch in previous_chunks[-2:]:  # Последние 2 чанка
                if prev_ch.text:
                    prev_texts.append(f"Страница {prev_ch.page}: {prev_ch.text[:200]}")
            if prev_texts:
                previous_context = "\n".join(prev_texts)

        # System prompt для строгого контроля формата (и для ускорения — жёсткие ограничения)
        system_prompt = (
            "Ты — аналитик по официальной статистике Республики Беларусь. "
            "Твоя задача — обогатить чанки документа структурированными метаданными. "
            "STRICT RULES:- Output ONLY a valid JSON array of exactly 5 objects - No text before or after JSON - No markdown, comments, explanations - chunk_id must exactly match input"
            "Ограничения: context <= 180 символов; metrics максимум 3 элемента; years максимум 4 элемента."
        )
        
        # Упрощенный промпт для одного или нескольких чанков
        if len(chunks_data) == 1:
            # Для одного чанка - более простой промпт с контекстом предыдущих страниц
            chunk = chunks_data[0]
            context_section = ""
            if previous_context:
                context_section = (
                    f"\nКОНТЕКСТ ПРЕДЫДУЩИХ СТРАНИЦ (для понимания общего контекста документа):\n"
                    f"{previous_context}\n\n"
                )
            
            prompt = (
                f"Документ: {pdf_name}\n\n"
                f"{context_section}"
                f"Чанк для обработки:\n"
                f"ID: {chunk['chunk_id']}\n"
                f"Страница: {chunk['page']}\n"
                f"Текст: {chunk['text'][:500]}...\n\n"
                "Верни JSON-массив с одним объектом. Формат:\n"
                "[{\"chunk_id\":\"...\",\"context\":\"...\",\"geo\":null,\"metrics\":null,\"years\":null,\"time_granularity\":null,\"oked\":null}]\n\n"
                "Поля:\n"
                "- chunk_id: точно такой же как ID выше\n"
                "- context: краткое, точное описание содержания чанка на русском языке (1-2 предложения), "
                "отражающее что находится в чанке с учётом предыдущих страниц\n"
                "- geo: название региона/города/области или null\n"
                "- metrics: список метрик в нижнем регистре на русском (максимум 3) или null\n"
                "- years: список годов (максимум 4) или null\n"
                "- time_granularity: 'year'/'quarter'/'month'/'day' или null\n"
                "- oked: код ОКЭД или null\n\n"
                "КРИТИЧЕСКИ ВАЖНО: Верни массив [{}], НЕ объект {}!"
            )
        else:
            # Для нескольких чанков — максимально четкий промпт с примером и контекстом
            context_section = ""
            if previous_context:
                context_section = (
                    f"\nКОНТЕКСТ ПРЕДЫДУЩИХ СТРАНИЦ (для понимания общего контекста документа):\n"
                    f"{previous_context}\n\n"
                )
            
            prompt = (
                f"Документ: {pdf_name}\n\n"
                f"{context_section}"
                f"Обработай {len(chunks_data)} чанков. Верни JSON-массив из РОВНО {len(chunks_data)} объектов.\n\n"
                "Формат ответа (пример для 2 чанков):\n"
                "[{\"chunk_id\":\"doc.pdf::page1::chunk0\",\"context\":\"Краткое описание\",\"geo\":null,\"metrics\":[\"метрика1\"],\"years\":[2024],\"time_granularity\":null,\"oked\":null},"
                "{\"chunk_id\":\"doc.pdf::page1::chunk1\",\"context\":\"Другое описание\",\"geo\":\"Минск\",\"metrics\":null,\"years\":null,\"time_granularity\":\"year\",\"oked\":null}]\n\n"
                "Правила для каждого объекта:\n"
                "- chunk_id: точно как в входных данных (обязательно!)\n"
                "- context: краткое, точное описание содержания чанка на русском языке (1-2 предложения), "
                "отражающее что находится в чанке с учётом предыдущих страниц, например, "
                "\"Производительность труда по ВВП, ВРП, ВВП, на одного занятого в экономике, рублей ВРП на одного занятого в экономике, рублей\"\n"
                "- metrics: только реальные названия из текста, нижний регистр, русский, максимум 3, или null\n"
                "- years: только целые числа, максимум 4, или null\n"
                "- geo: название региона/города/области или null\n"
                "- time_granularity: 'year'/'quarter'/'month'/'day' или null\n"
                "- oked: код ОКЭД или null\n\n"
                "Входные чанки:\n"
                f"{json.dumps(chunks_data, ensure_ascii=False, separators=(',',':'))}\n\n"
                "КРИТИЧЕСКИ ВАЖНО:\n"
                "1. Верни МАССИВ [{}, {}, ...], начинается с '[' и заканчивается ']'\n"
                "2. НЕ возвращай объект {}\n"
                f"3. В массиве должно быть РОВНО {len(chunks_data)} объектов\n"
                "4. Каждый объект должен содержать chunk_id из входных данных\n"
                "5. НЕ используй unicode escape-последовательности (\\u0412) - пиши русский текст напрямую"
            )

        # keep_alive: держит модель в памяти GPU между запросами для более равномерной нагрузки
        # "5m" = 5 минут (можно настроить через RAG_OLLAMA_KEEP_ALIVE)
        keep_alive = os.getenv("RAG_OLLAMA_KEEP_ALIVE", "5m")
        
        req_options = {
            "temperature": 0,
            "top_p": 1,
            # ограничиваем длину вывода (важно: иначе модель может “разливаться” на сотни секунд)
            "num_predict": min(250 * len(chunks_data) + 100, 3000),  # ~250 токенов на чанк, макс 3000
        }

        # Логируем, что отправляем в LLM (для отладки)
        ollama_config = {
            "model": getattr(getattr(self._llm, "config", None), "model", None),
            "base_url": getattr(getattr(self._llm, "config", None), "base_url", None),
            "timeout": getattr(getattr(self._llm, "config", None), "timeout", None),
            "format": "json",
            "options": req_options,
        }
        
        self._logger.log_llm_enrichment(
            event="request",
            pdf_name=pdf_name,
            chunks_count=len(chunks_data),
            chunk_ids=[c["chunk_id"] for c in chunks_data],
            pages=[c["page"] for c in chunks_data],
            system_prompt=system_prompt,
            prompt=prompt,
            ollama_config=ollama_config,
        )

        # Вызываем LLM с форсированием JSON формата
        # Ollama поддерживает параметр format для форсирования JSON
        # Делаем до 2 попыток при плохом качестве ответа
        max_retries_for_quality = 2
        enriched_data = []
        raw_response = ""
        
        for attempt in range(max_retries_for_quality):
            raw_response = self._llm.generate(
                prompt,
                system_prompt=system_prompt,
                format="json",
                # keep_alive передается на верхнем уровне payload в Ollama API
                keep_alive=keep_alive,
                # Параметры для ускорения/стабильности (Ollama options)
                options=req_options,
            )

            # Парсим JSON‑ответ
            enriched_data = self._parse_llm_batch_enrichment(raw_response)
            
            # Проверяем качество ответа
            valid_items = [item for item in enriched_data if isinstance(item, dict)]
            if len(valid_items) >= len(chunks_data) * 0.8:  # Хотя бы 80% чанков обогащено
                break  # Качество приемлемое, выходим
            
            # Если качество плохое и это не последняя попытка - повторяем
            if attempt < max_retries_for_quality - 1:
                print(f"   ⚠️  Низкое качество ответа ({len(valid_items)}/{len(chunks_data)}), повторная попытка {attempt + 2}/{max_retries_for_quality}...")
                time.sleep(1)  # Небольшая пауза перед повтором

        # Логируем ответ LLM (сырой) + результат парсинга
        valid_enriched_data_for_log = [item for item in enriched_data if isinstance(item, dict)]
        parsed_with_chunk_id = sum(
            1 for item in valid_enriched_data_for_log if item.get("chunk_id")
        )
        self._logger.log_llm_enrichment(
            event="response",
            pdf_name=pdf_name,
            chunks_count=len(chunks_data),
            chunk_ids=[c["chunk_id"] for c in chunks_data],
            raw_response=raw_response,
            parsed_items=len(valid_enriched_data_for_log),
            parsed_with_chunk_id=parsed_with_chunk_id,
        )
        
        # Отладочная информация
        if not enriched_data:
            print(f"⚠️  WARNING: LLM не вернул данные для обогащения чанков документа {pdf_name}")
            print(f"   Количество чанков: {len(chunks)}")
            print(f"   Длина ответа LLM: {len(raw_response)} символов")
            print(f"   Первые 1000 символов ответа LLM:\n{raw_response[:1000]}")
            if len(raw_response) > 1000:
                print(f"   Последние 500 символов ответа LLM:\n{raw_response[-500:]}")

        # Фильтруем только словари из enriched_data
        valid_enriched_data = [item for item in enriched_data if isinstance(item, dict)]
        
        # Создаём словарь chunk_id -> enriched_data для быстрого поиска
        enriched_map: Dict[str, Dict[str, Any]] = {}
        for item in valid_enriched_data:
            chunk_id = item.get("chunk_id")
            if chunk_id:
                enriched_map[str(chunk_id)] = item
        
        # Проверяем, сколько чанков было обогащено
        if len(enriched_map) < len(chunks):
            print(f"⚠️  WARNING: Только {len(enriched_map)} из {len(chunks)} чанков были обогащены для документа {pdf_name}")
            # Показываем примеры chunk_id для отладки
            if chunks:
                print(f"   Пример chunk_id из чанков: {chunks[0].id}")
            if enriched_map:
                example_id = list(enriched_map.keys())[0]
                print(f"   Пример chunk_id из ответа LLM: {example_id}")

        # Обогащаем чанки данными от LLM
        enriched_chunks: List[Chunk] = []
        
        # Если LLM вернул данные, но не все чанки найдены по ID, пытаемся сопоставить по порядку
        if len(enriched_map) < len(chunks) and valid_enriched_data:
            # Специальная обработка: если LLM вернул только 1 объект вместо массива
            # (частая проблема с маленькими моделями)
            if len(valid_enriched_data) == 1 and len(chunks) > 1:
                # LLM вернул только один объект - используем его для первого чанка
                # и создаем пустые объекты для остальных
                first_chunk = chunks[0]
                if first_chunk.id not in enriched_map:
                    enriched_map[first_chunk.id] = valid_enriched_data[0].copy()
                    enriched_map[first_chunk.id]["chunk_id"] = first_chunk.id
                # Для остальных чанков создаем пустые объекты (будут использованы fallback значения)
            # Пытаемся сопоставить по порядку (если количество совпадает)
            elif len(valid_enriched_data) == len(chunks):
                for i, ch in enumerate(chunks):
                    if ch.id not in enriched_map and i < len(valid_enriched_data):
                        enriched_map[ch.id] = valid_enriched_data[i]
                        # Убеждаемся, что chunk_id установлен правильно
                        enriched_map[ch.id]["chunk_id"] = ch.id
            # Если количество не совпадает, но есть данные - пытаемся сопоставить по порядку
            # для тех чанков, которые еще не обогащены
            elif len(valid_enriched_data) > 0:
                # Используем данные по порядку для необогащенных чанков
                used_indices = set()
                for i, ch in enumerate(chunks):
                    if ch.id not in enriched_map:
                        # Ищем первый неиспользованный элемент из valid_enriched_data
                        for j, data in enumerate(valid_enriched_data):
                            if j not in used_indices:
                                enriched_map[ch.id] = data.copy()
                                enriched_map[ch.id]["chunk_id"] = ch.id
                                used_indices.add(j)
                                break
        
        for ch in chunks:
            enriched = enriched_map.get(ch.id, {})
            
            # Если для одного чанка не нашли по ID, но есть один объект в enriched_data - используем его
            if not enriched and len(chunks) == 1 and valid_enriched_data:
                # Для одного чанка модель могла вернуть объект без chunk_id
                if len(valid_enriched_data) == 1:
                    enriched = valid_enriched_data[0]
                    # Добавляем chunk_id если его нет
                    if "chunk_id" not in enriched:
                        enriched["chunk_id"] = ch.id
                elif len(valid_enriched_data) > 0:
                    # Если несколько объектов, берем первый
                    enriched = valid_enriched_data[0]
                    if "chunk_id" not in enriched:
                        enriched["chunk_id"] = ch.id
            
            # Обновляем context (с fallback на text)
            if enriched.get("context"):
                context_str = str(enriched.get("context"))
                # Декодируем unicode escape-последовательности если они есть
                if "\\u" in context_str:
                    try:
                        import codecs
                        context_str = codecs.decode(context_str, 'unicode_escape')
                    except (UnicodeDecodeError, ValueError):
                        pass  # Если не получилось декодировать - оставляем как есть
                ch.context = context_str[:200]
            elif ch.text:
                ch.context = ch.text[:200]
            else:
                ch.context = "нет текста"
            
            # Обновляем метаданные: всегда обновляем, если ключ присутствует в enriched
            # Это позволяет явно установить None для полей, которые LLM вернул как null
            if "geo" in enriched:
                ch.geo = enriched["geo"]
            if "metrics" in enriched:
                # Нормализуем метрики: приводим к нижнему регистру и фильтруем только русские строки
                metrics = enriched["metrics"]
                if metrics and isinstance(metrics, list):
                    normalized_metrics = []
                    for m in metrics:
                        if isinstance(m, str) and m.strip():
                            # Приводим к нижнему регистру
                            normalized = m.strip().lower()
                            # Проверяем, что это русский текст (содержит кириллицу)
                            if any('\u0400' <= char <= '\u04FF' for char in normalized):
                                normalized_metrics.append(normalized)
                    ch.metrics = normalized_metrics if normalized_metrics else None
                else:
                    ch.metrics = None
            if "years" in enriched:
                ch.years = self._normalize_years(enriched["years"])
            if "time_granularity" in enriched:
                ch.time_granularity = enriched["time_granularity"]
            if "oked" in enriched:
                ch.oked = enriched["oked"]
            
            enriched_chunks.append(ch)

        return enriched_chunks


    @staticmethod
    def _parse_llm_single_enrichment(raw: str, expected_chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Парсит ответ LLM для одного чанка (объект, не массив).
        """
        if not raw:
            return None
        
        # Удаление markdown code blocks
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Попытка парсинга как объект
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # Нормализуем ключи (ID -> chunk_id, и т.д.)
                normalized = KnowledgeBaseBuilder._normalize_enrichment_object(data, expected_chunk_id)
                return normalized
        except json.JSONDecodeError:
            pass
        
        # Поиск объекта в тексте
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end+1])
                if isinstance(data, dict):
                    normalized = KnowledgeBaseBuilder._normalize_enrichment_object(data, expected_chunk_id)
                    return normalized
            except json.JSONDecodeError:
                pass
        
        return None
    
    @staticmethod
    def _normalize_enrichment_object(obj: Dict[str, Any], expected_chunk_id: str) -> Dict[str, Any]:
        """
        Нормализует объект обогащения: исправляет ключи, извлекает вложенные данные.
        """
        result = {}
        
        # Нормализация chunk_id (может быть ID, chunk_id, ИД и т.д.)
        for key in ["chunk_id", "ID", "ИД", "id", "chunkId"]:
            if key in obj:
                chunk_id_value = str(obj[key])
                # Исправляем формат если нужно (page153::chunk1351 -> полный формат)
                if "::" not in chunk_id_value and "::" in expected_chunk_id:
                    # Пытаемся извлечь номер из chunk_id и использовать expected_chunk_id
                    result["chunk_id"] = expected_chunk_id
                else:
                    result["chunk_id"] = chunk_id_value
                break
        if "chunk_id" not in result:
            result["chunk_id"] = expected_chunk_id
        
        # Извлечение остальных полей
        for field in ["context", "geo", "metrics", "years", "time_granularity", "oked"]:
            if field in obj:
                result[field] = obj[field]
            else:
                # Пробуем найти вложенные объекты
                found = False
                for key, value in obj.items():
                    if isinstance(value, dict):
                        if field in value:
                            result[field] = value[field]
                            found = True
                            break
                        # Рекурсивно ищем в глубоко вложенных объектах
                        elif any(isinstance(v, dict) for v in value.values() if isinstance(v, dict)):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, dict) and field in sub_value:
                                    result[field] = sub_value[field]
                                    found = True
                                    break
                            if found:
                                break
                if not found:
                    result[field] = None
        
        # Если объект содержит вложенные объекты с нужными полями, извлекаем первый
        # (например, когда модель возвращает {"Светлогорский": {"chunk_id": ..., "context": ...}})
        for key, value in obj.items():
            if isinstance(value, dict):
                # Проверяем, содержит ли вложенный объект нужные поля
                has_enrichment_fields = any(field in value for field in ["context", "geo", "metrics", "chunk_id"])
                if has_enrichment_fields:
                    # Используем вложенный объект
                    nested = KnowledgeBaseBuilder._normalize_enrichment_object(value, expected_chunk_id)
                    # Объединяем результаты (вложенные данные имеют приоритет)
                    for k, v in nested.items():
                        if v is not None or k not in result:
                            result[k] = v
                    break
        
        return result
    
    @staticmethod
    def _parse_llm_batch_enrichment(raw: str) -> List[Dict[str, Any]]:
        """
        Робастный парсер JSON‑ответа от LLM для батчевого enrichment.

        Ищет JSON‑массив в ответе и пытается его распарсить.
        Если найден объект вместо массива, пытается извлечь из него данные.
        Обрабатывает случаи, когда LLM возвращает несколько объектов подряд (конкатенированные).
        При ошибке возвращает пустой список.
        """
        if not raw:
            return []

        # Стратегия 1: Удаление markdown code blocks и очистка от мусорных символов
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]  # Удаляем ```json
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]  # Удаляем ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]  # Удаляем закрывающий ```
        cleaned = cleaned.strip()
        
        # Очистка от мусорных символов в начале (иногда LLM возвращает "{  "["...)
        # Удаляем пробелы и переносы строк перед первым значимым символом
        while cleaned and cleaned[0] in [' ', '\n', '\r', '\t']:
            cleaned = cleaned[1:]
        
        # Если ответ начинается с "{  "[" - это объект, содержащий строку с массивом
        # Пытаемся извлечь массив из строки
        if cleaned.startswith('{') and '"[{' in cleaned:
            # Ищем начало массива внутри строки
            array_start = cleaned.find('"[{')
            if array_start != -1:
                # Извлекаем строку с массивом
                array_str_start = array_start + 1  # Пропускаем кавычку
                # Ищем конец строки (закрывающая кавычка перед })
                array_str_end = cleaned.find('"', array_str_start + 1)
                if array_str_end != -1:
                    # Извлекаем и декодируем escape-последовательности
                    array_str = cleaned[array_str_start:array_str_end]
                    # Заменяем экранированные кавычки и другие escape-последовательности
                    try:
                        import codecs
                        array_str = codecs.decode(array_str, 'unicode_escape')
                        # Пытаемся распарсить как JSON
                        data = json.loads(array_str)
                        if isinstance(data, list):
                            return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data)
                    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                        pass

        # Стратегия 2: Попытка прямого парсинга всего ответа
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data)
            elif isinstance(data, dict):
                # LLM вернул объект вместо массива - пытаемся извлечь массив из него
                # Ищем ключи, которые могут содержать массив
                for key in ["chunks", "data", "results", "items", "array"]:
                    if key in data and isinstance(data[key], list):
                        return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data[key])
                # Если объект содержит chunk_id или нужные поля - это один элемент
                # Оборачиваем в массив (вызывающий код должен обработать случай, когда вернулся 1 объект вместо N)
                if "chunk_id" in data or any(key in data for key in ["context", "geo", "metrics", "years"]):
                    return KnowledgeBaseBuilder._validate_and_fix_enrichment_data([data])
        except json.JSONDecodeError:
            # Пытаемся декодировать unicode escape-последовательности перед парсингом
            try:
                import codecs
                decoded = codecs.decode(cleaned, 'unicode_escape')
                data = json.loads(decoded)
                if isinstance(data, list):
                    return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data)
                elif isinstance(data, dict):
                    if "chunk_id" in data or any(key in data for key in ["context", "geo", "metrics", "years"]):
                        return KnowledgeBaseBuilder._validate_and_fix_enrichment_data([data])
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                pass

        # Стратегия 3: Поиск JSON-массива в тексте
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, list):
                    return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data)
            except json.JSONDecodeError:
                pass

        # Стратегия 4: Поиск нескольких JSON объектов в тексте (конкатенированные объекты)
        # Это критично - LLM часто возвращает {"chunk_id":"...",...}{"chunk_id":"...",...} без массива
        objects = []
        i = 0
        while i < len(cleaned):
            if cleaned[i] == '{':
                # Находим закрывающую скобку для этого объекта
                depth = 0
                j = i
                while j < len(cleaned):
                    if cleaned[j] == '{':
                        depth += 1
                    elif cleaned[j] == '}':
                        depth -= 1
                        if depth == 0:
                            # Нашли полный объект
                            try:
                                obj_str = cleaned[i:j+1]
                                obj = json.loads(obj_str)
                                if isinstance(obj, dict):
                                    # Принимаем объект если есть chunk_id или хотя бы одно из нужных полей
                                    if "chunk_id" in obj or any(key in obj for key in ["context", "geo", "metrics", "years"]):
                                        objects.append(obj)
                            except json.JSONDecodeError:
                                # Пропускаем невалидный JSON
                                pass
                            i = j + 1  # Продолжаем поиск после этого объекта
                            break
                    j += 1
                else:
                    # Не нашли закрывающую скобку - пропускаем этот символ
                    i += 1
            else:
                i += 1
        
        if objects:
            return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(objects)

        # Стратегия 5: Попытка найти один JSON-объект и обернуть в массив
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    # Пытаемся найти массив внутри объекта
                    for key in ["chunks", "data", "results", "items", "array"]:
                        if key in data and isinstance(data[key], list):
                            return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data[key])
                    # Если объект содержит chunk_id, оборачиваем в массив
                    if "chunk_id" in data or any(key in data for key in ["context", "geo", "metrics", "years"]):
                        return KnowledgeBaseBuilder._validate_and_fix_enrichment_data([data])
            except json.JSONDecodeError:
                pass

        # Если ничего не сработало
        print(f"⚠️  WARNING: Не найден JSON-массив в ответе LLM. Первые 500 символов: {raw[:500]}")
        return []
    
    @staticmethod
    def _validate_and_fix_enrichment_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Валидирует и исправляет данные обогащения.
        """
        if not isinstance(data, list):
            print(f"⚠️  WARNING: Парсер вернул не список, а {type(data).__name__}")
            return []
        
        # Фильтруем только словари
        valid_items = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                # Пропускаем не-словари без предупреждения (могут быть строки из неправильного парсинга)
                continue
            valid_items.append(item)
        
        # Проверяем, что каждый элемент содержит необходимые поля
        for item in valid_items:
            # Убеждаемся, что все поля присутствуют (даже если null)
            required_fields = ["chunk_id", "context", "geo", "metrics", "years", "time_granularity", "oked"]
            for field in required_fields:
                if field not in item:
                    # Если поле отсутствует, добавляем его как None
                    item[field] = None
        
        return valid_items

    @staticmethod
    def _normalize_years(value: Any) -> Optional[List[int]]:
        """Нормализует значение years в список целых чисел."""
        if value is None:
            return None
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            try:
                return [int(value)]
            except ValueError:
                return None
        if isinstance(value, list):
            years: List[int] = []
            for v in value:
                try:
                    years.append(int(v))
                except (TypeError, ValueError):
                    continue
            return years or None
        return None

    # -------------------- FAISS -------------------- #

    def _build_faiss_index(self, chunks: List[Chunk], index_path: Path) -> None:
        """
        Строит FAISS IndexFlatIP по embeddings поля `context`.

        ВАЖНО:
        - HashVectorizer уже нормализует вектора, поэтому IndexFlatIP == cosine similarity.
        """
        if not chunks:
            # создаём пустой индекс на случай пустой базы (отладка)
            index = faiss.IndexFlatIP(self._vectorizer.dimension)
            faiss.write_index(index, str(index_path))
            return

        texts = [ch.context for ch in chunks]
        embeddings = self._vectorizer.embed_many(texts).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))

