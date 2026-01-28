from __future__ import annotations

"""
Модуль для LLM-обогащения чанков с использованием rolling context buffer.

Отвечает за:
- Батчевое обогащение чанков через Ollama
- Rolling context buffer (последние N чанков вместо всех)
- Парсинг JSON-ответов от LLM
- Валидацию и нормализацию данных
"""

from typing import List, Dict, Any, Optional, Deque
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
import codecs

from src.main.models import Chunk
from src.main.ollama_client import OllamaClient
from src.logs.logger import get_logger
from src.prepare_db.json_validator import ChunkValidator
from src.prepare_db.chunk_filter import ChunkFilter, ChunkType
from src.prepare_db.post_processor import EnrichmentPostProcessor


class RollingContextBuffer:
    """
    Rolling context buffer для хранения последних N чанков.
    
    Используется вместо хранения всех чанков в памяти для снижения потребления памяти
    и улучшения производительности при обработке больших документов.
    """

    def __init__(self, max_size: int = 10):
        """
        Инициализация буфера.
        
        Args:
            max_size: Максимальное количество чанков в буфере
        """
        self._buffer: Deque[Chunk] = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, chunk: Chunk) -> None:
        """Добавляет чанк в буфер (старые автоматически удаляются)."""
        self._buffer.append(chunk)

    def add_batch(self, chunks: List[Chunk]) -> None:
        """Добавляет батч чанков в буфер."""
        for chunk in chunks:
            self._buffer.append(chunk)

    def get_context(self, num_chunks: int = 2) -> List[Chunk]:
        """
        Получает последние N чанков для контекста.
        
        Args:
            num_chunks: Количество последних чанков для получения
            
        Returns:
            Список последних N чанков
        """
        return list(self._buffer)[-num_chunks:]

    def clear(self) -> None:
        """Очищает буфер."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Возвращает текущий размер буфера."""
        return len(self._buffer)


class LLMEnricher:
    """
    Класс для LLM-обогащения чанков с использованием rolling context buffer.
    
    Использует батчевую обработку и параллелизм на уровне батчей.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        batch_size: int = 5,
        batch_concurrency: int = 1,
        context_buffer_size: int = 10,
        reset_interval: int = 50,
    ):
        """
        Инициализация обогатителя.
        
        Args:
            llm_client: Клиент Ollama для запросов к LLM
            batch_size: Размер батча (сколько чанков в одном LLM-запросе)
            batch_concurrency: Количество параллельных батчей
            context_buffer_size: Размер rolling context buffer
            reset_interval: Интервал сброса контекста модели (в чанках)
        """
        self._llm = llm_client
        self._batch_size = batch_size
        self._batch_concurrency = batch_concurrency
        self._reset_interval = reset_interval
        self._logger = get_logger()
        self._validator = ChunkValidator()
        self._chunk_filter = ChunkFilter(skip_first_pages=3)
        self._post_processor = EnrichmentPostProcessor()
        
        # Rolling context buffer
        self._context_buffer = RollingContextBuffer(max_size=context_buffer_size)
        
        # Счетчик для сброса контекста
        self._chunks_since_reset = 0

    def enrich_chunks(
        self,
        pdf_name: str,
        chunks: List[Chunk],
        skip_first_pages: int = 3,
    ) -> List[Chunk]:
        """
        Обогащает чанки через LLM с использованием rolling context buffer.
        
        Args:
            pdf_name: Имя PDF файла
            chunks: Список чанков для обогащения
            skip_first_pages: Количество первых страниц для пропуска (только обложка)
            
        Returns:
            Список обогащенных чанков
        """
        if not chunks:
            return []

        # Используем ChunkFilter для классификации чанков
        data_chunks, metadata_chunks, skip_chunks = self._chunk_filter.filter_chunks(chunks)
        
        # Для metadata чанков используем упрощённое обогащение
        for chunk in metadata_chunks:
            chunk.context = chunk.text[:200] if chunk.text else "нет текста"
        
        # Для skip чанков просто пропускаем
        for chunk in skip_chunks:
            chunk.context = chunk.text[:200] if chunk.text else "нет текста"
        
        chunks_to_process = data_chunks
        chunks_to_skip = metadata_chunks + skip_chunks

        if not chunks_to_process:
            return chunks

        # Настройки батчирования через env (с fallback на параметры конструктора)
        batch_size = int(os.getenv("RAG_ENRICH_BATCH_SIZE", str(self._batch_size)))
        batch_concurrency = int(os.getenv("RAG_ENRICH_CONCURRENCY", str(self._batch_concurrency)))
        batch_size = max(1, batch_size)
        batch_concurrency = max(1, min(batch_concurrency, 8))

        print(
            f"   Обработка {len(chunks_to_process)} чанков "
            f"(пропущено {len(chunks_to_skip)} служебных чанков)"
        )
        print(
            f"   Режим: 1 чанк = 1 запрос, параллельность={batch_concurrency * batch_size}"
        )

        all_enriched_chunks: List[Chunk] = chunks_to_skip.copy()

        # Параллелим только батчи (а не каждый чанк)
        start_time = time.time()

        # Параллельная обработка: 1 чанк = 1 запрос
        # Батчинг остаётся на уровне параллелизма (несколько запросов одновременно)
        with ThreadPoolExecutor(max_workers=min(batch_concurrency * batch_size, len(chunks_to_process))) as executor:
            futures = []
            for chunk in chunks_to_process:
                # Получаем контекст из rolling buffer (последние 2 чанка)
                previous_chunks = self._context_buffer.get_context(num_chunks=2)
                
                fut = executor.submit(
                    self._enrich_single_chunk,
                    pdf_name,
                    chunk,
                    previous_chunks,
                )
                fut._rag_submit_ts = time.time()  # type: ignore[attr-defined]
                futures.append((fut, chunk))

            completed_chunks = 0
            total_chunks = len(chunks_to_process)

            for fut, original_chunk in futures:
                try:
                    enriched_chunk = fut.result()
                    submit_ts = getattr(fut, "_rag_submit_ts", None)
                    chunk_time = (time.time() - submit_ts) if submit_ts else 0.0
                    
                    if enriched_chunk:
                        all_enriched_chunks.append(enriched_chunk)
                        # Добавляем обогащенный чанк в rolling buffer
                        self._context_buffer.add(enriched_chunk)
                    else:
                        # Fallback: добавляем чанк без обогащения
                        if not original_chunk.context:
                            original_chunk.context = original_chunk.text[:200] if original_chunk.text else "нет текста"
                        all_enriched_chunks.append(original_chunk)
                        self._context_buffer.add(original_chunk)
                    
                    completed_chunks += 1
                    self._chunks_since_reset += 1

                    # Сброс контекста каждые N чанков
                    if self._chunks_since_reset >= self._reset_interval:
                        print(
                            f"   🔄 Сброс контекста модели после {completed_chunks} чанков..."
                        )
                        self._llm.reset_context()
                        self._chunks_since_reset = 0
                        time.sleep(0.5)

                    # Периодический вывод прогресса (каждые 10 чанков или в конце)
                    if completed_chunks % 10 == 0 or completed_chunks == total_chunks:
                        elapsed = time.time() - start_time
                        rate = completed_chunks / elapsed * 3600 if elapsed > 0 else 0
                        print(
                            f"   Прогресс: {completed_chunks}/{total_chunks} чанков | "
                            f"Скорость: {rate:.0f} чанков/час"
                        )

                except Exception as e:
                    submit_ts = getattr(fut, "_rag_submit_ts", None)
                    chunk_time = (time.time() - submit_ts) if submit_ts else 0.0
                    print(
                        f"   ⚠️  Ошибка при обработке чанка {original_chunk.id} "
                        f"(время: {chunk_time:.1f}с): {e}"
                    )
                    import traceback
                    traceback.print_exc()
                    # Fallback: добавляем чанк без обогащения
                    if not original_chunk.context:
                        original_chunk.context = original_chunk.text[:200] if original_chunk.text else "нет текста"
                    all_enriched_chunks.append(original_chunk)
                    completed_chunks += 1
                    self._chunks_since_reset += 1

        # Сортируем по исходному порядку
        chunk_order = {ch.id: i for i, ch in enumerate(chunks)}
        all_enriched_chunks.sort(key=lambda ch: chunk_order.get(ch.id, 999999))

        return all_enriched_chunks

    def _enrich_single_chunk(
        self,
        pdf_name: str,
        chunk: Chunk,
        previous_chunks: List[Chunk] | None = None,
    ) -> Optional[Chunk]:
        """
        Обогащает один чанк с учетом контекста предыдущих страниц.
        
        Args:
            pdf_name: Имя PDF файла
            chunk: Чанк для обогащения
            previous_chunks: Список чанков с предыдущих страниц для контекста
            
        Returns:
            Обогащенный чанк или None при ошибке
        """
        # Подготавливаем контекст предыдущих страниц
        previous_context = None
        if previous_chunks:
            prev_texts = []
            for prev_ch in previous_chunks[-2:]:  # Последние 2 чанка
                if prev_ch.text:
                    prev_texts.append(f"Страница {prev_ch.page}: {prev_ch.text[:200]}")
            if prev_texts:
                previous_context = "\n".join(prev_texts)

        # Упрощенный system prompt для одного объекта
        system_prompt = self._build_system_prompt_single()

        # Формируем промпт для одного чанка
        prompt = self._build_prompt_single(pdf_name, chunk, previous_context)

        # Параметры запроса
        keep_alive = os.getenv("RAG_OLLAMA_KEEP_ALIVE", "5m")
        req_options = {
            "temperature": 0,
            "top_p": 1,
            "num_predict": 300,  # Один объект - меньше токенов
        }

        # Логируем запрос
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
            chunks_count=1,
            chunk_ids=[chunk.id],
            pages=[chunk.page],
            system_prompt=system_prompt,
            prompt=prompt,
            ollama_config=ollama_config,
        )

        # Вызываем LLM с упрощенным retry (один объект всегда валиден)
        max_retries = 2
        enriched_data = None
        raw_response = ""

        for attempt in range(max_retries):
            raw_response = self._llm.generate(
                prompt,
                system_prompt=system_prompt,
                format="json",
                keep_alive=keep_alive,
                options=req_options,
            )

            # Парсим JSON-ответ (ожидаем один объект)
            enriched_data = self._parse_llm_single_enrichment(raw_response, chunk.id)
            
            if enriched_data:
                # ВСЕГДА перезаписываем chunk_id оригинальным значением
                # LLM не является источником истины для ID - это критично для целостности данных
                enriched_data["chunk_id"] = chunk.id
                # Если парсинг успешен - выходим из цикла
                break
            elif attempt < max_retries - 1:
                print(
                    f"   ⚠️  Не удалось распарсить ответ для {chunk.id} "
                    f"(попытка {attempt + 1}/{max_retries})"
                )
                time.sleep(0.5)

        # Логируем ответ
        parsed_successfully = enriched_data is not None
        self._logger.log_llm_enrichment(
            event="response",
            pdf_name=pdf_name,
            chunks_count=1,
            chunk_ids=[chunk.id],
            raw_response=raw_response,
            parsed_items=1 if parsed_successfully else 0,
            parsed_with_chunk_id=1 if (parsed_successfully and enriched_data and enriched_data.get("chunk_id")) else 0,
        )

        if not enriched_data:
            return None

        # Применяем данные обогащения к чанку
        enriched_chunk = self._apply_enrichment_data_single(chunk, enriched_data)
        
        # Post-processing: исправляем типичные ошибки LLM
        enriched_chunk = self._post_processor.process_chunk(enriched_chunk)

        return enriched_chunk

    def _build_system_prompt_single(self) -> str:
        """
        Строит упрощенный system prompt для одного объекта.
        
        Returns:
            System prompt для одного чанка
        """
        return (
            "Ты — аналитик по официальной статистике Республики Беларусь. "
            "Твоя задача — обогатить чанк документа структурированными метаданными.\n\n"
            "КРИТИЧЕСКИ ВАЖНО:\n"
            "1. Верни ТОЛЬКО валидный JSON-объект {}, начинающийся с '{' и заканчивающийся '}'\n"
            "2. НЕ возвращай массив [] - только объект {}\n"
            "3. chunk_id должен точно совпадать с входными данными (побайтно, без изменений)\n"
            "4. НЕ добавляй текст до или после JSON\n"
            "5. НЕ используй markdown code blocks (```json)\n"
            "6. НЕ добавляй префиксы типа 'ID:' к chunk_id - используй значение как есть\n\n"
            "JSON Schema для объекта:\n"
            "{\n"
            '  "chunk_id": "string (точно как во входных данных, без изменений)",\n'
            '  "context": "string (макс 256 символов, русский)",\n'
            '  "geo": "string | null",\n'
            '  "metrics": ["string"] | null (макс 5, только русские метрики, нижний регистр),\n'
            '  "years": [int] | null (макс 5, только годы из текста),\n'
            '  "time_granularity": "year" | "quarter" | "month" | "day" | null,\n'
            '  "oked": "string | null"\n'
            "}\n\n"
            "Правила для metrics:\n"
            "- Только реальные метрики из текста (например: 'удой молока', 'инвестиции')\n"
            "- НЕ определения, НЕ объекты наблюдения, НЕ заголовки\n"
            "- Только русский язык, нижний регистр\n"
            "- Максимум 5 элементов\n\n"
            "Правила для years:\n"
            "- Только годы, которые явно упоминаются в тексте\n"
            "- НЕ добавляй годы, если их нет в тексте (используй null)\n"
            "- НЕ добавляй годы для методологии, определений, содержания\n"
            "- Максимум 5 элементов\n"
        )

    def _build_system_prompt(
        self,
        chunks_count: int,
        format_error: bool = False,
        expected_count: Optional[int] = None,
    ) -> str:
        """
        Строит улучшенный system prompt с JSON Schema.
        
        Args:
            chunks_count: Количество чанков в батче
            format_error: True если была ошибка формата в предыдущей попытке
            expected_count: Ожидаемое количество элементов (для исправления ошибок)
        """
        base_prompt = (
            "Ты — аналитик по официальной статистике Республики Беларусь. "
            "Твоя задача — обогатить чанки документа структурированными метаданными.\n\n"
            "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
            "1. Верни ТОЛЬКО валидный JSON-массив, начинающийся с '[' и заканчивающийся ']'\n"
            "2. НЕ возвращай объект {} - только массив []\n"
            f"3. В массиве должно быть РОВНО {chunks_count} объектов\n"
            "4. chunk_id должен точно совпадать с входными данными (побайтно)\n"
            "5. НЕ добавляй текст до или после JSON\n"
            "6. НЕ используй markdown code blocks (```json)\n\n"
        )
        
        if format_error:
            base_prompt += (
                "⚠️ ОШИБКА ФОРМАТА В ПРЕДЫДУЩЕЙ ПОПЫТКЕ!\n"
                "Ты вернул объект {} вместо массива [] или неправильное количество элементов.\n"
                f"ОБЯЗАТЕЛЬНО верни массив из {expected_count or chunks_count} объектов.\n\n"
            )
        
        base_prompt += (
            "JSON Schema для каждого объекта:\n"
            "{\n"
            '  "chunk_id": "string (точно как во входных данных)",\n'
            '  "context": "string (макс 256 символов, русский)",\n'
            '  "geo": "string | null",\n'
            '  "metrics": ["string"] | null (макс 5, только русские метрики),\n'
            '  "years": [int] | null (макс 5, только годы из текста),\n'
            '  "time_granularity": "year" | "quarter" | "month" | "day" | null,\n'
            '  "oked": "string | null"\n'
            "}\n\n"
            "Правила для metrics:\n"
            "- Только реальные метрики из текста (например: 'удой молока', 'инвестиции')\n"
            "- НЕ определения, НЕ объекты наблюдения, НЕ заголовки\n"
            "- Только русский язык, нижний регистр\n"
            "- Максимум 5 элементов\n\n"
            "Правила для years:\n"
            "- Только годы, которые явно упоминаются в тексте\n"
            "- НЕ добавляй годы, если их нет в тексте (используй null)\n"
            "- НЕ добавляй годы для методологии, определений, содержания\n"
            "- Максимум 5 элементов\n"
        )
        
        return base_prompt

    def _check_chunk_ids_match(
        self,
        enriched_items: List[Dict[str, Any]],
        expected_chunk_ids: List[str],
    ) -> bool:
        """
        Проверяет точное совпадение chunk_id (с нормализацией).
        
        Args:
            enriched_items: Обогащенные данные от LLM
            expected_chunk_ids: Ожидаемые chunk_id
            
        Returns:
            True если все chunk_id совпадают
        """
        if len(enriched_items) != len(expected_chunk_ids):
            return False
        
        for i, (item, expected_id) in enumerate(zip(enriched_items, expected_chunk_ids)):
            actual_id = item.get("chunk_id", "")
            # Нормализуем для сравнения
            normalized_actual = ChunkFilter.normalize_chunk_id(actual_id)
            normalized_expected = ChunkFilter.normalize_chunk_id(expected_id)
            
            if normalized_actual != normalized_expected:
                return False
        
        return True

    def _fix_format_error(
        self,
        enriched_data: Any,
        expected_chunk_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Пытается исправить ошибку формата (объект вместо массива).
        
        Args:
            enriched_data: Данные от LLM (возможно, объект вместо массива)
            expected_chunk_ids: Ожидаемые chunk_id
            
        Returns:
            Исправленный список словарей
        """
        # Если это объект, пытаемся извлечь массив из него
        if isinstance(enriched_data, dict):
            # Ищем массив в ключах
            for key in ["chunks", "data", "results", "items", "array"]:
                if key in enriched_data and isinstance(enriched_data[key], list):
                    return enriched_data[key]
            
            # Если объект содержит chunk_id, оборачиваем в массив
            if "chunk_id" in enriched_data or any(
                key in enriched_data for key in ["context", "geo", "metrics", "years"]
            ):
                # Исправляем chunk_id если нужно
                if "chunk_id" not in enriched_data and expected_chunk_ids:
                    enriched_data["chunk_id"] = expected_chunk_ids[0]
                return [enriched_data]
        
        # Если это не список, возвращаем пустой список
        if not isinstance(enriched_data, list):
            return []
        
        return enriched_data

    def _build_prompt_single(
        self,
        pdf_name: str,
        chunk: Chunk,
        previous_context: Optional[str],
    ) -> str:
        """
        Строит промпт для одного чанка.
        
        Args:
            pdf_name: Имя PDF файла
            chunk: Чанк для обогащения
            previous_context: Контекст предыдущих страниц
            
        Returns:
            Промпт для LLM
        """
        context_section = ""
        if previous_context:
            context_section = (
                f"\nКОНТЕКСТ ПРЕДЫДУЩИХ СТРАНИЦ "
                f"(для понимания общего контекста документа):\n"
                f"{previous_context}\n\n"
            )

        return (
            f"Документ: {pdf_name}\n\n"
            f"{context_section}"
            f"Чанк для обработки:\n"
            f"chunk_id: {chunk.id}\n"
            f"Страница: {chunk.page}\n"
            f"Текст: {(chunk.text or '')[:500]}\n\n"
            "Верни JSON-объект (НЕ массив!). Формат:\n"
            '{"chunk_id":"...","context":"...","geo":null,"metrics":null,'
            '"years":null,"time_granularity":null,"oked":null}\n\n'
            "Поля:\n"
            f"- chunk_id: точно такой же как выше: '{chunk.id}' (без изменений, без префиксов)\n"
            "- context: краткое, точное описание содержания чанка на русском языке "
            "(1-2 предложения), отражающее что находится в чанке с учётом предыдущих страниц\n"
            "- geo: название региона/города/области или null\n"
            "- metrics: список метрик в нижнем регистре на русском (максимум 5) или null. "
            "Только реальные метрики из текста, не определения\n"
            "- years: список годов (максимум 9) или null. Только годы, явно упомянутые в тексте\n"
            "- time_granularity: 'year'/'quarter'/'month'/'day' или null\n"
            "- oked: код ОКЭД или null\n\n"
            "КРИТИЧЕСКИ ВАЖНО:\n"
            "1. Верни ОБЪЕКТ {}, НЕ массив []\n"
            "2. chunk_id должен быть точно таким же как выше (без 'ID:', без изменений)\n"
            "3. НЕ используй unicode escape-последовательности (\\u0412) - пиши русский текст напрямую"
        )

    def _build_prompt(
        self,
        pdf_name: str,
        chunks_data: List[Dict[str, Any]],
        previous_context: Optional[str],
    ) -> str:
        """Строит промпт для LLM (legacy метод для обратной совместимости)."""
        # Этот метод больше не используется, но оставлен для совместимости
        if len(chunks_data) == 1:
            chunk_data = chunks_data[0]
            # Создаем временный чанк для использования нового метода
            temp_chunk = Chunk(
                id=chunk_data["chunk_id"],
                context="",
                text=chunk_data["text"],
                source=pdf_name,
                page=chunk_data["page"],
            )
            return self._build_prompt_single(pdf_name, temp_chunk, previous_context)
        else:
            # Fallback для множественных чанков (не должно использоваться)
            return f"Обработай {len(chunks_data)} чанков..."

    def _apply_enrichment_data(
        self,
        chunks: List[Chunk],
        enriched_data: List[Dict[str, Any]],
    ) -> List[Chunk]:
        """Применяет данные обогащения к чанкам."""
        # Фильтруем только словари
        valid_enriched_data = [
            item for item in enriched_data if isinstance(item, dict)
        ]

        # Создаём словарь chunk_id -> enriched_data (с нормализацией)
        enriched_map: Dict[str, Dict[str, Any]] = {}
        for item in valid_enriched_data:
            chunk_id = item.get("chunk_id")
            if chunk_id:
                # Нормализуем chunk_id для сравнения
                normalized_id = ChunkFilter.normalize_chunk_id(str(chunk_id))
                enriched_map[normalized_id] = item
                # Сохраняем оригинальный chunk_id в данных
                item["chunk_id"] = normalized_id

        # Если не все чанки найдены по ID, пытаемся сопоставить по порядку
        if len(enriched_map) < len(chunks) and valid_enriched_data:
            if len(valid_enriched_data) == len(chunks):
                for i, ch in enumerate(chunks):
                    if ch.id not in enriched_map and i < len(valid_enriched_data):
                        enriched_map[ch.id] = valid_enriched_data[i]
                        enriched_map[ch.id]["chunk_id"] = ch.id
            elif len(valid_enriched_data) == 1 and len(chunks) > 1:
                first_chunk = chunks[0]
                if first_chunk.id not in enriched_map:
                    enriched_map[first_chunk.id] = valid_enriched_data[0].copy()
                    enriched_map[first_chunk.id]["chunk_id"] = first_chunk.id

        # Применяем данные к чанкам (с нормализацией chunk_id)
        enriched_chunks: List[Chunk] = []
        for ch in chunks:
            # Нормализуем chunk_id чанка для поиска
            normalized_chunk_id = ChunkFilter.normalize_chunk_id(ch.id)
            enriched = enriched_map.get(normalized_chunk_id, {})

            # Если для одного чанка не нашли по ID, но есть один объект - используем его
            if not enriched and len(chunks) == 1 and valid_enriched_data:
                enriched = valid_enriched_data[0]
                if "chunk_id" not in enriched:
                    enriched["chunk_id"] = ch.id

            # Валидируем и нормализуем данные
            validation_result = self._validator.validate_chunk(enriched, check_uniqueness=False)
            
            # Проверка качества metrics
            if enriched.get("metrics"):
                metrics_warnings = self._validator.validate_metrics_quality(
                    enriched["metrics"],
                    chunk_text=ch.text,
                )
                if metrics_warnings:
                    # Логируем предупреждения о качестве metrics
                    for warning in metrics_warnings:
                        print(f"   ⚠️  {warning}")
            
            if not validation_result.is_valid:
                # Применяем нормализацию для исправления ошибок
                if enriched.get("context"):
                    enriched["context"] = self._validator.normalize_context(
                        str(enriched["context"])
                    )
                if enriched.get("metrics"):
                    enriched["metrics"] = self._validator.normalize_metrics(
                        enriched["metrics"]
                    )
                if enriched.get("years"):
                    enriched["years"] = self._validator.normalize_years(enriched["years"])

            # Обновляем context
            if enriched.get("context"):
                context_str = str(enriched.get("context"))
                # Декодируем unicode escape-последовательности если они есть
                if "\\u" in context_str:
                    try:
                        context_str = codecs.decode(context_str, 'unicode_escape')
                    except (UnicodeDecodeError, ValueError):
                        pass
                ch.context = context_str[:200]
            elif ch.text:
                ch.context = ch.text[:200]
            else:
                ch.context = "нет текста"

            # Обновляем метаданные
            if "geo" in enriched:
                ch.geo = enriched["geo"]
            if "metrics" in enriched:
                ch.metrics = self._validator.normalize_metrics(enriched["metrics"])
            if "years" in enriched:
                ch.years = self._validator.normalize_years(enriched["years"])
            if "time_granularity" in enriched:
                ch.time_granularity = enriched["time_granularity"]
            if "oked" in enriched:
                ch.oked = enriched["oked"]

            enriched_chunks.append(ch)

        return enriched_chunks

    def _apply_enrichment_data_single(
        self,
        chunk: Chunk,
        enriched_data: Dict[str, Any],
    ) -> Chunk:
        """
        Применяет данные обогащения к одному чанку.
        
        Args:
            chunk: Чанк для обогащения
            enriched_data: Данные от LLM
            
        Returns:
            Обогащенный чанк
        """
        # КРИТИЧНО: всегда перезаписываем chunk_id оригинальным значением
        # LLM не является источником истины для ID - это гарантирует целостность данных
        enriched_data["chunk_id"] = chunk.id
        
        # Валидируем и нормализуем данные
        validation_result = self._validator.validate_chunk(enriched_data, check_uniqueness=False)
        
        # Проверка качества metrics
        if enriched_data.get("metrics"):
            metrics_warnings = self._validator.validate_metrics_quality(
                enriched_data["metrics"],
                chunk_text=chunk.text,
            )
            if metrics_warnings:
                for warning in metrics_warnings:
                    print(f"   ⚠️  {warning}")
        
        if not validation_result.is_valid:
            # Применяем нормализацию для исправления ошибок
            if enriched_data.get("context"):
                enriched_data["context"] = self._validator.normalize_context(
                    str(enriched_data["context"])
                )
            if enriched_data.get("metrics"):
                enriched_data["metrics"] = self._validator.normalize_metrics(
                    enriched_data["metrics"]
                )
            if enriched_data.get("years"):
                enriched_data["years"] = self._validator.normalize_years(enriched_data["years"])

        # Обновляем context
        if enriched_data.get("context"):
            context_str = str(enriched_data["context"])
            # Декодируем unicode escape-последовательности если они есть
            if "\\u" in context_str:
                try:
                    context_str = codecs.decode(context_str, 'unicode_escape')
                except (UnicodeDecodeError, ValueError):
                    pass
            chunk.context = context_str[:200]
        elif chunk.text:
            chunk.context = chunk.text[:200]
        else:
            chunk.context = "нет текста"

        # Обновляем метаданные
        if "geo" in enriched_data:
            chunk.geo = enriched_data["geo"]
        if "metrics" in enriched_data:
            chunk.metrics = self._validator.normalize_metrics(enriched_data["metrics"])
        if "years" in enriched_data:
            chunk.years = self._validator.normalize_years(enriched_data["years"])
        if "time_granularity" in enriched_data:
            chunk.time_granularity = enriched_data["time_granularity"]
        if "oked" in enriched_data:
            chunk.oked = enriched_data["oked"]

        return chunk

    @staticmethod
    def _parse_llm_single_enrichment(raw: str, expected_chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Парсит JSON-ответ от LLM для одного чанка (ожидает объект).
        
        Args:
            raw: Сырой ответ от LLM
            expected_chunk_id: Ожидаемый chunk_id
            
        Returns:
            Словарь с данными обогащения или None при ошибке
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

        # Очистка от мусорных символов
        while cleaned and cleaned[0] in [' ', '\n', '\r', '\t']:
            cleaned = cleaned[1:]

        # Прямой парсинг объекта
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # Проверяем, что это объект с нужными полями
                if "chunk_id" in data or any(
                    key in data for key in ["context", "geo", "metrics", "years"]
                ):
                    # Нормализуем chunk_id если нужно
                    if "chunk_id" not in data:
                        data["chunk_id"] = expected_chunk_id
                    return data
        except json.JSONDecodeError:
            # Пытаемся найти объект в тексте
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = cleaned[start : end + 1]
                try:
                    data = json.loads(snippet)
                    if isinstance(data, dict):
                        if "chunk_id" not in data:
                            data["chunk_id"] = expected_chunk_id
                        return data
                except json.JSONDecodeError:
                    pass

        return None

    @staticmethod
    def _parse_llm_batch_enrichment(raw: str) -> List[Dict[str, Any]]:
        """
        Робастный парсер JSON-ответа от LLM для батчевого enrichment.
        
        Использует множественные стратегии парсинга для обработки различных форматов ответов.
        """
        if not raw:
            return []

        # Стратегия 1: Удаление markdown code blocks
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Очистка от мусорных символов
        while cleaned and cleaned[0] in [' ', '\n', '\r', '\t']:
            cleaned = cleaned[1:]

        # Стратегия 2: Прямой парсинг
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return LLMEnricher._validate_and_fix_enrichment_data(data)
            elif isinstance(data, dict):
                for key in ["chunks", "data", "results", "items", "array"]:
                    if key in data and isinstance(data[key], list):
                        return LLMEnricher._validate_and_fix_enrichment_data(data[key])
                if "chunk_id" in data or any(
                    key in data for key in ["context", "geo", "metrics", "years"]
                ):
                    return LLMEnricher._validate_and_fix_enrichment_data([data])
        except json.JSONDecodeError:
            try:
                decoded = codecs.decode(cleaned, 'unicode_escape')
                data = json.loads(decoded)
                if isinstance(data, list):
                    return LLMEnricher._validate_and_fix_enrichment_data(data)
                elif isinstance(data, dict):
                    if "chunk_id" in data or any(
                        key in data for key in ["context", "geo", "metrics", "years"]
                    ):
                        return LLMEnricher._validate_and_fix_enrichment_data([data])
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
                    return LLMEnricher._validate_and_fix_enrichment_data(data)
            except json.JSONDecodeError:
                pass

        # Стратегия 4: Поиск нескольких JSON объектов (конкатенированные)
        objects = []
        i = 0
        while i < len(cleaned):
            if cleaned[i] == '{':
                depth = 0
                j = i
                while j < len(cleaned):
                    if cleaned[j] == '{':
                        depth += 1
                    elif cleaned[j] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                obj_str = cleaned[i:j+1]
                                obj = json.loads(obj_str)
                                if isinstance(obj, dict):
                                    if "chunk_id" in obj or any(
                                        key in obj
                                        for key in ["context", "geo", "metrics", "years"]
                                    ):
                                        objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
                    j += 1
                else:
                    i += 1
            else:
                i += 1

        if objects:
            return LLMEnricher._validate_and_fix_enrichment_data(objects)

        # Стратегия 5: Поиск одного JSON-объекта
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    for key in ["chunks", "data", "results", "items", "array"]:
                        if key in data and isinstance(data[key], list):
                            return LLMEnricher._validate_and_fix_enrichment_data(
                                data[key]
                            )
                    if "chunk_id" in data or any(
                        key in data for key in ["context", "geo", "metrics", "years"]
                    ):
                        return LLMEnricher._validate_and_fix_enrichment_data([data])
            except json.JSONDecodeError:
                pass

        print(
            f"⚠️  WARNING: Не найден JSON-массив в ответе LLM. "
            f"Первые 500 символов: {raw[:500]}"
        )
        return []

    @staticmethod
    def _validate_and_fix_enrichment_data(
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Валидирует и исправляет данные обогащения."""
        if not isinstance(data, list):
            print(f"⚠️  WARNING: Парсер вернул не список, а {type(data).__name__}")
            return []

        # Фильтруем только словари
        valid_items = []
        for item in data:
            if isinstance(item, dict):
                valid_items.append(item)

        # Убеждаемся, что все поля присутствуют
        required_fields = [
            "chunk_id",
            "context",
            "geo",
            "metrics",
            "years",
            "time_granularity",
            "oked",
        ]
        for item in valid_items:
            for field in required_fields:
                if field not in item:
                    item[field] = None

        return valid_items
