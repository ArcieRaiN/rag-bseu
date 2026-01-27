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

        # Разделяем чанки на те, что нужно обработать LLM и те, что можно пропустить
        chunks_to_process: List[Chunk] = []
        chunks_to_skip: List[Chunk] = []

        for chunk in chunks:
            if chunk.page <= skip_first_pages:
                # Для первых страниц просто используем text как context
                chunk.context = chunk.text[:200] if chunk.text else "нет текста"
                chunks_to_skip.append(chunk)
            else:
                chunks_to_process.append(chunk)

        if not chunks_to_process:
            return chunks

        # Настройки батчирования через env (с fallback на параметры конструктора)
        batch_size = int(os.getenv("RAG_ENRICH_BATCH_SIZE", str(self._batch_size)))
        batch_concurrency = int(os.getenv("RAG_ENRICH_CONCURRENCY", str(self._batch_concurrency)))
        batch_size = max(1, batch_size)
        batch_concurrency = max(1, min(batch_concurrency, 8))

        total_batches = (len(chunks_to_process) + batch_size - 1) // batch_size
        print(
            f"   Обработка {len(chunks_to_process)} чанков "
            f"(пропущено {len(chunks_to_skip)} чанков со страниц 1-{skip_first_pages})"
        )
        print(
            f"   Батчи: размер={batch_size}, батчей={total_batches}, "
            f"параллельность={batch_concurrency}"
        )

        all_enriched_chunks: List[Chunk] = chunks_to_skip.copy()

        # Параллелим только батчи (а не каждый чанк)
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=min(batch_concurrency, total_batches)) as executor:
            futures = []
            for start in range(0, len(chunks_to_process), batch_size):
                batch = chunks_to_process[start : start + batch_size]
                
                # Получаем контекст из rolling buffer (последние 2 чанка)
                previous_chunks = self._context_buffer.get_context(num_chunks=2)
                
                fut = executor.submit(
                    self._enrich_single_batch,
                    pdf_name,
                    batch,
                    previous_chunks,
                )
                fut._rag_submit_ts = time.time()  # type: ignore[attr-defined]
                futures.append((fut, batch))

            completed_batches = 0
            completed_chunks = 0

            for fut, original_batch in futures:
                try:
                    enriched_batch = fut.result()
                    submit_ts = getattr(fut, "_rag_submit_ts", None)
                    batch_time = (time.time() - submit_ts) if submit_ts else 0.0
                    
                    all_enriched_chunks.extend(enriched_batch)
                    
                    # Добавляем обогащенные чанки в rolling buffer
                    self._context_buffer.add_batch(enriched_batch)
                    
                    completed_batches += 1
                    completed_chunks += len(enriched_batch)
                    self._chunks_since_reset += len(enriched_batch)

                    # Сброс контекста каждые N чанков
                    if self._chunks_since_reset >= self._reset_interval:
                        print(
                            f"   🔄 Сброс контекста модели после {completed_chunks} чанков..."
                        )
                        self._llm.reset_context()
                        self._chunks_since_reset = 0
                        time.sleep(0.5)

                    elapsed = time.time() - start_time
                    rate = completed_chunks / elapsed * 3600 if elapsed > 0 else 0
                    print(
                        f"   Батч {completed_batches}/{total_batches}: "
                        f"{len(enriched_batch)} чанков за {batch_time:.1f}с | "
                        f"Всего: {completed_chunks}/{len(chunks_to_process)} | "
                        f"Скорость: {rate:.0f} чанков/час"
                    )

                    # Логируем прогресс батча
                    self._logger.log_prepare_db(
                        "batch",
                        pdf_name=pdf_name,
                        batch_number=completed_batches,
                        total_batches=total_batches,
                        chunks_in_batch=len(enriched_batch),
                        elapsed_time=batch_time,
                        total_chunks_processed=completed_chunks,
                        rate_per_hour=rate,
                    )
                except Exception as e:
                    submit_ts = getattr(fut, "_rag_submit_ts", None)
                    batch_time = (time.time() - submit_ts) if submit_ts else 0.0
                    completed_batches += 1
                    print(
                        f"   ⚠️  Ошибка при обработке батча {completed_batches}/{total_batches} "
                        f"(время: {batch_time:.1f}с): {e}"
                    )
                    import traceback
                    traceback.print_exc()
                    # Fallback: добавляем чанки без обогащения
                    for ch in original_batch:
                        if not ch.context:
                            ch.context = ch.text[:200] if ch.text else "нет текста"
                    all_enriched_chunks.extend(original_batch)
                    completed_chunks += len(original_batch)
                    self._chunks_since_reset += len(original_batch)

        # Сортируем по исходному порядку
        chunk_order = {ch.id: i for i, ch in enumerate(chunks)}
        all_enriched_chunks.sort(key=lambda ch: chunk_order.get(ch.id, 999999))

        return all_enriched_chunks

    def _enrich_single_batch(
        self,
        pdf_name: str,
        chunks: List[Chunk],
        previous_chunks: List[Chunk] | None = None,
    ) -> List[Chunk]:
        """
        Обогащает один батч чанков с учетом контекста предыдущих страниц.
        
        Args:
            pdf_name: Имя PDF файла
            chunks: Список чанков для обогащения
            previous_chunks: Список чанков с предыдущих страниц для контекста
            
        Returns:
            Список обогащенных чанков
        """
        if not chunks:
            return []

        # Формируем данные чанков для промпта
        chunks_data = []
        for ch in chunks:
            chunks_data.append({
                "chunk_id": ch.id,
                # Режем текст для ускорения генерации
                "text": (ch.text or "")[:350],
                "page": ch.page,
            })

        # Подготавливаем контекст предыдущих страниц
        previous_context = None
        if previous_chunks:
            prev_texts = []
            for prev_ch in previous_chunks[-2:]:  # Последние 2 чанка
                if prev_ch.text:
                    prev_texts.append(f"Страница {prev_ch.page}: {prev_ch.text[:200]}")
            if prev_texts:
                previous_context = "\n".join(prev_texts)

        # System prompt
        system_prompt = (
            "Ты — аналитик по официальной статистике Республики Беларусь. "
            "Твоя задача — обогатить чанки документа структурированными метаданными. "
            "STRICT RULES: Output ONLY a valid JSON array - No text before or after JSON - "
            "No markdown, comments, explanations - chunk_id must exactly match input. "
            "Ограничения: context <= 256 символов; metrics максимум 5 элементов; "
            "years максимум 5 элементов."
        )

        # Формируем промпт
        prompt = self._build_prompt(pdf_name, chunks_data, previous_context)

        # Параметры запроса
        keep_alive = os.getenv("RAG_OLLAMA_KEEP_ALIVE", "5m")
        req_options = {
            "temperature": 0,
            "top_p": 1,
            "num_predict": min(250 * len(chunks_data) + 100, 3000),
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
            chunks_count=len(chunks_data),
            chunk_ids=[c["chunk_id"] for c in chunks_data],
            pages=[c["page"] for c in chunks_data],
            system_prompt=system_prompt,
            prompt=prompt,
            ollama_config=ollama_config,
        )

        # Вызываем LLM с повторными попытками
        max_retries_for_quality = 2
        enriched_data = []
        raw_response = ""

        for attempt in range(max_retries_for_quality):
            raw_response = self._llm.generate(
                prompt,
                system_prompt=system_prompt,
                format="json",
                keep_alive=keep_alive,
                options=req_options,
            )

            # Парсим JSON-ответ
            enriched_data = self._parse_llm_batch_enrichment(raw_response)

            # Проверяем качество ответа
            valid_items = [item for item in enriched_data if isinstance(item, dict)]
            if len(valid_items) >= len(chunks_data) * 0.8:  # Хотя бы 80% чанков обогащено
                break

            if attempt < max_retries_for_quality - 1:
                print(
                    f"   ⚠️  Низкое качество ответа ({len(valid_items)}/{len(chunks_data)}), "
                    f"повторная попытка {attempt + 2}/{max_retries_for_quality}..."
                )
                time.sleep(1)

        # Логируем ответ
        valid_enriched_data_for_log = [
            item for item in enriched_data if isinstance(item, dict)
        ]
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

        # Обогащаем чанки данными от LLM
        enriched_chunks = self._apply_enrichment_data(chunks, enriched_data)

        return enriched_chunks

    def _build_prompt(
        self,
        pdf_name: str,
        chunks_data: List[Dict[str, Any]],
        previous_context: Optional[str],
    ) -> str:
        """Строит промпт для LLM."""
        context_section = ""
        if previous_context:
            context_section = (
                f"\nКОНТЕКСТ ПРЕДЫДУЩИХ СТРАНИЦ "
                f"(для понимания общего контекста документа):\n"
                f"{previous_context}\n\n"
            )

        if len(chunks_data) == 1:
            chunk = chunks_data[0]
            return (
                f"Документ: {pdf_name}\n\n"
                f"{context_section}"
                f"Чанк для обработки:\n"
                f"ID: {chunk['chunk_id']}\n"
                f"Страница: {chunk['page']}\n"
                f"Текст: {chunk['text'][:500]}...\n\n"
                "Верни JSON-массив с одним объектом. Формат:\n"
                '[{"chunk_id":"...","context":"...","geo":null,"metrics":null,'
                '"years":null,"time_granularity":null,"oked":null}]\n\n'
                "Поля:\n"
                "- chunk_id: точно такой же как ID выше\n"
                "- context: краткое, точное описание содержания чанка на русском языке "
                "(1-2 предложения), отражающее что находится в чанке с учётом предыдущих страниц\n"
                "- geo: название региона/города/области или null\n"
                "- metrics: список метрик в нижнем регистре на русском (максимум 5) или null\n"
                "- years: список годов (максимум 5) или null\n"
                "- time_granularity: 'year'/'quarter'/'month'/'day' или null\n"
                "- oked: код ОКЭД или null\n\n"
                "КРИТИЧЕСКИ ВАЖНО: Верни массив [{}], НЕ объект {}!"
            )
        else:
            return (
                f"Документ: {pdf_name}\n\n"
                f"{context_section}"
                f"Обработай {len(chunks_data)} чанков. "
                f"Верни JSON-массив из РОВНО {len(chunks_data)} объектов.\n\n"
                "Формат ответа (пример для 2 чанков):\n"
                '[{"chunk_id":"doc.pdf::page1::chunk0","context":"Краткое описание",'
                '"geo":null,"metrics":["метрика1"],"years":[2024],"time_granularity":null,'
                '"oked":null},'
                '{"chunk_id":"doc.pdf::page1::chunk1","context":"Другое описание",'
                '"geo":"Минск","metrics":null,"years":null,"time_granularity":"year",'
                '"oked":null}]\n\n'
                "Правила для каждого объекта:\n"
                "- chunk_id: точно как в входных данных (обязательно!)\n"
                "- context: краткое, точное описание содержания чанка на русском языке "
                "(1-2 предложения), отражающее что находится в чанке с учётом предыдущих страниц\n"
                "- metrics: только реальные названия из текста, нижний регистр, русский, "
                "максимум 5, или null\n"
                "- years: только целые числа, максимум 5, или null\n"
                "- geo: название региона/города/области или null\n"
                "- time_granularity: 'year'/'quarter'/'month'/'day' или null\n"
                "- oked: код ОКЭД или null\n\n"
                "Входные чанки:\n"
                f"{json.dumps(chunks_data, ensure_ascii=False, separators=(',', ':'))}\n\n"
                "КРИТИЧЕСКИ ВАЖНО:\n"
                "1. Верни МАССИВ [{}, {}, ...], начинается с '[' и заканчивается ']'\n"
                "2. НЕ возвращай объект {}\n"
                f"3. В массиве должно быть РОВНО {len(chunks_data)} объектов\n"
                "4. Каждый объект должен содержать chunk_id из входных данных\n"
                "5. НЕ используй unicode escape-последовательности (\\u0412) - "
                "пиши русский текст напрямую"
            )

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

        # Создаём словарь chunk_id -> enriched_data
        enriched_map: Dict[str, Dict[str, Any]] = {}
        for item in valid_enriched_data:
            chunk_id = item.get("chunk_id")
            if chunk_id:
                enriched_map[str(chunk_id)] = item

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

        # Применяем данные к чанкам
        enriched_chunks: List[Chunk] = []
        for ch in chunks:
            enriched = enriched_map.get(ch.id, {})

            # Если для одного чанка не нашли по ID, но есть один объект - используем его
            if not enriched and len(chunks) == 1 and valid_enriched_data:
                enriched = valid_enriched_data[0]
                if "chunk_id" not in enriched:
                    enriched["chunk_id"] = ch.id

            # Валидируем и нормализуем данные
            validation_result = self._validator.validate_chunk(enriched, check_uniqueness=False)
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
