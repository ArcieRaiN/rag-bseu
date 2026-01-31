from __future__ import annotations

"""
Модуль для LLM-обогащения чанков с использованием rolling context buffer.

Отвечает за:
- Обогащение чанков через Ollama (1 чанк = 1 запрос, без батчинга)
- Rolling context buffer (последние N чанков вместо всех)
- Парсинг JSON-ответов от LLM
- Валидацию и нормализацию данных
"""

from typing import List, Any, Optional, Deque, Dict
from collections import deque
import os
import time
import json
import codecs
from tqdm import tqdm
from sys import stdout
from colorama import init
init()


from src.main.models import Chunk
from src.main.ollama_client import OllamaClient
from src.main.logger import get_logger
from src.prepare_db.json_validator import ChunkValidator
from src.prepare_db.chunk_filter import ChunkFilter
from src.prepare_db.post_processor import EnrichmentPostProcessor


class RollingContextBuffer:
    """
    Rolling context buffer для хранения последних N чанков.
    """

    def __init__(self, max_size: int = 10):
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
        """Возвращает последние num_chunks чанков (если их меньше, возвращает все)."""
        if num_chunks <= 0:
            return []
        buf_list = list(self._buffer)
        if not buf_list:
            return []
        return buf_list[-num_chunks:]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class LLMEnricher:
    """
    Класс для LLM-обогащения чанков c использованием rolling context buffer (1 чанк = 1 запрос).
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        max_parallel_requests: int = 4,
        context_buffer_size: int = 10,
        reset_interval: int = 50,
    ):
        self._llm = llm_client
        self._max_parallel_requests = max(1, int(max_parallel_requests))
        self._reset_interval = max(1, int(reset_interval))
        self._logger = get_logger()
        self._validator = ChunkValidator()
        self._chunk_filter = ChunkFilter(skip_first_pages=3)
        self._post_processor = EnrichmentPostProcessor()
        self._context_buffer = RollingContextBuffer(max_size=context_buffer_size)
        self._chunks_since_reset = 0

    def enrich_chunks(
            self,
            pdf_name: str,
            chunks: List[Chunk],
            skip_first_pages: int = 3,
            show_progress: bool = True,
    ) -> List[Chunk]:

        """
        Обогащает чанки последовательно (1 чанк = 1 запрос).
        Возвращает список чанков в исходном порядке.
        """
        if not chunks:
            return []

        # Фильтрация
        data_chunks, metadata_chunks, skip_chunks = self._chunk_filter.filter_chunks(chunks)

        # Простое обогащение для метаданных / пропусков
        for chunk in metadata_chunks + skip_chunks:
            chunk.context = chunk.text[:200] if chunk.text else "нет текста"

        chunks_to_process = data_chunks
        chunks_to_skip = metadata_chunks + skip_chunks

        if not chunks_to_process:
            return chunks  # ничего для LLM

        total_chunks = len(chunks_to_process)
        self._logger.log("llm_enricher", {"event": "start", "total_chunks": total_chunks})

        all_enriched_chunks: List[Chunk] = chunks_to_skip.copy()
        start_time = time.time()
        completed_chunks = 0

        # Последовательная обработка — проще и без race condition для контекста
        progress_iter = tqdm(
            chunks_to_process,
            total=len(chunks_to_process),
            desc=f"LLM enrich: {pdf_name}",
            mininterval=0.5,
            disable=not show_progress,
            file=stdout,
            colour="green",
        )

        for idx, chunk in enumerate(progress_iter):
            original_chunk = chunk
            enriched_chunk: Optional[Chunk] = None

            try:
                previous_ctx = self._context_buffer.get_context(num_chunks=2)
                enriched_chunk = self._enrich_single_chunk(pdf_name, chunk, previous_ctx)

                if enriched_chunk:
                    self._context_buffer.add(enriched_chunk)
                    all_enriched_chunks.append(enriched_chunk)
                else:
                    if not original_chunk.context:
                        original_chunk.context = (
                            original_chunk.text[:200] if original_chunk.text else "нет текста"
                        )
                    self._context_buffer.add(original_chunk)
                    all_enriched_chunks.append(original_chunk)

                completed_chunks += 1
                self._chunks_since_reset += 1

                # reset контекста модели
                if self._chunks_since_reset >= self._reset_interval:
                    self._logger.log(
                        "llm_enricher",
                        {"event": "model_context_reset", "after_chunks": completed_chunks},
                    )
                    try:
                        self._llm.reset_context()
                    except Exception as e:
                        self._logger.log(
                            "llm_enricher",
                            {"event": "reset_error", "error": str(e)},
                        )
                    self._chunks_since_reset = 0

                # логируем прогресс (как было)
                if completed_chunks % 10 == 0 or completed_chunks == total_chunks:
                    elapsed = time.time() - start_time
                    rate = completed_chunks / elapsed * 3600 if elapsed > 0 else 0
                    self._logger.log(
                        "llm_enricher",
                        {
                            "event": "progress",
                            "completed": completed_chunks,
                            "total": total_chunks,
                            "rate_per_hour": rate,
                        },
                    )

                # 👉 опционально: показываем скорость в tqdm
                progress_iter.set_postfix(
                    rate=f"{rate:.1f}/h",
                    ctx=len(self._context_buffer),
                )

            except Exception as e:
                self._logger.log(
                    "llm_enricher",
                    {
                        "event": "error",
                        "chunk_id": getattr(original_chunk, "id", None),
                        "error": str(e),
                    },
                )
                if not original_chunk.context:
                    original_chunk.context = (
                        original_chunk.text[:200] if original_chunk.text else "нет текста"
                    )
                all_enriched_chunks.append(original_chunk)
                completed_chunks += 1
                self._chunks_since_reset += 1

        # Сортируем по исходному порядку документа
        chunk_order = {ch.id: i for i, ch in enumerate(chunks)}
        all_enriched_chunks.sort(key=lambda ch: chunk_order.get(ch.id, 999999))

        self._logger.log("llm_enricher", {"event": "done", "total_returned": len(all_enriched_chunks)})
        return all_enriched_chunks

    def _enrich_single_chunk(
        self,
        pdf_name: str,
        chunk: Chunk,
        previous_chunks: Optional[List[Chunk]] = None,
    ) -> Optional[Chunk]:
        """
        Обогащает один чанк с учетом контекста предыдущих страниц.
        """
        previous_context: Optional[str] = None
        if previous_chunks:
            prev_texts: List[str] = []
            for prev_ch in previous_chunks[-2:]:
                if prev_ch and getattr(prev_ch, "text", None):
                    prev_texts.append(f"Страница {prev_ch.page}: {prev_ch.text[:200]}")
            if prev_texts:
                previous_context = "\n".join(prev_texts)

        system_prompt = self._build_system_prompt_single()
        prompt = self._build_prompt_single(pdf_name, chunk, previous_context)

        keep_alive = os.getenv("RAG_OLLAMA_KEEP_ALIVE", "5m")
        req_options = {"temperature": 0, "top_p": 1, "num_predict": 300}

        ollama_config = {
            "model": getattr(getattr(self._llm, "config", None), "model", None),
            "base_url": getattr(getattr(self._llm, "config", None), "base_url", None),
            "timeout": getattr(getattr(self._llm, "config", None), "timeout", None),
            "format": "json",
            "options": req_options,
        }

        # Лог запроса
        try:
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
        except Exception:
            # на случай, если logger не поддерживает log_llm_enrichment
            self._logger.log("llm_enricher", {"event": "request", "chunk_id": chunk.id})

        max_retries = 2
        enriched_data: Optional[Dict[str, Any]] = None
        raw_response = ""

        for attempt in range(max_retries):
            try:
                raw_response = self._llm.generate(
                    prompt,
                    system_prompt=system_prompt,
                    format="json",
                    keep_alive=keep_alive,
                    options=req_options,
                )
            except Exception as e:
                # Сетевая/LLM ошибка — логируем и пробуем снова (до max_retries)
                self._logger.log("llm_enricher", {"event": "generate_error", "chunk_id": chunk.id, "attempt": attempt, "error": str(e)})
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                continue

            enriched_data = self._parse_llm_single_enrichment(raw_response, chunk.id)

            if enriched_data:
                # Гарантируем целостность id
                enriched_data["chunk_id"] = chunk.id
                break
            elif attempt < max_retries - 1:
                # парсинг провалился — пробуем ещё раз
                self._logger.log("llm_enricher", {"event": "parse_retry", "chunk_id": chunk.id, "attempt": attempt})
                time.sleep(0.5)

        parsed_successfully = enriched_data is not None

        # Лог ответа
        try:
            self._logger.log_llm_enrichment(
                event="response",
                pdf_name=pdf_name,
                chunks_count=1,
                chunk_ids=[chunk.id],
                raw_response=raw_response,
                parsed_items=1 if parsed_successfully else 0,
                parsed_with_chunk_id=1 if (parsed_successfully and enriched_data and enriched_data.get("chunk_id")) else 0,
            )
        except Exception:
            self._logger.log("llm_enricher", {"event": "response", "chunk_id": chunk.id, "parsed": parsed_successfully})

        if not enriched_data:
            return None

        enriched_chunk = self._apply_enrichment_data_single(chunk, enriched_data)
        enriched_chunk = self._post_processor.process_chunk(enriched_chunk)
        return enriched_chunk

    def _build_system_prompt_single(self) -> str:
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

    def _build_prompt_single(
        self,
        pdf_name: str,
        chunk: Chunk,
        previous_context: Optional[str],
    ) -> str:
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

    def _apply_enrichment_data_single(
        self,
        chunk: Chunk,
        enriched_data: Dict[str, Any],
    ) -> Chunk:
        """
        Применяет данные обогащения к одному чанку.
        """
        # Гарантируем ID оригинала
        enriched_data["chunk_id"] = chunk.id

        validation_result = self._validator.validate_chunk(enriched_data, check_uniqueness=False)

        # Проверка качества metrics
        if enriched_data.get("metrics"):
            metrics_warnings = self._validator.validate_metrics_quality(
                enriched_data["metrics"],
                chunk_text=chunk.text,
            )
            if metrics_warnings:
                for warning in metrics_warnings:
                    self._logger.log("llm_enricher", {"event": "metrics_warning", "chunk_id": chunk.id, "warning": warning})

        # Нормализация при необходимости
        if not validation_result.is_valid:
            if enriched_data.get("context"):
                enriched_data["context"] = self._validator.normalize_context(str(enriched_data["context"]))
            if enriched_data.get("metrics"):
                enriched_data["metrics"] = self._validator.normalize_metrics(enriched_data["metrics"])
            if enriched_data.get("years"):
                enriched_data["years"] = self._validator.normalize_years(enriched_data["years"])

        # Обновляем context
        if enriched_data.get("context"):
            context_str = str(enriched_data.get("context"))
            if "\\u" in context_str:
                try:
                    context_str = codecs.decode(context_str, "unicode_escape")
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
        Возвращает dict или None.
        """
        if not raw:
            return None

        cleaned = raw.strip()

        # Удаляем code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Удаление ведущих пробельных символов
        while cleaned and cleaned[0] in [" ", "\n", "\r", "\t"]:
            cleaned = cleaned[1:]

        # 1) Прямой парсинг
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # если есть поля интереса — возврат
                if "chunk_id" in data or any(k in data for k in ["context", "geo", "metrics", "years"]):
                    if "chunk_id" not in data:
                        data["chunk_id"] = expected_chunk_id
                    return data
        except json.JSONDecodeError:
            pass

        # 2) Попытка найти объект { ... } внутри текста
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
