"""
LLM-обогащение чанков метаданными через Ollama.

LLMEnricher последовательно обрабатывает чанки, извлекая
структурированные метаданные (search_context, geo, metrics, units, years)
для гибридного поиска. Поддерживает retry, периодический
сброс контекста модели и постобработку результатов.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import time
import logging
from tqdm import tqdm
from sys import stdout

from src.core.models import Chunk
from src.enrichers.ollama_client import OllamaClient
from src.enrichers.config import EnricherConfig
from src.enrichers.parsers import parse_single_enrichment
from src.enrichers.rule_metadata_extractor import RuleMetadataExtractor

# Optional project-specific helpers
try:
    from src.utils.logger import get_logger
    from src.utils.chunk_validator import ChunkValidator
    from src.ingestion.chunk_filter import ChunkFilter
    from src.utils.post_processor import EnrichmentPostProcessor
except Exception:
    ChunkValidator = None
    ChunkFilter = None
    EnrichmentPostProcessor = None

logger = logging.getLogger(__name__)


class LLMEnricher:
    """
    Последовательное обогащение чанков метаданными через Ollama LLM.

    Для каждого чанка:
    1. Формирует промпт с текстом чанка
    2. Отправляет запрос в LLM и парсит JSON-ответ
    3. Валидирует и нормализует полученные метаданные
    4. Применяет постобработку (фильтрация metrics, сборка search_context)

    Поддерживает retry при ошибках LLM/парсинга и периодический сброс
    контекста модели для предотвращения деградации качества ответов.
    """

    def __init__(self, llm_client: OllamaClient, config: Optional[EnricherConfig] = None):
        self._llm = llm_client
        self._cfg = config or EnricherConfig()
        self._validator = ChunkValidator() if ChunkValidator else None
        self._chunk_filter = ChunkFilter(skip_first_pages=5) if ChunkFilter else None
        self._post = EnrichmentPostProcessor() if EnrichmentPostProcessor else None
        self._rules = RuleMetadataExtractor()
        self._rag_logger = get_logger()
        self._chunks_since_reset = 0

    def enrich_chunks(
        self,
        pdf_name: str,
        chunks: List[Chunk],
        *,
        show_progress: bool = False,
    ) -> List[Chunk]:
        """
        Обогащает список чанков, сохраняя исходный порядок.

        Args:
            pdf_name: Имя PDF-файла (для логирования и промптов).
            chunks: Список чанков для обогащения.
            show_progress: Показывать tqdm progress bar.

        Returns:
            Список обогащённых чанков в исходном порядке.
        """
        if not chunks:
            return []

        # Filter chunks if filter available
        if self._chunk_filter:
            data_chunks, metadata_chunks, skip_chunks = self._chunk_filter.filter_chunks(chunks)
        else:
            data_chunks, metadata_chunks, skip_chunks = chunks, [], []

        # Set default search text for metadata/skip chunks
        for ch in (metadata_chunks + skip_chunks):
            ch.search_context = ch.text[:200] if ch.text else "нет текста"
            self._rules.apply_to_chunk(ch)

        to_process = data_chunks
        resulting = metadata_chunks + skip_chunks
        start_time = time.time()
        processed = 0

        progress_iter = tqdm(
            to_process,
            total=len(to_process),
            desc=f"LLM enrich: {pdf_name}",
            mininterval=0.5,
            disable=not show_progress,
            file=stdout,
            colour="green",
        )

        for chunk in progress_iter:
            try:
                enriched_chunk = self._enrich_single_chunk(pdf_name, chunk)
                resulting.append(enriched_chunk or chunk)

                processed += 1
                self._chunks_since_reset += 1

                if self._chunks_since_reset >= self._cfg.reset_interval:
                    try:
                        ok = self._llm.reset_context()
                        logger.info(
                            "LLM context reset (ok=%s) after %d chunks",
                            ok,
                            self._chunks_since_reset,
                        )
                    except Exception as e:
                        logger.warning("Error during LLM reset: %s", e)
                    self._chunks_since_reset = 0

            except Exception as e:
                logger.exception(
                    "Error enriching chunk %s: %s",
                    getattr(chunk, "id", None),
                    e,
                )
                if not chunk.search_context:
                    chunk.search_context = chunk.text[:256] if chunk.text else "нет текста"
                resulting.append(chunk)

        # Preserve original order
        order = {ch.id: i for i, ch in enumerate(chunks)}
        resulting.sort(key=lambda c: order.get(c.id, 999999))

        elapsed = time.time() - start_time
        logger.info("LLM enrichment done for %s: processed=%d total_returned=%d elapsed=%.2fs",
                    pdf_name, processed, len(resulting), elapsed)
        return resulting

    def _enrich_single_chunk(self, pdf_name: str, chunk: Chunk) -> Optional[Chunk]:
        """
        Обогащает один чанк через LLM с retry.

        Args:
            pdf_name: Имя PDF-файла.
            chunk: Чанк для обогащения.

        Returns:
            Обогащённый Chunk или None при неудаче парсинга после всех попыток.
        """
        self._rules.apply_to_chunk(chunk)
        system_prompt = self._build_system_prompt()
        prompt = self._build_prompt(pdf_name, chunk)

        raw_response = ""
        parsed: Optional[Dict[str, Any]] = None

        for attempt in range(self._cfg.max_retries):
            try:
                raw_response = self._llm.generate(
                    prompt,
                    system_prompt=system_prompt,
                    format="json",
                    keep_alive=self._cfg.keep_alive,
                    options=self._cfg.request_options or {},
                )
            except Exception as e:
                logger.warning(
                    "LLM generate failed for chunk %s attempt %d: %s",
                    chunk.id, attempt + 1, e
                )
                if attempt < self._cfg.max_retries - 1:
                    time.sleep(0.5)
                continue

            parsed = parse_single_enrichment(raw_response)
            if parsed:
                break
            logger.debug(
                "Parse failed for chunk %s on attempt %d, raw_len=%d",
                chunk.id, attempt + 1, len(raw_response)
            )
            if attempt < self._cfg.max_retries - 1:
                time.sleep(0.5)

        if not parsed:
            self._rag_logger.log_llm_enrichment_fail(
                pdf_name=pdf_name,
                chunk_id=chunk.id,
                page=getattr(chunk, "page", None),
                chunk_text=(chunk.text or "")[:1200],
                system_prompt=system_prompt,
                prompt=prompt,
                raw_response=raw_response,
                attempts=self._cfg.max_retries,
            )
            return None

        # ---------------- Apply parsed fields with normalization ---------------- #
        validator = self._validator or ChunkValidator()

        # search_context
        search_context_raw = parsed.get("search_context") or chunk.search_context or chunk.text or "нет текста"
        chunk.search_context = validator.normalize_search_context(str(search_context_raw))

        # metrics
        if "metrics" in parsed:
            parsed_metrics = validator.normalize_metrics(parsed.get("metrics"))
            chunk.metrics = self._merge_list(chunk.metrics, parsed_metrics)

        # years
        if "years" in parsed:
            parsed_years = validator.normalize_years(parsed.get("years"))
            if parsed_years:
                chunk.years = sorted(set([*(chunk.years or []), *parsed_years]))

        if "geo" in parsed:
            chunk.geo = self._merge_list(chunk.geo, parsed["geo"])

        if "units" in parsed:
            parsed_units = validator.normalize_units(parsed.get("units"))
            chunk.units = self._merge_list(chunk.units, parsed_units)

        # Optional validation
        if self._validator:
            try:
                valid = self._validator.validate_chunk(parsed, check_uniqueness=False)
                if not getattr(valid, "is_valid", True):
                    logger.debug("Validation flagged issues for chunk %s", chunk.id)
            except Exception:
                logger.debug("Validator error for chunk %s (ignored)", chunk.id)

        # Optional post-processing
        if self._post:
            try:
                chunk = self._post.process_chunk(chunk)
            except Exception:
                logger.debug("Post-processor failed for chunk %s (ignored)", chunk.id)

        return chunk

    @staticmethod
    def _merge_list(existing, incoming):
        values = []
        existing_items = [existing] if isinstance(existing, str) else (existing or [])
        incoming_items = [incoming] if isinstance(incoming, str) else (incoming or [])
        for item in existing_items:
            if item is not None:
                values.append(str(item).strip())
        for item in incoming_items:
            if item is not None:
                values.append(str(item).strip())
        seen = set()
        result = []
        for value in values:
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                result.append(value)
        return result or None

    def _build_system_prompt(self) -> str:
        return (
            "Ты — аналитик официальной статистики.\n"
            "Входной чанк — фрагмент статистического сборника, чаще всего табличные данные. "
            "Твоя задача — извлечь МЕТАДАННЫЕ для семантического поиска.\n"
            "Верни ТОЛЬКО один валидный JSON с полями:\n"
            "- search_context: строка, 1 предложение, которое обобщённо описывает содержание таблиц в чанке, без значений показателей\n"
            "- geo: список территорий, упомянутых в чанке (без агрегатов 'в целом по стране', 'итого')\n"
            "- metrics: список названий показателей как в таблице, без чисел, лет и единиц измерения\n"
            "- units: список единиц измерения как в таблице, например 'тыс. человек', 'млн рублей', 'процентов'\n"
            "- years: список лет, явно упомянутых в чанке\n\n"
            "Правила:\n"
            "1. Используй только данные из чанка.\n"
            "2. Не повторяй числовые строки таблицы в search_context.\n"
            "3. Если поле невозможно определить — верни пустой список [] или null.\n"
            "4. Ответ только JSON, без комментариев.\n\n"
            "Пример корректного ответа:\n"
            "{\n"
            '  "search_context": "Производство основных видов промышленной продукции на душу населения в Беларуси и России за 2018-2021 годы",\n'
            '  "geo": ["Беларусь", "Россия"],\n'
            '  "metrics": ["Производство нефти", "Производство мяса"],\n'
            '  "units": ["кг на душу населения"],\n'
            '  "years": [2018, 2019, 2020, 2021]\n'
            "}\n"
        )

    def _build_prompt(self, pdf_name: str, chunk: Chunk) -> str:
        text_snip = (chunk.text or "")[: self._cfg.prompt_char_limit]
        hints = []
        if chunk.search_context:
            hints.append(f"Предварительно извлечено правилами: {chunk.search_context}")
        if chunk.section:
            hints.append(f"Раздел: {chunk.section}")
        hints_text = "\n".join(hints)
        return (
            f"Документ: {pdf_name}\n"
            f"page: {chunk.page}\n\n"
            f"{hints_text}\n\n"
            "Содержимое чанка:\n"
            f"{text_snip}\n\n"
            "Сформируй JSON-объект строго по инструкции system prompt."
        )
