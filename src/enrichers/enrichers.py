# src/enrichers/enrichers.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
import time
import logging
from tqdm import tqdm
from sys import stdout

from src.core.models import Chunk
from src.enrichers.client import OllamaClient
from src.enrichers.config import EnricherConfig
from src.enrichers.parsers import parse_single_enrichment

# Optional project-specific helpers
try:
    from src.utils.json_validator import ChunkValidator
    from src.ingestion.chunk_filter import ChunkFilter
    from src.utils.post_processor import EnrichmentPostProcessor
except Exception:
    ChunkValidator = None
    ChunkFilter = None
    EnrichmentPostProcessor = None

logger = logging.getLogger(__name__)


class LLMEnricher:
    """
    Enriches chunks one by one using OllamaClient.

    Features:
    - retries on LLM or parse failure
    - optional validation and post-processing
    - logs progress and errors
    """

    def __init__(self, llm_client: OllamaClient, config: Optional[EnricherConfig] = None):
        self._llm = llm_client
        self._cfg = config or EnricherConfig()
        self._validator = ChunkValidator() if ChunkValidator else None
        self._chunk_filter = ChunkFilter(skip_first_pages=3) if ChunkFilter else None
        self._post = EnrichmentPostProcessor() if EnrichmentPostProcessor else None
        self._chunks_since_reset = 0

    def enrich_chunks(
        self,
        pdf_name: str,
        chunks: List[Chunk],
        *,
        show_progress: bool = False,
    ) -> List[Chunk]:
        """Sequentially enrich a list of chunks. Returns enriched chunks in original order."""
        if not chunks:
            return []

        # Filter chunks if filter available
        if self._chunk_filter:
            data_chunks, metadata_chunks, skip_chunks = self._chunk_filter.filter_chunks(chunks)
        else:
            data_chunks, metadata_chunks, skip_chunks = chunks, [], []

        # Set default context for metadata/skip chunks
        for ch in (metadata_chunks + skip_chunks):
            ch.context = ch.text[:200] if ch.text else "нет текста"

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
                if not chunk.context:
                    chunk.context = chunk.text[:256] if chunk.text else "нет текста"
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
        Enrich a single chunk using LLM.
        Returns enriched Chunk or None if parsing fails.
        """
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

            parsed = parse_single_enrichment(raw_response, chunk.id)
            if parsed:
                break
            logger.debug(
                "Parse failed for chunk %s on attempt %d, raw_len=%d",
                chunk.id, attempt + 1, len(raw_response)
            )
            if attempt < self._cfg.max_retries - 1:
                time.sleep(0.5)

        if not parsed:
            logger.warning(
                "Failed to parse enrichment for chunk %s after %d attempts",
                chunk.id, self._cfg.max_retries
            )
            return None

        # ---------------- Apply parsed fields with normalization ---------------- #
        validator = self._validator or ChunkValidator()

        # context
        context_raw = parsed.get("context") or chunk.text or "нет текста"
        chunk.context = validator.normalize_context(str(context_raw))

        # metrics
        if "metrics" in parsed:
            chunk.metrics = validator.normalize_metrics(parsed.get("metrics"))

        # years
        if "years" in parsed:
            chunk.years = validator.normalize_years(parsed.get("years"))

        # other fields without normalization
        for field in ["geo", "time_granularity", "oked"]:
            if field in parsed:
                setattr(chunk, field, parsed[field])

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


    def _build_system_prompt(self) -> str:
        return (
            "Вы — аналитик по официальной статистике. "
            "По входному чанк-объекту нужно вернуть ТОЛЬКО один JSON-объект с полями: "
            "chunk_id, context, geo, metrics, years, time_granularity, oked. "
            "Никакого другого текста."
        )

    def _build_prompt(self, pdf_name: str, chunk: Chunk) -> str:
        text_snip = (chunk.text or "")[:800]
        return (
            f"Документ: {pdf_name}\n"
            f"Чанк: id={chunk.id} page={chunk.page}\n"
            f"Текст:\n{text_snip}\n\n"
            "Верни JSON-объект с полями: chunk_id, context (1-2 предложения), "
            "geo, metrics, years, time_granularity, oked."
        )
