from __future__ import annotations

"""
LLM‑based reranking (PIPELINE 4).

Цель:
- переоценить Top‑10 кандидатов (ScoredChunk.hybrid_score)
- вернуть Top‑3 по rerank_score

ВАЖНО:
- rerank_score НЕ зависит от hybrid_score
- модуль не занимается генерацией финального ответа RAG
"""

from typing import List
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.models import EnrichedQuery, ScoredChunk
from src.core.config import RerankConfig
from src.enrichers.client import OllamaClient
from src.utils.logger import get_logger


class LLMReranker:
    """
    Prompt‑based reranker на основе Ollama.
    """

    def __init__(self, llm_client: OllamaClient, config: RerankConfig):
        self._llm = llm_client
        self._config = config
        self._logger = get_logger()

    def rerank(self, enriched_query: EnrichedQuery, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        if not candidates:
            return []

        rerank_start = time.time()
        system_prompt = self._build_system_prompt()
        original_order = {sc.chunk.id: i for i, sc in enumerate(candidates)}

        enriched_query_dict = {
            "query": enriched_query.query,
            "geo": enriched_query.geo,
            "metrics": enriched_query.metrics,
            "years": enriched_query.years,
            "time_granularity": enriched_query.time_granularity,
            "oked": enriched_query.oked,
        }

        max_workers = max(1, min(self._config.max_workers, len(candidates)))
        metadata_buckets: dict[str, str] = {}

        # Первая фаза: LLM оценивает каждый chunk параллельно
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self._score_single_chunk,
                    enriched_query,
                    sc,
                    system_prompt,
                    enriched_query_dict
                ): sc
                for sc in candidates
            }

            for future in as_completed(future_to_chunk):
                sc = future_to_chunk[future]
                try:
                    score = future.result()
                except Exception as e:
                    self._logger.log_llm_reranking(
                        event="error",
                        query=enriched_query.query,
                        candidates_count=1,
                        candidate_ids=[sc.chunk.id],
                        error=str(e),
                    )
                    score = 0.0

                # Применяем жёсткие priors по метаданным
                adjusted_score, bucket = self._apply_metadata_caps(enriched_query, sc, score)
                sc.rerank_score = adjusted_score
                metadata_buckets[sc.chunk.id] = bucket

        # Решаем, использовать rerank или оставить исходный порядок
        max_score = max((sc.rerank_score for sc in candidates), default=0.0)
        if max_score < self._config.min_relevance_score:
            candidates_sorted = sorted(candidates, key=lambda sc: original_order.get(sc.chunk.id, 1_000_000))
        else:
            candidates_sorted = sorted(candidates, key=lambda sc: sc.rerank_score, reverse=True)

        top_chunks = candidates_sorted[:self._config.top_k]

        self._log_final_rerank(enriched_query, candidates, top_chunks, metadata_buckets, rerank_start)
        return top_chunks

    # -------------------- Prompt & Parsing -------------------- #

    def _build_system_prompt(self) -> str:
        return (
            "Ты оцениваешь, помогает ли ОДИН фрагмент документа ответить на статистический запрос.\n"
            "Отвечай ТОЛЬКО одним JSON-объектом вида {\"id\": \"...\", \"score\": 0.0-1.0}.\n"
            "Не пиши ничего кроме JSON. Не используй markdown, комментарии и пояснения."
        )

    def _build_prompt_for_chunk(self, enriched_query: EnrichedQuery, scored_chunk: ScoredChunk) -> str:
        ch = scored_chunk.chunk
        query_block = {
            "query": enriched_query.query,
            "geo": enriched_query.geo,
            "metrics": enriched_query.metrics,
            "years": enriched_query.years,
        }
        chunk_block = {
            "id": ch.id,
            "context": ch.context,
            "geo": ch.geo,
            "metrics": ch.metrics,
            "years": ch.years,
            "time_granularity": ch.time_granularity,
            "oked": ch.oked,
        }
        instruction = (
            "Оцени, насколько этот фрагмент помогает ответить на запрос.\n"
            "Используй шкалу 0.0–1.0 с примерами: 0 — не относится, 0.3 — близко, 0.6 — частично, 0.9 — идеально.\n"
            "Верни один JSON с полями \"id\" и \"score\"."
        )
        return f"{instruction}\n\nЗапрос:\n{json.dumps(query_block, ensure_ascii=False, indent=2)}\n\nФрагмент:\n{json.dumps(chunk_block, ensure_ascii=False, indent=2)}"

    def _score_single_chunk(self, enriched_query: EnrichedQuery, scored_chunk: ScoredChunk,
                            system_prompt: str, enriched_query_dict: dict) -> float:
        """
        Оценка одного чанка через LLM. Возвращает score в диапазоне [0,1].
        """
        ch_id = scored_chunk.chunk.id
        last_raw = ""
        for attempt in range(1, self._config.max_retries + 1):
            prompt = self._build_prompt_for_chunk(enriched_query, scored_chunk)
            last_raw = self._llm.generate(
                prompt,
                system_prompt=system_prompt,
                model=self._config.model_name,
                temperature=self._config.temperature,
                format="json"
            )

            score = self._parse_single_score(last_raw, ch_id)
            if score is not None:
                return score
        return 0.0

    @staticmethod
    def _parse_single_score(raw: str, expected_id: str) -> float | None:
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            text = text[first_newline + 1 if first_newline != -1 else 0:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
            score = float(data.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except Exception:
            return None

    # -------------------- Метаданные как priors -------------------- #

    def _apply_metadata_caps(self, enriched_query: EnrichedQuery, scored_chunk: ScoredChunk, raw_score: float) -> tuple[float, str]:
        bucket = "good"
        score = max(0.0, min(1.0, raw_score))
        ch = scored_chunk.chunk

        if enriched_query.metrics:
            q_metrics = {m.lower() for m in enriched_query.metrics if isinstance(m, str)}
            ch_metrics = {m.lower() for m in (ch.metrics or []) if isinstance(m, str)}
            if not q_metrics.intersection(ch_metrics):
                score = min(score, 0.3)
                bucket = "no_metric"

        if enriched_query.geo and ch.geo:
            q_geo, c_geo = enriched_query.geo.lower(), ch.geo.lower()
            if q_geo not in c_geo and c_geo not in q_geo and bucket == "good":
                score = min(score, 0.3)
                bucket = "geo_mismatch"

        if enriched_query.years and not ch.years and bucket == "good":
            score = min(score, 0.5)
            bucket = "missing_years"

        if bucket == "good":
            if score < 0.05:
                bucket = "no_metric"
            elif score < 0.4:
                bucket = "partial"
        return score, bucket

    # -------------------- Logging -------------------- #

    def _log_final_rerank(self, enriched_query: EnrichedQuery, candidates: List[ScoredChunk],
                          top_chunks: List[ScoredChunk], metadata_buckets: dict[str, str], start_time: float) -> None:
        elapsed = time.time() - start_time
        top_10_before = [{ "id": sc.chunk.id, "hybrid_score": sc.hybrid_score } for sc in candidates]
        top_3_after = [{ "id": sc.chunk.id, "rerank_score": sc.rerank_score } for sc in top_chunks]
        self._logger.log_llm_reranking(
            event="final",
            query=enriched_query.query,
            candidates_count=len(candidates),
            top_10_candidates=top_10_before,
            top_3_candidates=top_3_after,
            buckets=metadata_buckets,
            elapsed_time=elapsed
        )

