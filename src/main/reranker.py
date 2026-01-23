from __future__ import annotations

"""
LLM‑based reranking (PIPELINE 4).

Цель:
- переоценить Top‑10 кандидатов (ScoredChunk.hybrid_score)
- вернуть Top‑3 по rerank_score

ВАЖНО:
- rerank_score НЕ зависит от hybrid_score (мы не смешиваем их)
- модуль не занимается генерацией финального ответа RAG
"""

from typing import List
import json

from src.main.models import EnrichedQuery, ScoredChunk, RerankConfig
from src.main.ollama_client import OllamaClient


class LLMReranker:
    """
    Prompt‑based reranker на основе Ollama.

    Интерфейс максимально простой:
    - на вход EnrichedQuery и список ScoredChunk (Top‑10 от HybridSearcher)
    - на выход тот же список, но с заполненным полем rerank_score и отсортированный по нему
    """

    def __init__(self, llm_client: OllamaClient, config: RerankConfig):
        self._llm = llm_client
        self._config = config

    def rerank(self, enriched_query: EnrichedQuery, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        if not candidates:
            return []

        prompt = self._build_prompt(enriched_query, candidates)
        raw = self._llm.generate(prompt, temperature=self._config.temperature)
        scores_by_id = self._parse_llm_rerank_scores(raw)

        for sc in candidates:
            sc.rerank_score = float(scores_by_id.get(sc.chunk.id, 0.0))

        # Сортируем ТОЛЬКО по rerank_score
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        return candidates[: self._config.top_k]

    # -------------------- Prompt & Parsing -------------------- #

    def _build_prompt(self, enriched_query: EnrichedQuery, candidates: List[ScoredChunk]) -> str:
        """
        Подготовка промпта для LLM‑reranker‑а.

        Модель видит:
        - структурированное описание enriched_query
        - список чанков с метаданными
        и должна вернуть JSON:
        [
          {"id": "chunk_id", "score": 0.0–1.0},
          ...
        ]
        """
        query_block = {
            "query": enriched_query.query,
            "geo": enriched_query.geo,
            "metrics": enriched_query.metrics,
            "years": enriched_query.years,
            "time_granularity": enriched_query.time_granularity,
            "oked": enriched_query.oked,
        }

        chunks_block = []
        for sc in candidates:
            ch = sc.chunk
            chunks_block.append(
                {
                    "id": ch.id,
                    "context": ch.context,
                    "text": ch.text,
                    "geo": ch.geo,
                    "metrics": ch.metrics,
                    "years": ch.years,
                    "time_granularity": ch.time_granularity,
                    "oked": ch.oked,
                }
            )

        instruction = (
            "Ты работаешь как reranker для статистических документов.\n"
            "Твоя задача — ОЦЕНИТЬ, насколько каждый чанк помогает ответить на запрос.\n"
            "Учти смысл запроса, метрику, географию, период и уровень агрегации.\n"
            "Игнорируй поверхностные совпадения и шуточные/шумовые фрагменты.\n\n"
            "Верни JSON‑массив, где для каждого чанка указан score от 0 до 1.\n"
            "Формат:\n"
            "[\n"
            '  {"id": "chunk_id", "score": 0.0},\n'
            "  ...\n"
            "]\n"
            "Не добавляй комментариев и лишнего текста."
        )

        prompt = (
            f"{instruction}\n\n"
            f"Enriched query:\n{json.dumps(query_block, ensure_ascii=False, indent=2)}\n\n"
            f"Candidates:\n{json.dumps(chunks_block, ensure_ascii=False, indent=2)}\n"
        )
        return prompt

    @staticmethod
    def _parse_llm_rerank_scores(raw: str) -> dict[str, float]:
        """
        Робастный парсер JSON‑ответа reranker‑а.
        При ошибке парсинга все скоринги будут нулевыми.
        """
        if not raw:
            return {}

        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return {}

        snippet = raw[start : end + 1]
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            return {}

        if not isinstance(data, list):
            return {}

        result: dict[str, float] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            cid = item.get("id")
            score = item.get("score")
            if cid is None:
                continue
            try:
                val = float(score)
            except (TypeError, ValueError):
                continue
            # жёстко ограничиваем диапазон [0,1]
            val = max(0.0, min(1.0, val))
            result[str(cid)] = val
        return result

