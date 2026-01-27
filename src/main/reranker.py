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
import time

from src.main.models import EnrichedQuery, ScoredChunk, RerankConfig
from src.main.ollama_client import OllamaClient
from src.logs.logger import get_logger


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
        self._logger = get_logger()

    def rerank(self, enriched_query: EnrichedQuery, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        if not candidates:
            return []

        rerank_start_time = time.time()
        
        prompt = self._build_prompt(enriched_query, candidates)
        system_prompt = self._build_system_prompt()
        
        # Логируем запрос
        ollama_config = {
            "model": getattr(getattr(self._llm, "config", None), "model", None),
            "base_url": getattr(getattr(self._llm, "config", None), "base_url", None),
            "timeout": getattr(getattr(self._llm, "config", None), "timeout", None),
            "temperature": self._config.temperature,
        }
        
        enriched_query_dict = {
            "query": enriched_query.query,
            "geo": enriched_query.geo,
            "metrics": enriched_query.metrics,
            "years": enriched_query.years,
            "time_granularity": enriched_query.time_granularity,
            "oked": enriched_query.oked,
        }
        
        self._logger.log_llm_reranking(
            event="request",
            query=enriched_query.query,
            enriched_query=enriched_query_dict,
            candidates_count=len(candidates),
            candidate_ids=[sc.chunk.id for sc in candidates],
            system_prompt=system_prompt,
            prompt=prompt,
            ollama_config=ollama_config,
        )
        
        raw = self._llm.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=self._config.temperature
        )
        
        scores_by_id = self._parse_llm_rerank_scores(raw)

        for sc in candidates:
            sc.rerank_score = float(scores_by_id.get(sc.chunk.id, 0.0))

        # Сортируем ТОЛЬКО по rerank_score
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        top_chunks = candidates[: self._config.top_k]
        
        # Логируем ответ
        elapsed_time = time.time() - rerank_start_time
        self._logger.log_llm_reranking(
            event="response",
            query=enriched_query.query,
            enriched_query=enriched_query_dict,
            candidates_count=len(candidates),
            candidate_ids=[sc.chunk.id for sc in candidates],
            raw_response=raw,
            rerank_scores=scores_by_id,
            top_k=len(top_chunks),
            elapsed_time=elapsed_time,
        )
        
        return top_chunks

    # -------------------- Prompt & Parsing -------------------- #

    def _build_system_prompt(self) -> str:
        """
        Системный промпт для reranker.
        """
        return (
            "Ты — эксперт по оценке релевантности документов для статистических запросов. "
            "Твоя задача — точно оценить, насколько каждый чанк документа помогает ответить на запрос пользователя. "
            "STRICT RULES:- Output ONLY a valid JSON array of exactly 5 objects - No text before or after JSON - No markdown, comments, explanations - chunk_id must exactly match input"
        )

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
                    "text": ch.text[:500] if ch.text else "",  # Ограничиваем длину для ускорения
                    "geo": ch.geo,
                    "metrics": ch.metrics,
                    "years": ch.years,
                    "time_granularity": ch.time_granularity,
                    "oked": ch.oked,
                }
            )

        instruction = (
            "Ты работаешь как reranker для статистических документов Республики Беларусь.\n"
            "Твоя задача — ОЦЕНИТЬ, насколько каждый чанк помогает ответить на запрос пользователя.\n\n"
            "КРИТИЧЕСКИ ВАЖНО при оценке:\n"
            "1. МЕТРИКА: Чанк должен содержать данные по запрашиваемой метрике (например, 'удой молока', "
            "'инвестиции в основной капитал'). Игнорируй поверхностные совпадения слов — важна смысловая релевантность.\n"
            "2. ГЕОГРАФИЯ: Если в запросе указан регион/город/область, чанк должен содержать данные именно по этому региону. "
            "Если география не указана, приоритет отдавай чанкам с общереспубликанскими данными.\n"
            "3. ПЕРИОД (years): Чанк должен содержать данные за запрашиваемые годы. "
            "Если годы не указаны, приоритет отдавай чанкам с актуальными данными (2023-2024).\n"
            "4. УРОВЕНЬ АГРЕГАЦИИ (time_granularity): Если указан уровень (year/quarter/month/day), "
            "чанк должен соответствовать этому уровню. Если не указан, приоритет отдавай годовым данным.\n"
            "5. ОКЭД: Если указан код ОКЭД, чанк должен относиться к соответствующей отрасли.\n\n"
            "ИГНОРИРУЙ:\n"
            "- Поверхностные совпадения слов без смысловой связи\n"
            "- Шуточные/шумовые фрагменты (оглавления, титульные страницы, контакты)\n"
            "- Чанки, которые не содержат фактических данных по запросу\n\n"
            "Верни JSON‑массив, где для каждого чанка указан score от 0.0 до 1.0.\n"
            "Score 1.0 = идеальное соответствие всем критериям\n"
            "Score 0.5-0.9 = хорошее соответствие, но есть незначительные несоответствия\n"
            "Score 0.1-0.4 = слабое соответствие, только частично релевантен\n"
            "Score 0.0 = не релевантен запросу\n\n"
            "Формат ответа:\n"
            "[\n"
            '  {"id": "chunk_id", "score": 0.0},\n'
            "  ...\n"
            "]\n\n"
            "В массиве должно быть РОВНО столько объектов, сколько кандидатов в списке ниже. "
            "Каждый объект должен содержать id из списка кандидатов."
        )

        prompt = (
            f"{instruction}\n\n"
            f"Запрос пользователя:\n{json.dumps(query_block, ensure_ascii=False, indent=2)}\n\n"
            f"Кандидаты для ранжирования ({len(chunks_block)} чанков):\n"
            f"{json.dumps(chunks_block, ensure_ascii=False, indent=2)}\n"
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

