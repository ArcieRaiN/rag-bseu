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

from typing import List, Optional
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        system_prompt = self._build_system_prompt()

        # Сохраняем исходный порядок кандидатов (по hybrid_score)
        original_order = {sc.chunk.id: i for i, sc in enumerate(candidates)}

        # Общий снэпшот обогащённого запроса (для логов)
        enriched_query_dict = {
            "query": enriched_query.query,
            "geo": enriched_query.geo,
            "metrics": enriched_query.metrics,
            "years": enriched_query.years,
            "time_granularity": enriched_query.time_granularity,
            "oked": enriched_query.oked,
        }

        # Параллельная оценка чанков (per‑chunk scoring)
        max_workers = max(1, min(self._config.max_workers, len(candidates)))
        metadata_buckets: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self._score_single_chunk,
                    enriched_query,
                    sc,
                    system_prompt,
                    enriched_query_dict,
                ): sc
                for sc in candidates
            }

            for future in as_completed(future_to_chunk):
                sc = future_to_chunk[future]
                try:
                    score = future.result()
                except Exception as e:
                    # На всякий случай логируем и считаем score=0.0
                    self._logger.log_llm_reranking(
                        event="error",
                        query=enriched_query.query,
                        candidates_count=1,
                        candidate_ids=[sc.chunk.id],
                        error=str(e),
                    )
                    score = 0.0

                # Применяем жёсткие priors по метаданным и классифицируем bucket
                adjusted_score, bucket = self._apply_metadata_caps(
                    enriched_query, sc, float(score)
                )
                sc.rerank_score = adjusted_score
                metadata_buckets[sc.chunk.id] = bucket

        # Решаем, использовать ли rerank или оставить hybrid‑порядок
        max_score = max((sc.rerank_score for sc in candidates), default=0.0)
        if max_score < self._config.min_relevance_score:
            # Reranker «ничего не нашёл» — сохраняем hybrid‑порядок
            candidates_sorted = sorted(
                candidates, key=lambda sc: original_order.get(sc.chunk.id, 1_000_000)
            )
        else:
            # Используем только rerank_score
            candidates_sorted = sorted(
                candidates, key=lambda sc: sc.rerank_score, reverse=True
            )

        top_chunks = candidates_sorted[: self._config.top_k]

        # Логируем финальный результат
        elapsed_time = time.time() - rerank_start_time
        self._logger.log_llm_reranking(
            event="final",
            query=enriched_query.query,
            enriched_query=enriched_query_dict,
            candidates_count=len(candidates),
            candidate_ids=[sc.chunk.id for sc in candidates_sorted],
            rerank_scores={sc.chunk.id: sc.rerank_score for sc in candidates_sorted},
            buckets=metadata_buckets,
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
            "Ты оцениваешь, помогает ли ОДИН фрагмент документа ответить на статистический запрос.\n"
            "Отвечай ТОЛЬКО одним JSON-объектом вида {\"id\": \"...\", \"score\": 0.0-1.0}.\n"
            "Не пиши ничего кроме JSON. Не используй markdown, комментарии и пояснения."
        )

    def _build_prompt_for_chunk(
        self,
        enriched_query: EnrichedQuery,
        scored_chunk: ScoredChunk,
    ) -> str:
        """
        Короткий промпт для оценки ОДНОГО чанка.
        """
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
            "Используй следующую шкалу:\n"
            "- 0.0: фрагмент не относится к запросу.\n"
            "- около 0.3: тема рядом, но не та метрика или нет конкретных данных.\n"
            "- около 0.6: нужная метрика есть, но не те годы/география/уровень агрегации.\n"
            "- около 0.9: фрагмент содержит именно те данные, которые нужны запросу.\n"
            "Если фрагмент частично полезен, используй промежуточные значения между этими примерами.\n"
            "Верни один JSON-объект с полями \"id\" и \"score\" (0.0–1.0)."
        )

        prompt = (
            f"{instruction}\n\n"
            f"Запрос:\n{json.dumps(query_block, ensure_ascii=False, indent=2)}\n\n"
            f"Фрагмент:\n{json.dumps(chunk_block, ensure_ascii=False, indent=2)}\n"
        )
        return prompt

    def _score_single_chunk(
        self,
        enriched_query: EnrichedQuery,
        scored_chunk: ScoredChunk,
        system_prompt: str,
        enriched_query_dict: dict,
    ) -> float:
        """
        Оценка одного чанка через LLM. Возвращает score в диапазоне [0,1].
        При любой ошибке/невалидном ответе возвращает 0.0.
        """
        ch_id = scored_chunk.chunk.id

        ollama_config = {
            # Для reranking всегда используем специализированную модель из конфигурации
            "model": self._config.model_name,
            "base_url": getattr(getattr(self._llm, "config", None), "base_url", None),
            "timeout": getattr(getattr(self._llm, "config", None), "timeout", None),
            "temperature": self._config.temperature,
        }

        last_raw = ""
        for attempt in range(1, self._config.max_retries + 1):
            prompt = self._build_prompt_for_chunk(enriched_query, scored_chunk)

            # Логируем запрос
            self._logger.log_llm_reranking(
                event="request",
                query=enriched_query.query,
                enriched_query=enriched_query_dict,
                candidates_count=1,
                candidate_ids=[ch_id],
                system_prompt=system_prompt,
                prompt=prompt,
                ollama_config=ollama_config,
                attempt=attempt,
                prompt_type="per_chunk",
            )

            raw = self._llm.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=self._config.temperature,
                 # Переопределяем модель на специализированный reranker
                 model=self._config.model_name,
                format="json",
            )
            last_raw = raw

            score = self._parse_single_score(raw, ch_id)
            if score is not None:
                # Логируем успешный ответ
                self._logger.log_llm_reranking(
                    event="response",
                    query=enriched_query.query,
                    enriched_query=enriched_query_dict,
                    candidates_count=1,
                    candidate_ids=[ch_id],
                    raw_response=raw,
                    rerank_scores={ch_id: score},
                    top_k=1,
                )
                return score

        # Логируем неудачный парсинг после всех попыток
        self._logger.log_llm_reranking(
            event="response_parse_failed",
            query=enriched_query.query,
            enriched_query=enriched_query_dict,
            candidates_count=1,
            candidate_ids=[ch_id],
            raw_response=last_raw,
        )
        return 0.0

    @staticmethod
    def _parse_single_score(raw: str, expected_id: str) -> float | None:
        """
        Парсит JSON-объект {id, score} из ответа LLM.
        Возвращает float в [0,1] или None при неуспехе.
        """
        if not raw:
            return None

        text = raw.strip()

        # Удаляем возможные ```json ... ``` обёртки
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        # Пытаемся найти объект в тексте
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        snippet = text[start : end + 1]
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        cid = str(data.get("id") or expected_id)
        if cid != expected_id:
            # Модель могла не вернуть id, но это не критично — подставляем expected_id
            cid = expected_id

        score = data.get("score")
        try:
            val = float(score)
        except (TypeError, ValueError):
            return None

        # Жёстко ограничиваем диапазон [0,1]
        val = max(0.0, min(1.0, val))
        return val

    # -------------------- Метаданные как жёсткие priors -------------------- #

    def _apply_metadata_caps(
        self,
        enriched_query: EnrichedQuery,
        scored_chunk: ScoredChunk,
        raw_score: float,
    ) -> tuple[float, str]:
        """
        Применяет жёсткие ограничения по метаданным (metrics/geo/years)
        и возвращает (скор после ограничений, bucket‑метку для логов).

        Bucket может быть:
        - "no_metric"      — метрика не совпадает с запросом
        - "geo_mismatch"   — география противоречит запросу
        - "missing_years"  — в запросе есть годы, а в чанке нет
        - "partial"        — частично полезный фрагмент
        - "good"           — хороший кандидат
        """
        bucket = "good"
        score = float(max(0.0, min(1.0, raw_score)))

        ch = scored_chunk.chunk

        # --- Метрика как строгий фильтр ---
        if enriched_query.metrics:
            q_metrics = {m.lower() for m in enriched_query.metrics if isinstance(m, str)}
            ch_metrics = {
                m.lower() for m in (ch.metrics or []) if isinstance(m, str)
            }
            if not q_metrics.intersection(ch_metrics):
                # Совсем другая тема — не выше 0.3
                score = min(score, 0.3)
                bucket = "no_metric"

        # --- География: если явно не совпадает ---
        if enriched_query.geo and ch.geo:
            q_geo = enriched_query.geo.lower()
            c_geo = ch.geo.lower()
            if q_geo not in c_geo and c_geo not in q_geo:
                score = min(score, 0.3)
                # Не затираем более строгий bucket "no_metric"
                if bucket == "good":
                    bucket = "geo_mismatch"

        # --- Годы: если в запросе есть, а в чанке нет ---
        if enriched_query.years and not ch.years:
            score = min(score, 0.5)
            if bucket == "good":
                bucket = "missing_years"

        # Финальная классификация bucket по итоговому score, если он не был переопределён
        if bucket == "good":
            if score < 0.05:
                bucket = "no_metric"
            elif score < 0.4:
                bucket = "partial"
            else:
                bucket = "good"

        return score, bucket


class CrossEncoderReranker:
    """
    Cross-encoder reranker (HF sentence-transformers CrossEncoder).

    На вход:
    - EnrichedQuery
    - список ScoredChunk (Top-K от HybridSearcher)

    На выход:
    - Top-N по cross-encoder score (с сохранением metadata caps)
    """

    def __init__(self, config: RerankConfig):
        self._config = config
        self._logger = get_logger()
        self._model = None  # lazy init
        self._device = None  # lazy init

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            # device можно переопределить через env (cpu/cuda)
            import os

            device = os.getenv("RAG_RERANK_DEVICE")
            self._device = device
            # Важно: не полагаться на predict(), т.к. он может применять activation/softmax.
            # Мы будем извлекать raw logits напрямую через model.model(**inputs).logits.
            if device:
                self._model = CrossEncoder(self._config.model_name, device=device, max_length=512)
            else:
                self._model = CrossEncoder(self._config.model_name, max_length=512)
        return self._model

    @staticmethod
    def _sigmoid(x: float) -> float:
        # стабильная сигмоида для логитов умеренной величины
        import math

        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _predict_raw_logits(self, pairs: List[list[str]]) -> List[float]:
        """
        Возвращает raw logits cross-encoder'а без softmax/sigmoid.

        Почему не model.predict():
        sentence-transformers может применять activation (sigmoid) и/или softmax,
        что схлопывает шкалу и ухудшает разделение релевантности.
        """
        model = self._get_model()
        # CrossEncoder хранит HF модель/токенизатор
        hf_model = getattr(model, "model", None)
        tokenizer = getattr(model, "tokenizer", None)
        if hf_model is None or tokenizer is None:
            # fallback: используем predict как есть (лучше, чем упасть)
            scores = model.predict(pairs)
            return [float(s) for s in scores]

        import torch

        device = getattr(hf_model, "device", None) or torch.device("cpu")
        encoded = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            out = hf_model(**encoded)
            logits = out.logits

        # logits shape:
        # - (batch, 1) for regression / single-label classification
        # - (batch, 2) for binary classification (softmax)
        if logits.dim() == 2 and logits.size(-1) == 1:
            vals = logits.squeeze(-1)
        elif logits.dim() == 2 and logits.size(-1) >= 2:
            # Берём “positive class” логит (обычно индекс 1).
            vals = logits[:, 1]
        else:
            vals = logits.view(-1)

        return [float(x) for x in vals.detach().cpu().tolist()]

    def _apply_metadata_penalties_logit(
        self,
        enriched_query: EnrichedQuery,
        scored_chunk: ScoredChunk,
        raw_logit: float,
    ) -> tuple[float, str]:
        """
        Жёсткие priors по метаданным, но в logit-space (штрафы), без обрезки шкалы.

        Важно: НЕ clamp в [0,1] — иначе убиваем смысл raw logits.
        """
        score = float(raw_logit)
        bucket = "good"

        ch = scored_chunk.chunk

        # --- Метрика как строгий фильтр ---
        if enriched_query.metrics:
            q_metrics = {m.lower() for m in enriched_query.metrics if isinstance(m, str)}
            ch_metrics = {m.lower() for m in (ch.metrics or []) if isinstance(m, str)}
            if not q_metrics.intersection(ch_metrics):
                # сильный штраф (переводит в “нерелевант” почти при любом базовом logit)
                score -= 5.0
                bucket = "no_metric"

        # --- География: если явно не совпадает ---
        if enriched_query.geo and ch.geo:
            q_geo = enriched_query.geo.lower()
            c_geo = ch.geo.lower()
            if q_geo not in c_geo and c_geo not in q_geo:
                score -= 3.0
                if bucket == "good":
                    bucket = "geo_mismatch"

        # --- Годы: если в запросе есть, а в чанке нет ---
        if enriched_query.years and not ch.years:
            score -= 1.5
            if bucket == "good":
                bucket = "missing_years"

        if bucket == "good":
            bucket = "good"

        return score, bucket

    def rerank(self, enriched_query: EnrichedQuery, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        if not candidates:
            return []

        rerank_start_time = time.time()

        # Сохраняем исходный порядок кандидатов (по hybrid_score)
        original_order = {sc.chunk.id: i for i, sc in enumerate(candidates)}

        enriched_query_dict = {
            "query": enriched_query.query,
            "geo": enriched_query.geo,
            "metrics": enriched_query.metrics,
            "years": enriched_query.years,
            "time_granularity": enriched_query.time_granularity,
            "oked": enriched_query.oked,
        }

        # Собираем пары (query, passage)
        pairs: List[list[str]] = []
        for sc in candidates:
            ch = sc.chunk
            # Для cross-encoder лучше давать компактный passage:
            # context + первые N символов текста
            passage = (ch.context or "").strip()
            raw = (ch.text or "").strip()
            if raw:
                passage = f"{passage}\n\n{raw[:800]}" if passage else raw[:800]
            pairs.append([enriched_query.query, passage])

        # Получаем raw logits без activation/softmax
        scores = self._predict_raw_logits(pairs)

        metadata_buckets: dict[str, str] = {}
        for sc, score in zip(candidates, scores):
            adjusted_logit, bucket = self._apply_metadata_penalties_logit(enriched_query, sc, float(score))
            sc.rerank_score = adjusted_logit
            metadata_buckets[sc.chunk.id] = bucket

        # Порог min_relevance_score оставляем в интерпретации [0..1] (probability),
        # поэтому сравниваем через sigmoid(max_logit). Это монотонно и не ломает ранжирование.
        max_logit = max((sc.rerank_score for sc in candidates), default=float("-inf"))
        max_prob = self._sigmoid(max_logit) if max_logit != float("-inf") else 0.0
        if max_prob < self._config.min_relevance_score:
            candidates_sorted = sorted(
                candidates, key=lambda sc: original_order.get(sc.chunk.id, 1_000_000)
            )
        else:
            candidates_sorted = sorted(candidates, key=lambda sc: sc.rerank_score, reverse=True)

        top_chunks = candidates_sorted[: self._config.top_k]

        elapsed_time = time.time() - rerank_start_time
        # Логируем через существующий logger (без prompt/system_prompt)
        # Важно: rerank_scores тут — raw logits (после мета-штрафов), чтобы видеть реальный разброс.
        self._logger.log_llm_reranking(
            event="final_cross_encoder",
            query=enriched_query.query,
            enriched_query=enriched_query_dict,
            candidates_count=len(candidates),
            candidate_ids=[sc.chunk.id for sc in candidates_sorted],
            rerank_scores={sc.chunk.id: sc.rerank_score for sc in candidates_sorted},
            buckets=metadata_buckets,
            top_k=len(top_chunks),
            elapsed_time=elapsed_time,
            extra={
                "rerank_logit_min": float(min((sc.rerank_score for sc in candidates), default=0.0)),
                "rerank_logit_max": float(max((sc.rerank_score for sc in candidates), default=0.0)),
                "rerank_max_prob": float(max_prob),
            },
        )

        return top_chunks

