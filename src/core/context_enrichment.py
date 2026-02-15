from __future__ import annotations
"""
Context enrichment (PIPELINE 2) для пользовательских запросов.

Ответственность:
- построить embedding запроса (SentenceVectorizer)
- запросить LLM (Ollama) для извлечения:
  geo / metrics / years / time_granularity / oked
- применить default_enrichment, если каких‑то полей нет
- вернуть EnrichedQuery, пригодный для retrieval и reranking
"""

from typing import Any, Dict, List, Optional
import json

from src.core.models import EnrichedQuery
from src.enrichers.client import OllamaClient
from src.vectorstore.vectorizer import SentenceVectorizer

DEFAULT_ENRICHMENT: Dict[str, Any] = {
    "geo": [],
    "metrics": [],
    "years": [],
    "time_granularity": None,
    "oked": None,
}


def _fallback(value, default):
    """Возвращает default только если value is None, иначе оставляет value."""
    return default if value is None else value


class QueryContextEnricher:
    """
    Класс‑фасад над LLM для обогащения пользовательских запросов.

    Вынесен в отдельный модуль, чтобы:
    - не смешивать работу с LLM и retrieval
    - обеспечить простое мокирование в тестах
    """

    def __init__(self, vectorizer: SentenceVectorizer, llm_client: OllamaClient):
        self._vectorizer = vectorizer
        self._llm_client = llm_client

    # -------------------- Публичный интерфейс -------------------- #

    def enrich(self, query: str) -> EnrichedQuery:
        """
        Построить embedding запроса и обогатить его структурированными полями через LLM.

        ВАЖНО:
        - если LLM отвечает невалидным JSON, мы НЕ падаем, а используем default_enrichment
        """
        # embedding запроса
        embedded = self._vectorizer.embed(query)

        # LLM enrichment
        llm_raw = self._call_llm_for_enrichment(query)
        parsed = self._safe_parse_llm_response(llm_raw)

        # Применяем значения по умолчанию только если None
        geo = _fallback(parsed.get("geo"), DEFAULT_ENRICHMENT["geo"])
        metrics = _fallback(parsed.get("metrics"), DEFAULT_ENRICHMENT["metrics"])
        years = _fallback(parsed.get("years"), DEFAULT_ENRICHMENT["years"])
        time_granularity = _fallback(parsed.get("time_granularity"), DEFAULT_ENRICHMENT["time_granularity"])
        oked = _fallback(parsed.get("oked"), DEFAULT_ENRICHMENT["oked"])

        # Возвращаем EnrichedQuery с list вместо np.ndarray
        return EnrichedQuery(
            query=query,
            embedded_query=embedded.astype("float32"),
            geo=geo,
            years=years,
            metrics=metrics,
            time_granularity=time_granularity,
            oked=oked,
            raw_llm_response=llm_raw,
        )

    # -------------------- Взаимодействие с LLM -------------------- #

    def _call_llm_for_enrichment(self, query: str) -> str:
        """
        Один LLM‑запрос для извлечения структурированных полей из пользовательского запроса.

        Ожидаемый формат ответа — JSON с ключами:
        {
          "geo": ["..."],
          "metrics": ["...", "..."],
          "years": [2020, 2021],
          "time_granularity": "year",
          "oked": null
        }
        """
        prompt = (
            "Ты аналитик по официальной статистике Республики Беларусь.\n"
            "По пользовательскому запросу нужно извлечь СТРУКТУРУ запроса "
            "для последующего семантического поиска.\n\n"
            "Верни ТОЛЬКО один валидный JSON с полями:\n"
            "- geo: список территорий (если не указаны — [])\n"
            "- metrics: список названий показателей, без чисел, лет и единиц измерения\n"
            "- years: список целых лет (если указан диапазон — разверни)\n"
            "- time_granularity: 'year', 'quarter', 'month' или null\n"
            "- oked: строка (код ОКЭД) или null\n\n"

            "Правила:\n"
            "1. Используй только информацию из запроса.\n"
            "2. Ничего не додумывай.\n"
            "3. Если поле невозможно определить — верни [] или null.\n"
            "4. Ответ строго JSON, без комментариев.\n\n"

            f"Запрос: \"{query}\""
        )
        return self._llm_client.generate(prompt, format="json")

    # -------------------- Вспомогательные методы -------------------- #

    @staticmethod
    def _safe_parse_llm_response(raw: str) -> Dict[str, Any]:
        """
        Пытается распарсить JSON из LLM‑ответа.
        При ошибке возвращает пустой dict, чтобы сработали дефолты.
        """
        if not raw:
            return {}
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        snippet = raw[start : end + 1]
        try:
            data = json.loads(snippet)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
