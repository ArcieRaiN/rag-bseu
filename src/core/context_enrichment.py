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

from src.core.models import EnrichedQuery
from src.enrichers.client import OllamaClient
from src.vectorstore.vectorizer import SentenceVectorizer
from src.core.utils import normalize_geo, normalize_list, normalize_years

DEFAULT_ENRICHMENT: Dict[str, Any] = {
    "geo": "Республика Беларусь",
    "years": [2020, 2021, 2022, 2023, 2024],
    "metrics": None,
    "time_granularity": "year",
    "oked": None,
}


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

        # Применяем значения по умолчанию
        geo = normalize_geo(parsed.get("geo")) or DEFAULT_ENRICHMENT["geo"]
        years = normalize_years(parsed.get("years")) or DEFAULT_ENRICHMENT["years"]
        metrics = normalize_list(parsed.get("metrics")) or DEFAULT_ENRICHMENT["metrics"]
        time_granularity = parsed.get("time_granularity") or DEFAULT_ENRICHMENT["time_granularity"]
        oked = parsed.get("oked") or DEFAULT_ENRICHMENT["oked"]

        # Возвращаем EnrichedQuery с list вместо np.ndarray
        return EnrichedQuery(
            query=query,
            embedded_query=embedded.astype("float32").tolist(),
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
          "geo": "...",
          "metrics": ["...", "..."],
          "years": [2020, 2021],
          "time_granularity": "year",
          "oked": null
        }
        """
        prompt = (
            "Ты аналитик по официальной статистике Республики Беларусь.\n"
            "По пользовательскому запросу нужно аккуратно извлечь следующие поля:\n"
            "- geo: строка с географическим объектом\n"
            "- metrics: список показателей (например, 'добыча нефти')\n"
            "- years: список целых годов (если диапазон — развернуть)\n"
            "- time_granularity: 'year' | 'quarter' | 'month' | 'day'\n"
            "- oked: код или описание из ОКЭД\n\n"
            "Верни ТОЛЬКО один JSON‑объект без пояснений.\n\n"
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

    @staticmethod
    def _merge_metrics(
        primary: Optional[List[str]],
        secondary: Optional[List[str]],
        limit: int = 5,
    ) -> Optional[List[str]]:
        """
        Объединяет списки метрик, убирает дубликаты, лимитирует результат.
        """
        out: List[str] = []
        for src in (primary or []), (secondary or []):
            for m in src:
                mm = str(m).strip().lower()
                if mm and mm not in out:
                    out.append(mm)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        return out or None
