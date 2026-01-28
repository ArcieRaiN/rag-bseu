from __future__ import annotations

"""
Context enrichment (PIPELINE 2) для пользовательских запросов.

Ответственность:
- построить embedding запроса (HashVectorizer)
- запросить LLM (Ollama) для извлечения:
  geo / metrics / years / time_granularity / oked
- применить default_enrichment, если каких‑то полей нет
- вернуть EnrichedQuery, пригодный для retrieval и reranking
"""

from typing import Any, Dict, List, Optional
import json

import numpy as np

from src.main.models import EnrichedQuery
from src.main.ollama_client import OllamaClient
from src.main.vectorizer import HashVectorizer


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

    def __init__(
        self,
        vectorizer: HashVectorizer,
        llm_client: OllamaClient,
    ):
        self._vectorizer = vectorizer
        self._llm_client = llm_client

    # -------------------- Публичный интерфейс -------------------- #

    def enrich(self, query: str) -> EnrichedQuery:
        """
        Построить embedding запроса и обогатить его структурированными полями через LLM.

        ВАЖНО:
        - если LLM отвечает невалидным JSON, мы НЕ падаем, а используем default_enrichment
          и максимально возможный парсинг.
        """
        embedded = self._vectorizer.embed(query)
        llm_raw = self._call_llm_for_enrichment(query)
        parsed = self._safe_parse_llm_response(llm_raw)

        # Применяем значения по умолчанию
        geo = self._normalize_geo(parsed.get("geo")) or DEFAULT_ENRICHMENT["geo"]
        years = self._normalize_years(parsed.get("years")) or DEFAULT_ENRICHMENT["years"]

        metrics = self._normalize_list(parsed.get("metrics"))
        # Эвристики поверх LLM: если LLM "поплыл", добиваем метрики из текста запроса.
        # Критичный кейс: запросы про студентов должны всегда иметь metric "численность студентов".
        metrics = self._merge_metrics(
            primary=self._heuristic_metrics_from_query(query),
            secondary=metrics,
            limit=5,
        )
        time_granularity = parsed.get("time_granularity") or DEFAULT_ENRICHMENT["time_granularity"]
        oked = parsed.get("oked") or DEFAULT_ENRICHMENT["oked"]

        return EnrichedQuery(
            query=query,
            embedded_query=embedded.astype(np.float32),
            geo=str(geo),
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
            "- geo: строка с географическим объектом (страна, область, город и т.п.)\n"
            "- metrics: список показателей (например, 'добыча нефти', 'численность населения')\n"
            "- years: список целых годов (если в запросе указан диапазон, разверни его в перечень)\n"
            "- time_granularity: 'year' | 'quarter' | 'month' | 'day'\n"
            "- oked: код или укрупнённое описание из ОКЭД при наличии, иначе null\n\n"
            "Верни ТОЛЬКО один JSON‑объект без пояснений.\n\n"
            f"Запрос: \"{query}\""
        )
        # Ожидаем строгий JSON-ответ → явно запрашиваем format="json"
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
        # Простейший хак: найти первую и последнюю фигурные скобки.
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        snippet = raw[start : end + 1]
        try:
            data = json.loads(snippet)
            if not isinstance(data, dict):
                return {}
            return data
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _normalize_years(value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            try:
                return [int(value)]
            except ValueError:
                return None
        if isinstance(value, list):
            years: List[int] = []
            for v in value:
                try:
                    years.append(int(v))
                except (TypeError, ValueError):
                    continue
            return years or None
        return None

    @staticmethod
    def _normalize_list(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            # допускаем разделение по запятым
            if "," in value:
                parts = [p.strip() for p in value.split(",") if p.strip()]
                return parts or None
            return [value]
        if isinstance(value, list):
            out = [str(v).strip() for v in value if str(v).strip()]
            return out or None
        return None

    @staticmethod
    def _normalize_geo(value: Any) -> Optional[str]:
        """
        Нормализует geo из LLM:
        - строка -> строка
        - список -> "A, B"
        """
        if value is None:
            return None
        if isinstance(value, str):
            v = value.strip()
            return v or None
        if isinstance(value, list):
            parts = [str(v).strip() for v in value if str(v).strip()]
            return ", ".join(parts) if parts else None
        return str(value).strip() or None

    @staticmethod
    def _heuristic_metrics_from_query(query: str) -> Optional[List[str]]:
        """
        Быстрые эвристики для метрик, чтобы не зависеть полностью от LLM.
        """
        q = (query or "").lower()
        # типичная опечатка из логов
        q = q.replace("струдент", "студент")

        metrics: List[str] = []
        if "студент" in q:
            # "сколько/число/численность студентов" -> одна целевая метрика
            metrics.append("численность студентов")

        return metrics or None

    @staticmethod
    def _merge_metrics(
        primary: Optional[List[str]],
        secondary: Optional[List[str]],
        limit: int = 5,
    ) -> Optional[List[str]]:
        out: List[str] = []
        for src in (primary or []), (secondary or []):
            for m in src:
                mm = str(m).strip().lower()
                if not mm:
                    continue
                if mm not in out:
                    out.append(mm)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        return out or None

