from __future__ import annotations

"""
Детерминированная постобработка обогащённых чанков.

Фильтрация невалидных метрик и сборка единого search_context
из заголовков, показателей и структурированных метаданных.
Никаких LLM-вызовов — только правила и словари.
"""

from typing import List, Optional
import re

from src.core.models import Chunk


class EnrichmentPostProcessor:
    """
    Минимальный post-processor для нормализации и усиления семантики чанков.
    """

    # 1. Паттерны для определения "не метрик"
    NON_METRIC_PATTERNS = [
        re.compile(r"^.{120,}$", re.IGNORECASE),
        re.compile(r"^[\d\s,.;:%–—-]+$", re.IGNORECASE),
    ]

    # 2. Словарь эвристического обогащения контекста
    # ключ — слово, которое добавляем
    # значение — список триггеров
    HEURISTIC_CONTEXT_MAP = {
        "образование": ["школ", "обучен", "учащ", "студент"],
        "здравоохранение": ["медицин", "больниц", "здравоох"],
        "цифровизация": ["цифров", "информацион", "it", "икт"],
        "работа": ["занятост", "работ", "безработ"],
        "население": ["населен", "демограф"],
    }

    def process_chunk(self, chunk: Chunk) -> Chunk:
        """
        Обрабатывает один чанк, исправляя типичные ошибки.

        Args:
            chunk: Чанк для обработки

        Returns:
            Обработанный чанк
        """

        # 1. Фильтрация metrics
        if chunk.metrics:
            chunk.metrics = self._filter_valid_metrics(chunk.metrics)

        # 2. Единственный поисковый контекст используется для FAISS/BM25.
        chunk.search_context = self._build_search_context(chunk)

        return chunk

    # ------------------------------------------------------------------
    # ЭТАП 1. Metrics
    # ------------------------------------------------------------------

    def _filter_valid_metrics(self, metrics: List[str]) -> Optional[List[str]]:
        """
        Оставляет только валидные метрики.

        Правило:
        - длина метрики < 20 символов
        """
        if not metrics:
            return None

        valid: List[str] = []

        for metric in metrics:
            m = metric.strip()
            if not m:
                continue

            is_invalid = any(p.search(m) for p in self.NON_METRIC_PATTERNS)
            if is_invalid:
                continue

            valid.append(m)

        return valid or None

    # ------------------------------------------------------------------
    # ЭТАП 2. Эвристические тематические подсказки
    # ------------------------------------------------------------------

    def _extract_heuristic_terms(
        self,
        text: Optional[str],
    ) -> List[str]:
        """Возвращает тематические подсказки на основе словаря эвристик."""
        base = (text or "").lower()

        additions: List[str] = []

        for word, triggers in self.HEURISTIC_CONTEXT_MAP.items():
            for t in triggers:
                if t in base:
                    additions.append(word)
                    break

        return sorted(set(additions))

    def _build_search_context(self, chunk: Chunk) -> str:
        parts: List[str] = []
        if chunk.search_context:
            parts.append(chunk.search_context)
        if chunk.section and chunk.section.lower() not in " ".join(parts).lower():
            parts.append(f"раздел: {chunk.section}")
        heuristic_terms = self._extract_heuristic_terms(chunk.text)
        if heuristic_terms:
            parts.append("темы: " + ", ".join(heuristic_terms))
        if chunk.metrics:
            parts.append("показатели: " + ", ".join(chunk.metrics[:6]))
        if chunk.units:
            parts.append("единицы измерения: " + ", ".join(chunk.units[:4]))
        if chunk.geo:
            geo_values = chunk.geo if isinstance(chunk.geo, list) else [chunk.geo]
            parts.append("география: " + ", ".join(str(g) for g in geo_values[:8]))
        if chunk.years:
            years_sorted = sorted(set(chunk.years))
            years_repr = (
                f"{years_sorted[0]}-{years_sorted[-1]}"
                if len(years_sorted) > 6
                else ", ".join(str(y) for y in years_sorted)
            )
            parts.append(f"годы: {years_repr}")
        return self._clean_search_context(" | ".join(parts), max_length=700)

    # ------------------------------------------------------------------
    # ЭТАП 3. Очистка search_context
    # ------------------------------------------------------------------

    def _clean_search_context(self, search_context: str, *, max_length: int = 700) -> str:
        """
        Минимальная очистка:
        - замена \n на пробел
        - ограничение длины
        """
        cleaned = search_context.replace("\n", " ").strip()

        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]

        return cleaned
