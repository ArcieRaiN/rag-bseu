from __future__ import annotations

"""
Модуль для post-processing обогащенных данных.

Цель:
- минималистичный, детерминированный пайплайн
- никаких inference / LLM-подобных эвристик
- только жёстко контролируемые правила
"""

from typing import List, Optional
import re

from src.core.models import Chunk


class EnrichmentPostProcessor:
    """
    Минимальный post-processor для нормализации и усиления семантики чанков.
    """

    # 1. Паттерны для определения "не метрик"
    # Пока ровно одна проверка: длина метрики
    NON_METRIC_PATTERNS = [
        re.compile(r"^.{20,}$", re.IGNORECASE),  # длина строки >= 20
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

        # 2. Эвристическое обогащение context
        chunk.context = self._enrich_context_with_heuristics(
            context=chunk.context,
            text=chunk.text,
        )

        # 3. Усиление context метаданными
        chunk.context = self._enrich_context_with_metadata(chunk)

        # 4. Очистка context
        if chunk.context:
            chunk.context = self._clean_context(chunk.context)

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

            valid.append(m.lower())

        return valid or None

    # ------------------------------------------------------------------
    # ЭТАП 2. Эвристическое обогащение context
    # ------------------------------------------------------------------

    def _enrich_context_with_heuristics(
        self,
        context: Optional[str],
        text: Optional[str],
    ) -> Optional[str]:
        """
        Добавляет слова в context на основе словаря эвристик.
        """
        base = f"{context or ''} {text or ''}".lower()

        additions: List[str] = []

        for word, triggers in self.HEURISTIC_CONTEXT_MAP.items():
            for t in triggers:
                if t in base:
                    additions.append(word)
                    break

        if not additions:
            return context

        additions_str = ", ".join(sorted(set(additions)))

        if context:
            return f"{context} | {additions_str}"
        return additions_str

    # ------------------------------------------------------------------
    # ЭТАП 3. Усиление метаданными
    # ------------------------------------------------------------------

    def _enrich_context_with_metadata(self, chunk: Chunk) -> Optional[str]:
        """
        Добавляет в context:
        - показатели
        - географию
        - годы
        """
        parts: List[str] = []

        if chunk.metrics:
            parts.append(f"показатели: {', '.join(chunk.metrics)}")

        if chunk.geo:
            parts.append(f"география: {chunk.geo}")

        if chunk.years:
            years_sorted = sorted(chunk.years)
            if len(years_sorted) > 1:
                years_repr = f"{years_sorted[0]}-{years_sorted[-1]}"
            else:
                years_repr = str(years_sorted[0])
            parts.append(f"годы: {years_repr}")

        if not parts:
            return chunk.context

        meta = "; ".join(parts)

        if chunk.context:
            return f"{chunk.context} | {meta}"
        return meta

    # ------------------------------------------------------------------
    # ЭТАП 4. Очистка context
    # ------------------------------------------------------------------

    def _clean_context(self, context: str) -> str:
        """
        Минимальная очистка:
        - замена \n на пробел
        - ограничение длины
        """
        cleaned = context.replace("\n", " ").strip()

        if len(cleaned) > 256:
            cleaned = cleaned[:256]

        return cleaned
