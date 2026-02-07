from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class ChunkValidator:
    """
    Валидатор и нормализатор для данных обогащения чанков.
    """

    MAX_CONTEXT_LENGTH = 256
    MAX_METRICS_COUNT = 5

    def __init__(self):
        self._seen_chunk_ids: Set[str] = set()

    # -------------------- Валидация -------------------- #

    def validate_chunk(
        self,
        chunk_data: Dict[str, Any],
        check_uniqueness: bool = True,
    ) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        # chunk_id
        chunk_id = chunk_data.get("chunk_id")
        if not chunk_id:
            errors.append("chunk_id отсутствует")
        elif check_uniqueness:
            if chunk_id in self._seen_chunk_ids:
                errors.append(f"chunk_id '{chunk_id}' дублируется")
            else:
                self._seen_chunk_ids.add(chunk_id)

        # context
        context = chunk_data.get("context")
        if context:
            if not isinstance(context, str):
                errors.append("context должен быть строкой")
            elif len(context) > self.MAX_CONTEXT_LENGTH:
                warnings.append(f"context обрезан до {self.MAX_CONTEXT_LENGTH} символов")

        # metrics
        metrics = chunk_data.get("metrics")
        if metrics is not None:
            if not isinstance(metrics, list):
                errors.append("metrics должен быть списком или null")
            else:
                if len(metrics) > self.MAX_METRICS_COUNT:
                    warnings.append(f"metrics превышает {self.MAX_METRICS_COUNT} элементов")
                for i, metric in enumerate(metrics):
                    if not isinstance(metric, str):
                        errors.append(f"metrics[{i}] должен быть строкой")
                    else:
                        if not any('\u0400' <= ch <= '\u04FF' for ch in metric):
                            warnings.append(f"metrics[{i}] '{metric}' не содержит кириллицу")

        # years
        years = chunk_data.get("years")
        if years is not None:
            if not isinstance(years, list):
                errors.append("years должен быть списком или null")
            else:
                for i, year in enumerate(years):
                    if not isinstance(year, int):
                        try:
                            int(year)
                        except (ValueError, TypeError):
                            errors.append(f"years[{i}] должен быть целым числом")

        # geo
        geo = chunk_data.get("geo")
        if geo is not None and not isinstance(geo, str):
            warnings.append("geo должен быть строкой или null")

        # time_granularity
        tg = chunk_data.get("time_granularity")
        if tg is not None:
            valid_tg = {"year", "quarter", "month", "day"}
            if tg not in valid_tg:
                warnings.append(f"time_granularity '{tg}' не в {valid_tg}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_batch(
        self,
        chunks_data: List[Dict[str, Any]],
        expected_chunk_ids: Optional[List[str]] = None,
        check_uniqueness: bool = True,
    ) -> tuple[List[ValidationResult], bool]:
        results = []
        if not isinstance(chunks_data, list):
            return [ValidationResult(False, ["Ответ должен быть массивом"], [])], False

        for i, chunk in enumerate(chunks_data):
            results.append(self.validate_chunk(chunk, check_uniqueness))
        is_valid = all(r.is_valid for r in results)
        return results, is_valid

    # -------------------- Нормализация для LLMEnricher -------------------- #

    def normalize_context(self, context: str) -> str:
        """Обрезает context до 256 символов"""
        return context[:self.MAX_CONTEXT_LENGTH] if context else ""

    def normalize_metrics(self, metrics: Optional[List[str]]) -> Optional[List[str]]:
        """Обрезает до MAX_METRICS_COUNT, оставляет только русские строки"""
        if not metrics or not isinstance(metrics, list):
            return None
        normalized = []
        for metric in metrics[:self.MAX_METRICS_COUNT]:
            if isinstance(metric, str) and any('\u0400' <= ch <= '\u04FF' for ch in metric):
                normalized.append(metric.strip())
        return normalized if normalized else None

    def normalize_years(self, years: Optional[List[Any]]) -> Optional[List[int]]:
        """Фильтрует только целые числа, без ограничения по количеству"""
        if not years or not isinstance(years, list):
            return None
        normalized = []
        for y in years:
            if isinstance(y, int):
                normalized.append(y)
            else:
                try:
                    normalized.append(int(y))
                except (ValueError, TypeError):
                    continue
        return normalized if normalized else None

    def reset(self) -> None:
        """Сбрасывает состояние валидатора"""
        self._seen_chunk_ids.clear()
