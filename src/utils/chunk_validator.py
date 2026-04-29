"""
Валидация и нормализация метаданных обогащённых чанков.

ChunkValidator проверяет корректность полей LLM-ответа
(search_context, metrics, years, geo) и нормализует значения:
обрезка длины search_context, фильтрация невалидных метрик, приведение years к int.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re


@dataclass
class ValidationResult:
    """Результат валидации одного чанка."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class ChunkValidator:
    """
    Валидатор и нормализатор для данных обогащения чанков.
    """

    MAX_SEARCH_CONTEXT_LENGTH = 700
    MAX_METRICS_COUNT = 8

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

        # search_context
        search_context = chunk_data.get("search_context")
        if search_context:
            if not isinstance(search_context, str):
                errors.append("search_context должен быть строкой")
            elif len(search_context) > self.MAX_SEARCH_CONTEXT_LENGTH:
                warnings.append(f"search_context обрезан до {self.MAX_SEARCH_CONTEXT_LENGTH} символов")

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

        # geo (list of strings or a single string)
        geo = chunk_data.get("geo")
        if geo is not None and not isinstance(geo, (str, list)):
            warnings.append("geo должен быть null, строкой или списком строк")

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

    def normalize_search_context(self, search_context: str) -> str:
        """Обрезает search_context до допустимой длины."""
        return search_context[:self.MAX_SEARCH_CONTEXT_LENGTH] if search_context else ""

    def normalize_metrics(self, metrics: Optional[List[str]]) -> Optional[List[str]]:
        """Обрезает до MAX_METRICS_COUNT, оставляет содержательные строки."""
        if not metrics or not isinstance(metrics, list):
            return None
        normalized = []
        for metric in metrics[:self.MAX_METRICS_COUNT]:
            if not isinstance(metric, str):
                continue
            clean = re.sub(r"\s+", " ", metric).strip(" .;:")
            if not clean:
                continue
            has_letters = any(ch.isalpha() for ch in clean)
            digit_share = sum(ch.isdigit() for ch in clean) / max(len(clean), 1)
            if has_letters and digit_share < 0.35:
                normalized.append(clean)
        return normalized if normalized else None

    def normalize_units(self, units: Optional[List[str]]) -> Optional[List[str]]:
        """Нормализует список единиц измерения."""
        if not units or not isinstance(units, list):
            return None
        normalized = []
        seen = set()
        for unit in units[:8]:
            if not isinstance(unit, str):
                continue
            clean = re.sub(r"\s+", " ", unit.lower()).strip(" .;:")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
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
