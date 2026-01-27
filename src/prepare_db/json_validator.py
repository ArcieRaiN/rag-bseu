from __future__ import annotations

"""
Модуль для валидации данных обогащения чанков.

Проверяет:
- Уникальность chunk_id
- Ограничения на длину context
- Валидность metrics (русские строки, максимум 5)
- Валидность years (целые числа, максимум 5)
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Результат валидации чанка."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ChunkValidator:
    """
    Валидатор для данных обогащения чанков.
    
    Правила валидации:
    - chunk_id: обязателен, должен быть уникальным
    - context: максимум 256 символов
    - metrics: максимум 5 элементов, только русские строки
    - years: максимум 5 элементов, только целые числа
    """

    MAX_CONTEXT_LENGTH = 256
    MAX_METRICS_COUNT = 5
    MAX_YEARS_COUNT = 5

    def __init__(self):
        """Инициализация валидатора."""
        self._seen_chunk_ids: Set[str] = set()

    def validate_chunk(
        self,
        chunk_data: Dict[str, Any],
        check_uniqueness: bool = True,
    ) -> ValidationResult:
        """
        Валидирует данные одного чанка.
        
        Args:
            chunk_data: Словарь с данными чанка
            check_uniqueness: Проверять ли уникальность chunk_id
            
        Returns:
            ValidationResult с результатами валидации
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Проверка chunk_id
        chunk_id = chunk_data.get("chunk_id")
        if not chunk_id:
            errors.append("chunk_id отсутствует")
        elif check_uniqueness:
            if chunk_id in self._seen_chunk_ids:
                errors.append(f"chunk_id '{chunk_id}' дублируется")
            else:
                self._seen_chunk_ids.add(chunk_id)

        # Проверка context
        context = chunk_data.get("context")
        if context:
            if not isinstance(context, str):
                errors.append("context должен быть строкой")
            elif len(context) > self.MAX_CONTEXT_LENGTH:
                errors.append(
                    f"context превышает {self.MAX_CONTEXT_LENGTH} символов "
                    f"(текущая длина: {len(context)})"
                )
                # Обрезаем для предупреждения
                warnings.append(
                    f"context обрезан до {self.MAX_CONTEXT_LENGTH} символов"
                )

        # Проверка metrics
        metrics = chunk_data.get("metrics")
        if metrics is not None:
            if not isinstance(metrics, list):
                errors.append("metrics должен быть списком или null")
            else:
                if len(metrics) > self.MAX_METRICS_COUNT:
                    errors.append(
                        f"metrics содержит {len(metrics)} элементов, "
                        f"максимум {self.MAX_METRICS_COUNT}"
                    )
                
                # Проверка на русские строки
                for i, metric in enumerate(metrics):
                    if not isinstance(metric, str):
                        errors.append(f"metrics[{i}] должен быть строкой")
                    else:
                        metric_lower = metric.strip().lower()
                        # Проверяем наличие кириллицы
                        has_cyrillic = any(
                            '\u0400' <= char <= '\u04FF' for char in metric_lower
                        )
                        if not has_cyrillic:
                            warnings.append(
                                f"metrics[{i}] '{metric}' не содержит кириллицу"
                            )

        # Проверка years
        years = chunk_data.get("years")
        if years is not None:
            if not isinstance(years, list):
                errors.append("years должен быть списком или null")
            else:
                if len(years) > self.MAX_YEARS_COUNT:
                    errors.append(
                        f"years содержит {len(years)} элементов, "
                        f"максимум {self.MAX_YEARS_COUNT}"
                    )
                
                # Проверка на целые числа
                for i, year in enumerate(years):
                    if not isinstance(year, int):
                        try:
                            int(year)  # Попытка преобразования
                        except (ValueError, TypeError):
                            errors.append(f"years[{i}] должен быть целым числом")

        # Проверка geo (опционально)
        geo = chunk_data.get("geo")
        if geo is not None and not isinstance(geo, str):
            warnings.append("geo должен быть строкой или null")

        # Проверка time_granularity (опционально)
        time_granularity = chunk_data.get("time_granularity")
        if time_granularity is not None:
            valid_granularities = {"year", "quarter", "month", "day"}
            if time_granularity not in valid_granularities:
                warnings.append(
                    f"time_granularity '{time_granularity}' не в списке "
                    f"допустимых значений: {valid_granularities}"
                )

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )

    def validate_batch(
        self,
        chunks_data: List[Dict[str, Any]],
        check_uniqueness: bool = True,
    ) -> List[ValidationResult]:
        """
        Валидирует батч чанков.
        
        Args:
            chunks_data: Список словарей с данными чанков
            check_uniqueness: Проверять ли уникальность chunk_id
            
        Returns:
            Список ValidationResult для каждого чанка
        """
        results = []
        for chunk_data in chunks_data:
            result = self.validate_chunk(chunk_data, check_uniqueness=check_uniqueness)
            results.append(result)
        return results

    def reset(self) -> None:
        """Сбрасывает состояние валидатора (очищает seen_chunk_ids)."""
        self._seen_chunk_ids.clear()

    def normalize_context(self, context: str) -> str:
        """
        Нормализует context, обрезая до максимальной длины.
        
        Args:
            context: Исходный context
            
        Returns:
            Обрезанный context
        """
        if len(context) > self.MAX_CONTEXT_LENGTH:
            return context[:self.MAX_CONTEXT_LENGTH]
        return context

    def normalize_metrics(self, metrics: Optional[List[str]]) -> Optional[List[str]]:
        """
        Нормализует metrics: обрезает до максимума, фильтрует русские строки.
        
        Args:
            metrics: Исходный список метрик
            
        Returns:
            Нормализованный список метрик или None
        """
        if not metrics:
            return None
        
        if not isinstance(metrics, list):
            return None
        
        normalized = []
        for metric in metrics[:self.MAX_METRICS_COUNT]:
            if isinstance(metric, str):
                metric_lower = metric.strip().lower()
                # Проверяем наличие кириллицы
                has_cyrillic = any(
                    '\u0400' <= char <= '\u04FF' for char in metric_lower
                )
                if has_cyrillic:
                    normalized.append(metric_lower)
        
        return normalized if normalized else None

    def normalize_years(self, years: Optional[List[int]]) -> Optional[List[int]]:
        """
        Нормализует years: обрезает до максимума, фильтрует целые числа.
        
        Args:
            years: Исходный список годов
            
        Returns:
            Нормализованный список годов или None
        """
        if not years:
            return None
        
        if not isinstance(years, list):
            return None
        
        normalized = []
        for year in years[:self.MAX_YEARS_COUNT]:
            if isinstance(year, int):
                normalized.append(year)
            else:
                try:
                    normalized.append(int(year))
                except (ValueError, TypeError):
                    continue
        
        return normalized if normalized else None
