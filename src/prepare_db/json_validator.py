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
    MAX_YEARS_COUNT = 9  # Увеличено для поддержки временных рядов (2016-2024)

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
        expected_chunk_ids: Optional[List[str]] = None,
        check_uniqueness: bool = True,
    ) -> tuple[List[ValidationResult], bool]:
        """
        Валидирует батч чанков с строгой проверкой формата.
        
        Args:
            chunks_data: Список словарей с данными чанков
            expected_chunk_ids: Ожидаемые chunk_id для точного сравнения
            check_uniqueness: Проверять ли уникальность chunk_id
            
        Returns:
            Кортеж (results, is_valid_format):
            - results: список ValidationResult для каждого чанка
            - is_valid_format: True если формат корректен (массив нужной длины)
        """
        # Проверка формата: должен быть список
        if not isinstance(chunks_data, list):
            # Возвращаем пустой результат с ошибкой
            return [ValidationResult(
                is_valid=False,
                errors=["Ответ должен быть массивом, получен объект"],
                warnings=[],
            )], False
        
        # Проверка количества элементов
        if expected_chunk_ids:
            if len(chunks_data) != len(expected_chunk_ids):
                # Добавляем ошибку в первый результат
                first_result = ValidationResult(
                    is_valid=False,
                    errors=[
                        f"Неверное количество элементов: ожидалось {len(expected_chunk_ids)}, "
                        f"получено {len(chunks_data)}"
                    ],
                    warnings=[],
                )
                return [first_result], False
        
        results = []
        for i, chunk_data in enumerate(chunks_data):
            # Проверка exact chunk_id match
            if expected_chunk_ids and i < len(expected_chunk_ids):
                expected_id = expected_chunk_ids[i]
                actual_id = chunk_data.get("chunk_id")
                
                if actual_id != expected_id:
                    # Нормализуем для сравнения
                    from src.prepare_db.chunk_filter import ChunkFilter
                    normalized_expected = ChunkFilter.normalize_chunk_id(expected_id)
                    normalized_actual = ChunkFilter.normalize_chunk_id(actual_id) if actual_id else ""
                    
                    if normalized_actual != normalized_expected:
                        chunk_data = chunk_data.copy()
                        chunk_data["chunk_id"] = expected_id  # Исправляем chunk_id
                        result = self.validate_chunk(chunk_data, check_uniqueness=check_uniqueness)
                        result.errors.insert(0, f"chunk_id не совпадает: ожидался '{expected_id}', получен '{actual_id}'")
                        results.append(result)
                        continue
            
            result = self.validate_chunk(chunk_data, check_uniqueness=check_uniqueness)
            results.append(result)
        
        is_valid_format = len(chunks_data) == len(expected_chunk_ids) if expected_chunk_ids else True
        return results, is_valid_format
    
    def validate_metrics_quality(self, metrics: Optional[List[str]], chunk_text: Optional[str] = None) -> List[str]:
        """
        Проверяет качество metrics: должны быть реальными метриками, а не определениями.
        
        Args:
            metrics: Список метрик для проверки
            chunk_text: Текст чанка для контекста (опционально)
            
        Returns:
            Список предупреждений о качестве метрик
        """
        warnings = []
        if not metrics:
            return warnings
        
        # Паттерны для определения "не метрик"
        non_metric_patterns = [
            r"^коммерческие организации$",
            r"^цифровые технологии$",
            r"^организации$",
            r"^технологии$",
            r"^определение",
            r"^понятие",
            r"^термин",
            r"\.\.\.",
            r"^[А-ЯЁ\s]{20,}",  # Длинные заголовки (более 20 символов заглавными)
        ]
        
        import re
        for i, metric in enumerate(metrics):
            metric_lower = metric.lower().strip()
            
            # Проверка на слишком длинные "метрики" (вероятно, это определения)
            if len(metric) > 50:
                warnings.append(
                    f"metrics[{i}] слишком длинный (возможно, это определение, а не метрика): '{metric[:50]}...'"
                )
            
            # Проверка на паттерны "не метрик"
            for pattern in non_metric_patterns:
                if re.search(pattern, metric_lower, re.IGNORECASE):
                    warnings.append(
                        f"metrics[{i}] похож на определение, а не метрику: '{metric}'"
                    )
                    break
        
        return warnings

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
        # Ограничиваем максимум MAX_YEARS_COUNT годов (9 для поддержки временных рядов)
        for year in years[:self.MAX_YEARS_COUNT]:
            if isinstance(year, int):
                normalized.append(year)
            else:
                try:
                    normalized.append(int(year))
                except (ValueError, TypeError):
                    continue
        
        return normalized if normalized else None
