"""
Unit-тесты для ChunkValidator.

Проверяют:
- Валидацию chunk_id (уникальность, обязательность)
- Валидацию context (длина <= 256 символов)
- Валидацию metrics (максимум 5, русские строки)
- Валидацию years (максимум 5, целые числа)
- Нормализацию данных
"""

import pytest
from src.prepare_db.json_validator import ChunkValidator, ValidationResult


class TestChunkValidator:
    """Тесты для ChunkValidator."""

    def test_valid_chunk(self):
        """Тест валидного чанка."""
        validator = ChunkValidator()
        chunk_data = {
            "chunk_id": "doc.pdf::page1::chunk0",
            "context": "Краткое описание чанка",
            "geo": "Минск",
            "metrics": ["удой молока", "инвестиции"],
            "years": [2023, 2024],
            "time_granularity": "year",
            "oked": "01.11",
        }
        result = validator.validate_chunk(chunk_data)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_chunk_id(self):
        """Тест отсутствующего chunk_id."""
        validator = ChunkValidator()
        chunk_data = {
            "context": "Описание",
        }
        result = validator.validate_chunk(chunk_data)
        assert not result.is_valid
        assert any("chunk_id отсутствует" in error for error in result.errors)

    def test_duplicate_chunk_id(self):
        """Тест дублирующегося chunk_id."""
        validator = ChunkValidator()
        chunk_id = "doc.pdf::page1::chunk0"
        chunk_data1 = {"chunk_id": chunk_id, "context": "Описание 1"}
        chunk_data2 = {"chunk_id": chunk_id, "context": "Описание 2"}
        
        result1 = validator.validate_chunk(chunk_data1)
        assert result1.is_valid
        
        result2 = validator.validate_chunk(chunk_data2)
        assert not result2.is_valid
        assert any("дублируется" in error for error in result2.errors)

    def test_context_too_long(self):
        """Тест слишком длинного context."""
        validator = ChunkValidator()
        long_context = "а" * 300  # Превышает MAX_CONTEXT_LENGTH = 256
        chunk_data = {
            "chunk_id": "doc.pdf::page1::chunk0",
            "context": long_context,
        }
        result = validator.validate_chunk(chunk_data)
        assert not result.is_valid
        assert any("превышает" in error for error in result.errors)

    def test_metrics_too_many(self):
        """Тест слишком большого количества metrics."""
        validator = ChunkValidator()
        chunk_data = {
            "chunk_id": "doc.pdf::page1::chunk0",
            "metrics": ["метрика1", "метрика2", "метрика3", "метрика4", "метрика5", "метрика6"],
        }
        result = validator.validate_chunk(chunk_data)
        assert not result.is_valid
        assert any("максимум" in error and "metrics" in error for error in result.errors)

    def test_metrics_non_russian(self):
        """Тест метрик не на русском языке."""
        validator = ChunkValidator()
        chunk_data = {
            "chunk_id": "doc.pdf::page1::chunk0",
            "metrics": ["milk production", "investments"],  # Английский текст
        }
        result = validator.validate_chunk(chunk_data)
        # Должно быть предупреждение, но не ошибка
        assert result.is_valid  # Валидация проходит, но есть предупреждение
        assert any("кириллицу" in warning for warning in result.warnings)

    def test_years_too_many(self):
        """Тест слишком большого количества years."""
        validator = ChunkValidator()
        chunk_data = {
            "chunk_id": "doc.pdf::page1::chunk0",
            "years": [2020, 2021, 2022, 2023, 2024, 2025],
        }
        result = validator.validate_chunk(chunk_data)
        assert not result.is_valid
        assert any("максимум" in error and "years" in error for error in result.errors)

    def test_years_non_integer(self):
        """Тест years с нецелочисленными значениями."""
        validator = ChunkValidator()
        chunk_data = {
            "chunk_id": "doc.pdf::page1::chunk0",
            "years": [2023.5, "2024", "not a year"],
        }
        result = validator.validate_chunk(chunk_data)
        assert not result.is_valid
        assert any("целым числом" in error for error in result.errors)

    def test_time_granularity_invalid(self):
        """Тест невалидного time_granularity."""
        validator = ChunkValidator()
        chunk_data = {
            "chunk_id": "doc.pdf::page1::chunk0",
            "time_granularity": "invalid",
        }
        result = validator.validate_chunk(chunk_data)
        # Должно быть предупреждение, но не ошибка
        assert result.is_valid
        assert any("time_granularity" in warning for warning in result.warnings)

    def test_normalize_context(self):
        """Тест нормализации context."""
        validator = ChunkValidator()
        long_context = "а" * 300
        normalized = validator.normalize_context(long_context)
        assert len(normalized) == 256
        assert normalized == "а" * 256

    def test_normalize_metrics(self):
        """Тест нормализации metrics."""
        validator = ChunkValidator()
        metrics = [
            "УДОЙ МОЛОКА",  # Должно быть приведено к нижнему регистру
            "инвестиции",
            "milk production",  # Должно быть отфильтровано (не русский)
            "метрика3",
            "метрика4",
            "метрика5",
            "метрика6",  # Должно быть обрезано (максимум 5)
        ]
        normalized = validator.normalize_metrics(metrics)
        assert normalized is not None
        assert len(normalized) == 5
        assert all(isinstance(m, str) for m in normalized)
        assert all(m.islower() for m in normalized)
        assert "milk production" not in normalized

    def test_normalize_years(self):
        """Тест нормализации years."""
        validator = ChunkValidator()
        years = [2020, "2021", 2022.5, "not a year", 2023, 2024, 2025]
        normalized = validator.normalize_years(years)
        assert normalized is not None
        assert len(normalized) == 5  # Максимум 5
        assert all(isinstance(y, int) for y in normalized)
        assert 2020 in normalized
        assert 2021 in normalized
        assert 2023 in normalized

    def test_validate_batch(self):
        """Тест валидации батча чанков."""
        validator = ChunkValidator()
        chunks_data = [
            {"chunk_id": "doc1::page1::chunk0", "context": "Описание 1"},
            {"chunk_id": "doc1::page1::chunk1", "context": "Описание 2"},
            {"chunk_id": "doc1::page1::chunk2", "context": "Описание 3"},
        ]
        results = validator.validate_batch(chunks_data)
        assert len(results) == 3
        assert all(r.is_valid for r in results)

    def test_reset(self):
        """Тест сброса состояния валидатора."""
        validator = ChunkValidator()
        chunk_id = "doc.pdf::page1::chunk0"
        
        # Первый чанк проходит
        result1 = validator.validate_chunk({"chunk_id": chunk_id, "context": "Описание 1"})
        assert result1.is_valid
        
        # Второй чанк с тем же ID должен быть отклонен
        result2 = validator.validate_chunk({"chunk_id": chunk_id, "context": "Описание 2"})
        assert not result2.is_valid
        
        # После сброса тот же ID снова проходит
        validator.reset()
        result3 = validator.validate_chunk({"chunk_id": chunk_id, "context": "Описание 3"})
        assert result3.is_valid

    def test_validate_batch_without_uniqueness_check(self):
        """Тест валидации батча без проверки уникальности."""
        validator = ChunkValidator()
        chunks_data = [
            {"chunk_id": "doc1::page1::chunk0", "context": "Описание 1"},
            {"chunk_id": "doc1::page1::chunk0", "context": "Описание 2"},  # Дубликат
        ]
        results = validator.validate_batch(chunks_data, check_uniqueness=False)
        # Оба должны быть валидными, так как проверка уникальности отключена
        assert len(results) == 2
        assert all(r.is_valid for r in results)
