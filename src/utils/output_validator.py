from __future__ import annotations

"""
Валидатор JSON-ответа OutputPipeline.

Двухуровневая проверка:
1. Синтаксическая — парсится ли JSON?
2. Семантическая — корректна ли структура (columns, rows, title)?

Поддерживает флаг no_data: true — LLM сигнализирует об отсутствии данных.
При no_data семантические проверки пропускаются.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"columns", "rows"}


class OutputValidator:
    """
    Валидирует JSON-ответ LLM и конвертирует в pandas DataFrame.
    """

    def validate(
        self, json_path: Path
    ) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
        """
        Полная валидация файла output_df.json.

        Returns:
            (is_valid, parsed_data_or_None, list_of_errors)
        """
        errors: List[str] = []

        # --- 1. Синтаксическая проверка ---
        data = self._check_syntax(json_path, errors)
        if data is None:
            return False, None, errors

        # --- 2. Семантическая проверка ---
        self._check_semantics(data, errors)
        if errors:
            return False, data, errors

        return True, data, []

    # ------------------------------------------------------------------
    # Синтаксическая проверка
    # ------------------------------------------------------------------
    @staticmethod
    def _check_syntax(
        json_path: Path, errors: List[str]
    ) -> Optional[Dict[str, Any]]:
        if not json_path.exists():
            errors.append(f"Файл не найден: {json_path}")
            return None

        raw = json_path.read_text(encoding="utf-8").strip()
        if not raw:
            errors.append("Файл пуст")
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")
            return None

        if not isinstance(data, dict):
            errors.append(f"Ожидался JSON-объект, получен {type(data).__name__}")
            return None

        return data

    # ------------------------------------------------------------------
    # Семантическая проверка
    # ------------------------------------------------------------------
    @staticmethod
    def is_no_data(data: Dict[str, Any]) -> bool:
        return data.get("no_data") is True

    @staticmethod
    def _check_semantics(data: Dict[str, Any], errors: List[str]) -> None:
        if data.get("no_data") is True:
            return

        # title
        title = data.get("title")
        if not title or not isinstance(title, str):
            errors.append("Отсутствует или невалидное поле 'title' (ожидается непустая строка)")

        # columns
        columns = data.get("columns")
        if not isinstance(columns, list) or len(columns) == 0:
            errors.append("Отсутствует или пустое поле 'columns' (ожидается непустой список)")
            return
        for i, col in enumerate(columns):
            if not isinstance(col, str) or not col.strip():
                errors.append(f"columns[{i}] должен быть непустой строкой, получено: {col!r}")

        # rows
        rows = data.get("rows")
        if not isinstance(rows, list) or len(rows) == 0:
            errors.append("Отсутствует или пустое поле 'rows' (ожидается непустой список)")
            return

        expected_len = len(columns)
        for i, row in enumerate(rows):
            if not isinstance(row, list):
                errors.append(f"rows[{i}] должен быть списком, получено: {type(row).__name__}")
                continue
            if len(row) != expected_len:
                errors.append(
                    f"rows[{i}] длина {len(row)} != columns длина {expected_len}"
                )
            for j, val in enumerate(row):
                if val is None:
                    errors.append(f"rows[{i}][{j}] содержит null")

    # ------------------------------------------------------------------
    # Конвертация в DataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
        """
        Конвертирует валидный JSON-ответ в pandas DataFrame.
        Предполагается, что validate() уже прошла успешно.
        """
        columns = data["columns"]
        rows = data["rows"]
        return pd.DataFrame(rows, columns=columns)
