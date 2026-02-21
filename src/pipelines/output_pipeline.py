from __future__ import annotations

"""
PIPELINE: формирование пользовательского вывода.

Этапы:
1. Построение 4-блочного промпта (роль, запрос, чанки, JSON-инструкция)
2. Вызов LLM через OllamaClient -> JSON-строка
3. Сохранение raw JSON в output_df.json
4. Валидация + конвертация в pandas DataFrame
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, List

import pandas as pd

from src.core.models import PipelineResult, ScoredChunk
from src.enrichers.client import OllamaClient, OllamaConfig
from src.utils.output_validator import OutputValidator
from src.utils.logger import get_logger

logger = logging.getLogger(__name__)


class OutputPipeline:
    """
    Pipeline финального вывода: LLM-генерация табличного JSON,
    валидация, конвертация в pandas DataFrame.
    """

    MAX_CHUNK_TEXT_LEN = 1500
    TOP_K_CHUNKS = 3

    def __init__(
        self,
        output_dir: Path,
        *,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._ollama = ollama_client or OllamaClient(
            OllamaConfig(num_predict=2048, format="json")
        )
        self._validator = OutputValidator()
        self._rag_logger = get_logger()

    def run(
        self,
        result: PipelineResult,
        user_query: str,
    ) -> Optional[pd.DataFrame]:
        """
        Основной метод: LLM-генерация JSON -> валидация -> DataFrame.

        Returns:
            DataFrame при успешной валидации, None при ошибке.
            Побочный эффект: сохраняет output_df.json, логирует ошибки.
        """
        print("[OutputPipeline] Генерация табличного ответа...")
        t0 = time.perf_counter()

        chunks = result.top_chunks[: self.TOP_K_CHUNKS]
        if not chunks:
            print("[OutputPipeline] Нет чанков для обработки.")
            return None

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(user_query, chunks)

        raw_json = self._call_llm(system_prompt, user_prompt)
        if not raw_json:
            print("[OutputPipeline] LLM не вернула ответ.")
            return None

        json_path = self.output_dir / "output_df.json"
        self._save_raw_json(raw_json, json_path)

        is_valid, data, errors = self._validator.validate(json_path)
        if not is_valid:
            print(f"[OutputPipeline] Валидация провалена: {errors}")
            self._rag_logger.log_output_validation_fail(
                user_query=user_query,
                raw_json=raw_json,
                errors=errors,
                system_prompt=system_prompt,
                prompt=user_prompt,
            )
            return None

        df = self._validator.to_dataframe(data)
        elapsed = time.perf_counter() - t0
        print(f"[OutputPipeline] Готово за {elapsed:.2f}s, shape={df.shape}")
        return df

    @property
    def title(self) -> Optional[str]:
        """Заголовок последнего успешного ответа (из JSON)."""
        json_path = self.output_dir / "output_df.json"
        if not json_path.exists():
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("title")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Блок 1: Роль модели (system prompt)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "Ты — аналитик статистических данных Беларуси.\n"
            "Твоя задача — извлечь структурированную таблицу из предоставленных "
            "фрагментов статистических сборников в ответ на запрос пользователя.\n"
            "Отвечай ТОЛЬКО валидным JSON. Никакого текста вне JSON."
        )

    # ------------------------------------------------------------------
    # Блоки 2-4: user_query + чанки + JSON-инструкция
    # ------------------------------------------------------------------
    def _build_user_prompt(
        self, user_query: str, chunks: List[ScoredChunk]
    ) -> str:
        chunks_block = self._format_chunks(chunks)

        json_instruction = (
            "Верни ТОЛЬКО один JSON-объект строго следующей схемы:\n"
            "{\n"
            '  "title": "<краткое описание таблицы>",\n'
            '  "columns": ["<название столбца 1>", "<название столбца 2>", ...],\n'
            '  "rows": [[значение1, значение2, ...], ...],\n'
            '  "chart_type": "bar"\n'
            "}\n\n"
            "Правила:\n"
            "1. columns — список строк, названия столбцов таблицы.\n"
            "2. rows — список списков, каждая строка содержит значения в порядке columns.\n"
            "3. Длина каждого элемента rows ДОЛЖНА совпадать с длиной columns.\n"
            "4. Числовые значения — числа (int или float), НЕ строки.\n"
            "5. Не оставляй пустых (null) ячеек.\n"
            "6. chart_type всегда \"bar\".\n"
            "7. Используй ТОЛЬКО данные из предоставленных фрагментов.\n"
            "8. Ответ — ТОЛЬКО JSON, без markdown, без комментариев."
        )

        return (
            f"=== ЗАПРОС ПОЛЬЗОВАТЕЛЯ ===\n"
            f"{user_query}\n\n"
            f"=== ФРАГМЕНТЫ ДАННЫХ ===\n"
            f"{chunks_block}\n\n"
            f"=== ИНСТРУКЦИЯ ПО ФОРМАТУ ===\n"
            f"{json_instruction}"
        )

    def _format_chunks(self, chunks: List[ScoredChunk]) -> str:
        parts: List[str] = []
        for i, sc in enumerate(chunks, 1):
            text = (sc.chunk.text or "")[:self.MAX_CHUNK_TEXT_LEN]
            parts.append(
                f"--- Фрагмент {i} (источник: {sc.chunk.source}, стр. {sc.chunk.page}) ---\n"
                f"{text}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Вызов LLM
    # ------------------------------------------------------------------
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        try:
            response = self._ollama.generate(
                user_prompt,
                system_prompt=system_prompt,
                format="json",
                temperature=0.0,
                num_predict=2048,
            )
            return response or None
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Сохранение raw JSON
    # ------------------------------------------------------------------
    @staticmethod
    def _save_raw_json(raw_json: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw_json)
