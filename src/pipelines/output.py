from __future__ import annotations

"""
Генерация табличного ответа через LLM.

Этапы:
  1. Построение 4-блочного промпта (роль, запрос, чанки, JSON-инструкция)
  2. Вызов LLM (Ollama, JSON mode, temperature=0)
  3. Валидация JSON-схемы (OutputValidator) + конвертация в DataFrame

JSON-схема: 1NF-таблица, unit в скобках в названии столбца,
no_data: true при отсутствии данных, source_fragment для атрибуции.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, List

import pandas as pd

from src.core.models import PipelineResult, ScoredChunk
from src.enrichers.ollama_client import OllamaClient, OllamaConfig
from src.utils.output_validator import OutputValidator
from src.utils.logger import get_logger

logger = logging.getLogger(__name__)


class OutputPipeline:
    """
    Pipeline финального вывода: LLM-генерация табличного JSON,
    валидация, конвертация в pandas DataFrame.
    """

    MAX_CHUNK_TEXT_LEN = 4000
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
        self._last_chunk_sources: List[str] = []
        self._last_no_data: bool = False

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
        self._last_no_data = False

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

        if self._validator.is_no_data(data):
            self._last_no_data = True
            print("[OutputPipeline] LLM: данных по запросу не найдено.")
            return None

        df = self._validator.to_dataframe(data)
        frag_idx = data.get("source_fragment")
        if isinstance(frag_idx, int) and 1 <= frag_idx <= len(chunks):
            src_chunk = chunks[frag_idx - 1].chunk
        else:
            src_chunk = chunks[0].chunk
        self._last_chunk_sources = [
            f"{src_chunk.source}, стр. {src_chunk.page}" if src_chunk.page else src_chunk.source
        ]
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

    @property
    def sources(self) -> List[str]:
        return self._last_chunk_sources

    @property
    def no_data(self) -> bool:
        return self._last_no_data

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
            '  "no_data": false,\n'
            '  "title": "<краткое описание таблицы>",\n'
            '  "unit": "<единица измерения числовых значений>",\n'
            '  "columns": ["<категория>", "<показатель (единица)>"],\n'
            '  "rows": [[значение_категории, число], ...],\n'
            '  "chart_type": "bar",\n'
            '  "source_fragment": <номер фрагмента 1, 2 или 3>\n'
            "}\n\n"
            "Правила:\n"
            "1. Таблица ВСЕГДА в первой нормальной форме (1NF): каждая строка — "
            "одно наблюдение. Годы, страны, категории — это ЗНАЧЕНИЯ в столбцах, "
            "а НЕ названия столбцов. Пример: columns=[\"Год\", \"Цена (руб/кг)\"], "
            "rows=[[2020, 2.37], [2021, 2.12]].\n"
            "2. columns — список строк. Первый столбец — категория (Год, Страна, "
            "Показатель и т.д.), остальные — числовые значения.\n"
            "3. Длина каждого элемента rows ДОЛЖНА совпадать с длиной columns.\n"
            "4. Числовые значения — числа (int или float), НЕ строки.\n"
            "5. Не оставляй пустых (null) ячеек.\n"
            "6. chart_type всегда \"bar\".\n"
            "7. unit — единица измерения числовых значений, взятая из заголовка "
            "или шапки таблицы в источнике (руб/кг, тыс. человек, млн руб, тонн, "
            "%, и т.д.). ВКЛЮЧАЙ единицу измерения в скобках в название числового "
            "столбца: \"Цена (руб/кг)\", \"Население (тыс. человек)\".\n"
            "8. Используй ТОЛЬКО данные из предоставленных фрагментов. Извлекай "
            "числа ТОЧНО как они указаны в источнике — НЕ округляй, НЕ умножай, "
            "НЕ выдумывай значения. ЗАПРЕЩЕНО генерировать данные за годы или "
            "периоды, которых нет во фрагментах. ВНИМАТЕЛЬНО сопоставляй значения "
            "с заголовками столбцов (годами): первое число после названия строки "
            "соответствует первому году в шапке, второе — второму и т.д. "
            "НЕ сдвигай значения относительно годов.\n"
            "9. Включай ТОЛЬКО строки, непосредственно относящиеся к запросу "
            "пользователя. НЕ включай все строки таблицы — только те, которые "
            "отвечают на вопрос.\n"
            "10. ОБЯЗАТЕЛЬНО читай сноски и примечания к таблицам (обычно внизу, "
            "начинаются с цифры и скобки, например «1)»). Если указана деноминация "
            "(например «с учетом деноминации, уменьшение в 10 000 раз»), ПРИВЕДИ "
            "все значения к современному масштабу: раздели старые значения (до года "
            "деноминации) на указанный коэффициент.\n"
            "11. Если в предоставленных фрагментах НЕТ данных для ответа на запрос, "
            "верни {\"no_data\": true, \"title\": \"\", \"unit\": \"\", "
            "\"columns\": [], \"rows\": [], \"chart_type\": \"bar\", "
            "\"source_fragment\": 0}. НЕ выдумывай, не экстраполируй.\n"
            "12. source_fragment — номер фрагмента (1, 2 или 3), из которого "
            "ты взял основные данные для таблицы.\n"
            "13. Ответ — ТОЛЬКО JSON, без markdown, без комментариев."
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
