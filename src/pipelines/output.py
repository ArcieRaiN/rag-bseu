from __future__ import annotations

"""
Генерация табличного ответа через LLM.

Этапы:
  1. Построение 4-блочного промпта (роль, запрос, чанки, JSON-инструкция)
  2. Вызов LLM (Ollama, JSON mode, temperature=0)
  3. Валидация JSON-схемы (OutputValidator) + конвертация в DataFrame

JSON-схема: 1NF-таблица, unit в скобках в названии столбца,
no_data: true при отсутствии данных, source_fragment для атрибуции

Порог hybrid_score: при max(hybrid_score) среди топ-3 строго меньше min_hybrid_score
LLM не вызывается (снижение галлюцинаций при слабом совпадении с базой).
Значение hybrid_score — как в PipelineResult после поиска (при RRF это сумма RRF;
при включённом reranker поле перезаписывается — порог нужно перекалибровать).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.core.models import PipelineResult, ScoredChunk
from src.enrichers.ollama_client import OllamaClient, OllamaConfig
from src.utils.output_validator import OutputValidator
from src.utils.logger import get_logger

logger = logging.getLogger(__name__)

# Согласовано с p10 в reports/v4/hybrid_score_percentiles.json (перекалибровка 1500 запросов).
_DEFAULT_MIN_HYBRID_SCORE = 0.032
_MIN_HYBRID_SCORE_ARG_UNSET = object()


def _parse_min_hybrid_score_from_env() -> Optional[float]:
    raw = os.getenv("RAG_MIN_HYBRID_SCORE")
    if raw is None or raw.strip() == "":
        return _DEFAULT_MIN_HYBRID_SCORE
    if raw.strip().lower() in ("off", "none", "false", "0"):
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid RAG_MIN_HYBRID_SCORE=%r, using default", raw)
        return _DEFAULT_MIN_HYBRID_SCORE


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
        min_hybrid_score: object = _MIN_HYBRID_SCORE_ARG_UNSET,
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
        self._last_kb_miss: bool = False
        if min_hybrid_score is _MIN_HYBRID_SCORE_ARG_UNSET:
            self._min_hybrid_score: Optional[float] = _parse_min_hybrid_score_from_env()
        else:
            self._min_hybrid_score = min_hybrid_score  # type: ignore[assignment]

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
        self._last_kb_miss = False

        chunks = result.top_chunks[: self.TOP_K_CHUNKS]
        if not chunks:
            print("[OutputPipeline] Нет чанков для обработки.")
            return None

        if self._min_hybrid_score is not None:
            mx = max(float(sc.hybrid_score) for sc in chunks)
            if mx < self._min_hybrid_score:
                self._last_kb_miss = True
                self._last_chunk_sources = []
                logger.info(
                    "OutputPipeline: skip LLM (weak retrieval): max_hybrid_score=%.6f < min=%.6f",
                    mx,
                    self._min_hybrid_score,
                )
                print(
                    "[OutputPipeline] Пропуск LLM: max hybrid_score "
                    f"({mx:.6f}) < порог ({self._min_hybrid_score})."
                )
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

    @property
    def kb_miss(self) -> bool:
        """True если LLM не вызывался из-за слабого hybrid_score среди топ-3."""
        return self._last_kb_miss


    # ------------------------------------------------------------------
    # Блок 1: Роль модели (system prompt)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "Ты — аналитик официальной статистики Беларуси.\n"
            "Твоя задача — извлекать из фрагментов статистических сборников "
            "строго структурированные табличные данные по запросу пользователя.\n\n"

            "Ты ОБЯЗАН:\n"
            "- извлекать данные точно как в источнике;\n"
            "- формировать таблицу в первой нормальной форме (1NF);\n"
            "- возвращать ТОЛЬКО валидный JSON.\n\n"

            "Запрещено:\n"
            "- выдумывать данные;\n"
            "- интерполировать значения;\n"
            "- добавлять годы или категории, которых нет в источнике;\n"
            "- менять масштаб чисел без явного указания в примечаниях;\n"
            "- выводить текст вне JSON."
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
            '  "title": "Динамика <название показателя>, <единицы измерения>",\n'
            '  "unit": "<единицы измерения>",\n'
            '  "columns": ["<категория>", "<числовой показатель с единицей измерения>"],\n'
            '  "rows": [[значение_категории, значение_показателя], ...],\n'
            '  "chart_type": "bar",\n'
            '  "source_fragment": <номер фрагмента>\n'
            "}\n\n"

            "СТРОГИЕ ПРАВИЛА:\n\n"

            "1. title ОБЯЗАТЕЛЬНО формируй по шаблону:\n"
            "\"Динамика <название показателя>, <единицы измерения>\".\n"
            "Примеры:\n"
            "- \"Динамика инвестиций в основной капитал, млн руб.\"\n"
            "- \"Динамика численности населения, тыс. человек\"\n"
            "- \"Динамика рентабельности продаж, %\".\n\n"

            "2. Таблица ОБЯЗАТЕЛЬНО должна быть в первой нормальной форме (1NF).\n"
            "Каждая строка rows — одно наблюдение.\n"
            "Годы, регионы, страны и категории должны быть значениями строк, а НЕ названиями столбцов.\n\n"

            "3. Для временных рядов используй структуру:\n"
            'columns = ["Год", "<показатель (единицы)>"]\n'
            "Пример:\n"
            'columns = ["Год", "Инфляция (%)"]\n'
            'rows = [[2020, 5.5], [2021, 9.97]]\n\n'

            "4. Первый столбец columns — категориальный.\n"
            "Остальные столбцы — числовые.\n\n"

            "5. Длина каждой строки rows должна строго совпадать "
            "с количеством columns.\n\n"

            "6. Числовые значения записывай только как числа "
            "(int или float), НЕ как строки.\n\n"

            "7. Не используй null, пустые строки или пропуски.\n\n"

            "8. chart_type всегда \"bar\".\n\n"

            "9. unit — единица измерения показателя, извлечённая "
            "из заголовка, подзаголовка, шапки таблицы или примечаний.\n\n"

            "10. Название числового столбца ОБЯЗАТЕЛЬНО должно содержать "
            "единицу измерения в скобках.\n"
            "Примеры:\n"
            "- \"ВВП (млн руб.)\"\n"
            "- \"Население (тыс. человек)\"\n"
            "- \"Рентабельность (%)\".\n\n"

            "11. Используй ТОЛЬКО данные из предоставленных фрагментов.\n"
            "НЕ округляй значения.\n"
            "НЕ изменяй порядок данных.\n"
            "НЕ генерируй отсутствующие годы.\n\n"

            "12. ВНИМАТЕЛЬНО сопоставляй значения с заголовками.\n"
            "Первое значение строки соответствует первому заголовку года,\n"
            "второе — второму и т.д.\n"
            "Запрещено смещать значения относительно годов.\n\n"

            "13. Включай только данные, непосредственно относящиеся "
            "к запросу пользователя.\n\n"

            "14. ОБЯЗАТЕЛЬНО анализируй сноски и примечания.\n"
            "Если указана деноминация или изменение масштаба,\n"
            "приводи значения к актуальному масштабу.\n\n"

            "15. Если данных недостаточно для построения таблицы,\n"
            "верни:\n"
            "{\n"
            '  "no_data": true,\n'
            '  "title": "",\n'
            '  "unit": "",\n'
            '  "columns": [],\n'
            '  "rows": [],\n'
            '  "chart_type": "bar",\n'
            '  "source_fragment": 0\n'
            "}\n\n"

            "16. source_fragment — номер основного фрагмента "
            "(1, 2 или 3), из которого извлечены данные.\n\n"

            "17. Ответ должен содержать ТОЛЬКО JSON.\n"
            "Без markdown.\n"
            "Без пояснений.\n"
            "Без комментариев."
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
