from __future__ import annotations

"""
PIPELINE: формирование пользовательского вывода.

Назначение:
- Преобразовать результат query_pipeline
- Подготовить output для пользователя:
  - таблицы
  - графики
  - JSON
  - LLM-ответ (будущее)

На текущем этапе — архитектурная заглушка.
"""

from typing import Optional
from pathlib import Path

from src.core.models import PipelineResult


class OutputPipeline:
    """
    Pipeline финального вывода.

    Архитектурная роль:
    - получает результат QueryPipeline
    - решает, КАК его представить пользователю
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        mode: str = "raw",
    ):
        """
        Args:
            output_dir: директория для сохранения результатов (usage/outputs)
            mode: режим вывода (raw | table | chart | llm)
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode

    def run(self, result: PipelineResult) -> None:
        """
        Основной метод пайплайна.

        Args:
            result: результат QueryPipeline
        """
        print("📤 OutputPipeline: формирование результата...")
        print(f"🧩 Режим вывода: {self.mode}")

        if self.mode == "raw":
            self._output_raw(result)
        elif self.mode == "table":
            self._output_table(result)
        elif self.mode == "chart":
            self._output_chart(result)
        elif self.mode == "llm":
            self._output_llm(result)
        else:
            raise ValueError(f"Unknown output mode: {self.mode}")

        print("✅ OutputPipeline завершён")

    # ------------------------------------------------------------------
    # Заглушки этапов вывода
    # ------------------------------------------------------------------

    def _output_raw(self, result: PipelineResult) -> None:
        """
        RAW-вывод: просто печать топ-чанков.

        Используется для отладки и проверки retrieval/rerank.
        """
        print("\n=== TOP CHUNKS ===")
        for i, ch in enumerate(result.top_chunks, 1):
            print(f"\n[{i}] score={ch.score:.4f}")
            print(ch.chunk.context)

    def _output_table(self, result: PipelineResult) -> None:
        """
        Заглушка под табличный вывод.
        """
        raise NotImplementedError("Table output pipeline is not implemented yet")

    def _output_chart(self, result: PipelineResult) -> None:
        """
        Заглушка под графики.
        """
        raise NotImplementedError("Chart output pipeline is not implemented yet")

    def _output_llm(self, result: PipelineResult) -> None:
        """
        Заглушка под LLM-ответ пользователю.
        """
        raise NotImplementedError("LLM output pipeline is not implemented yet")
