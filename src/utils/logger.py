# C:\Users\Alex\Downloads\projects\rag-bseu\src\utils\logger.py

"""
Централизованное логирование для RAG-системы.

Логи сохраняются в формате JSONL (JSON Lines) для удобства анализа и отладки.
Поддерживаются отдельные методы для:
- LLM reranking
- LLM enrichment
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional
import time


class RAGLogger:
    """
    Потокобезопасный логгер для записи в JSONL файлы.

    Каждый лог-файл содержит JSON-объекты, по одному на строку (JSONL формат).
    Это удобно для append во время долгого прогона и для анализа.
    """

    def __init__(self, logs_dir: Path):
        """
        Args:
            logs_dir: Директория для сохранения логов
        """
        self._logs_dir = Path(logs_dir)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}

    def _get_lock(self, log_name: str) -> threading.Lock:
        """Получить или создать lock для конкретного лог-файла."""
        if log_name not in self._locks:
            self._locks[log_name] = threading.Lock()
        return self._locks[log_name]

    def log(self, log_name: str, record: Dict[str, Any]) -> None:
        """
        Записать одну запись в лог-файл.

        Args:
            log_name: Имя лог-файла (без расширения .json)
            record: Словарь с данными для логирования
        """
        log_path = self._logs_dir / f"{log_name}.json"
        lock = self._get_lock(log_name)

        # Добавляем timestamp если его нет
        if "ts" not in record:
            record["ts"] = time.time()

        line = json.dumps(record, ensure_ascii=False)

        with lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


    # =========================
    # LLM Enrichment Failures
    # =========================
    def log_llm_enrichment_fail(
        self,
        *,
        pdf_name: str,
        chunk_id: str,
        page: int | None,
        chunk_text: str | None,
        system_prompt: str,
        prompt: str,
        raw_response: str,
        attempts: int,
        error_type: str = "parse_failed",
        **kwargs: Any,
    ) -> None:
        """
        Логирование неудачных enrichment-попыток (parse / validation / etc).
        """
        record: Dict[str, Any] = {
            "event": "enrichment_fail",
            "error_type": error_type,
            "pdf_name": pdf_name,
            "chunk_id": chunk_id,
            "page": page,
            "attempts": attempts,
            "chunk_text": chunk_text,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "raw_response": raw_response,
            **kwargs,
        }

        self.log("llm-enrichment-fails", record)



# =========================
# Singleton Instance
# =========================
_logs_dir = Path(__file__).resolve().parent.parent.parent / "usage" / "logs"
_global_logger: Optional[RAGLogger] = None

def get_logger(logs_dir: Optional[Path] = None) -> RAGLogger:
    """
    Получить глобальный экземпляр логгера.

    Args:
        logs_dir: Директория для логов (по умолчанию usage/logs)
    
    Returns:
        Экземпляр RAGLogger
    """
    global _global_logger
    if _global_logger is None:
        if logs_dir is None:
            logs_dir = _logs_dir
        _global_logger = RAGLogger(logs_dir)
    return _global_logger
