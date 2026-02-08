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
    # LLM Reranking Logging
    # =========================
    def log_llm_reranking(
        self,
        event: str,
        query: Optional[str] = None,
        enriched_query: Optional[Dict[str, Any]] = None,
        candidates_count: Optional[int] = None,
        candidate_ids: Optional[list] = None,
        system_prompt: Optional[str] = None,
        prompt: Optional[str] = None,
        raw_response: Optional[str] = None,
        rerank_scores: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
        elapsed_time: Optional[float] = None,
        ollama_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Логирование для LLM reranking.
        """
        record: Dict[str, Any] = {"event": event, **kwargs}
        if query:
            record["query"] = query
        if enriched_query:
            record["enriched_query"] = enriched_query
        if candidates_count is not None:
            record["candidates_count"] = candidates_count
        if candidate_ids:
            record["candidate_ids"] = candidate_ids
        if system_prompt:
            record["system_prompt"] = system_prompt
        if prompt:
            record["prompt"] = prompt
        if raw_response:
            record["raw_response"] = raw_response
        if rerank_scores:
            record["rerank_scores"] = rerank_scores
        if top_k is not None:
            record["top_k"] = top_k
        if elapsed_time is not None:
            record["elapsed_time"] = elapsed_time
        if ollama_config:
            record["ollama"] = ollama_config

        self.log("reranking-log", record)

    # =========================
    # LLM Enrichment Logging
    # =========================
    def log_llm_enrichment(
        self,
        event: str,
        pdf_name: Optional[str] = None,
        chunk_ids: Optional[list] = None,
        system_prompt: Optional[str] = None,
        prompt: Optional[str] = None,
        raw_response: Optional[str] = None,
        parsed_items: Optional[int] = None,
        ollama_config: Optional[Dict[str, Any]] = None,
        include_raw_response: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Логирование шагов обогащения LLM.

        Args:
            event: 'request' или 'response'
            pdf_name: название документа
            chunk_ids: список id обрабатываемых чанков
            system_prompt: системный промпт
            prompt: пользовательский промпт
            raw_response: ответ от LLM
            parsed_items: количество успешно распарсенных полей
            ollama_config: конфиг Ollama
            include_raw_response: если False, raw_response не логируется
        """
        record: Dict[str, Any] = {"event": event, **kwargs}
        if pdf_name:
            record["pdf_name"] = pdf_name
        if chunk_ids:
            record["chunk_ids"] = chunk_ids
        if system_prompt:
            record["system_prompt"] = system_prompt
        if prompt:
            record["prompt"] = prompt
        if raw_response and include_raw_response:
            record["raw_response"] = raw_response
        if parsed_items is not None:
            record["parsed_items"] = parsed_items
        if ollama_config:
            record["ollama"] = ollama_config

        self.log("llm-enrichment-log", record)

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
