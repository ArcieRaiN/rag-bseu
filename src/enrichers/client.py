from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol
import time
import requests
from src.utils.logger import get_logger

logger = get_logger()


# --- HTTP client interface для тестов / DI ---
class HTTPClient(Protocol):
    def post(self, url: str, json: Dict[str, Any], timeout: float) -> Any:
        ...


# --- Конфигурация Ollama ---
@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    timeout: float = 120.0

    # Дефолтные параметры генерации (можно переопределить в generate)
    temperature: float = 0.0
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    num_predict: int = 800

    # Формат ответа (например, json)
    format: Optional[str] = None
    keep_alive: str = "5m"  # default keep_alive для generate


class OllamaClient:
    """
    Клиент для Ollama HTTP API с логированием и retry.

    Поддержка Dependency Injection через http_client (для юнит-тестов).
    """

    def __init__(self, config: Optional[OllamaConfig] = None, http_client: Optional[HTTPClient] = None):
        self.config = config or OllamaConfig()
        self._http_client = http_client or requests

    def reset_context(self) -> bool:
        """
        Сбросить контекст модели на сервере.
        Возвращает True при успехе, False при ошибке.
        """
        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": "",
            "stream": False,
            "keep_alive": "0",
            "num_predict": 1,
        }
        try:
            resp = self._http_client.post(url, json=payload, timeout=5.0)
            resp.raise_for_status()
            logger.log("llm_client", {"event": "reset_context_success"})
            return True
        except Exception as e:
            logger.log("llm_client", {"event": "reset_context_failed", "error": str(e)})
            return False

    def generate(
            self,
            prompt: str,
            *,
            system_prompt: Optional[str] = None,
            max_retries: int = 3,
            **params: Any,
    ) -> str:
        """
        Сгенерировать текст от Ollama.

        Аргументы:
        - prompt: основной текст запроса
        - system_prompt: опциональный системный промпт
        - max_retries: число попыток при таймауте
        - params: любые дополнительные параметры, переопределяющие дефолты
        """
        url = f"{self.config.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repeat_penalty": self.config.repeat_penalty,
            "num_predict": self.config.num_predict,
            "keep_alive": self.config.keep_alive,
        }
        if self.config.format:
            payload["format"] = self.config.format
        if system_prompt:
            payload["system"] = system_prompt
        payload.update(params)

        for attempt in range(1, max_retries + 1):
            try:
                logger.log("llm_client", {
                    "event": "request",
                    "attempt": attempt,
                    "payload": payload,
                })
                resp = self._http_client.post(url, json=payload, timeout=self.config.timeout)
                resp.raise_for_status()
                data = resp.json()
                result = data.get("response", "") or ""
                logger.log("llm_client", {
                    "event": "response",
                    "attempt": attempt,
                    "response_length": len(result),
                    "raw_response": result[:500],  # первые 500 символов
                })
                return result
            except requests.exceptions.ReadTimeout:
                backoff = (2 ** (attempt - 1)) * 2
                logger.log("llm_client", {
                    "event": "timeout",
                    "attempt": attempt,
                    "backoff": backoff,
                })
                if attempt < max_retries:
                    time.sleep(backoff)
                    continue
                raise
            except requests.exceptions.RequestException as e:
                logger.log("llm_client", {"event": "request_error", "error": str(e)})
                raise
