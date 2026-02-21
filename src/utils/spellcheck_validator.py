"""
Валидатор spellcheck: проверяет, не исказил ли spellchecker смысл user_query.

Если spellchecker испортил смысл запроса (например, заменил термин на другой),
пайплайну следует использовать оригинальный user_query.

Реализация: Ollama LLM для семантической проверки + fallback на эвристики.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.enrichers.client import OllamaClient


@dataclass
class SpellcheckValidationResult:
    """Результат проверки spellcheck."""

    query_to_use: str  # Какой запрос использовать в пайплайне
    was_corrupted: bool  # True если spellcheck исказил смысл
    method: str  # "unchanged" | "ollama_ok" | "ollama_corrupted" | "fallback_original"


class SpellcheckValidator:
    """
    Проверяет, сохранил ли spellchecker смысл user_query.
    При искажении смысла возвращает оригинальный запрос.
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self._ollama = ollama_client or OllamaClient()

    def validate(
        self,
        original_query: str,
        corrected_query: str,
        *,
        use_ollama: bool = True,
    ) -> SpellcheckValidationResult:
        """
        Определяет, какой запрос использовать: оригинал или исправленный.

        Args:
            original_query: Исходный user_query
            corrected_query: Результат spellchecker.correct_query()
            use_ollama: Использовать Ollama для семантической проверки

        Returns:
            SpellcheckValidationResult с query_to_use и флагом was_corrupted
        """
        if original_query.strip() == corrected_query.strip():
            return SpellcheckValidationResult(
                query_to_use=corrected_query,
                was_corrupted=False,
                method="unchanged",
            )

        if use_ollama:
            result = self._validate_via_ollama(original_query, corrected_query)
            if result is not None:
                return result

        # Fallback: при сомнениях или недоступности Ollama — используем оригинал
        return SpellcheckValidator._heuristic_fallback(
            original_query, corrected_query
        )

    def _validate_via_ollama(
        self, original: str, corrected: str
    ) -> Optional[SpellcheckValidationResult]:
        """Проверка через Ollama: сохранился ли смысл."""
        system_prompt = (
            "Ты проверяешь, не исказил ли spellchecker смысл запроса. "
            "Оригинал — это то, что написал пользователь. "
            "Исправленный — результат автоматической проверки орфографии. "
            "Ответь ТОЛЬКО одним словом: OK если смысл сохранён (только исправлены опечатки), "
            "CORRUPTED если смысл изменён, искажён или подставлено другое слово с другим значением."
        )
        prompt = (
            f"Оригинал: {original}\n"
            f"Исправленный: {corrected}\n"
            "Ответ (OK или CORRUPTED):"
        )
        try:
            response = self._ollama.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                num_predict=20,
            )
            answer = (response or "").strip().upper()
            if "CORRUPT" in answer or "CORRUPTED" in answer:
                return SpellcheckValidationResult(
                    query_to_use=original,
                    was_corrupted=True,
                    method="ollama_corrupted",
                )
            if "OK" in answer or "ДА" in answer or "YES" in answer:
                return SpellcheckValidationResult(
                    query_to_use=corrected,
                    was_corrupted=False,
                    method="ollama_ok",
                )
        except Exception:
            pass  # Fallback below

        return None

    @staticmethod
    def _heuristic_fallback(
        original: str, corrected: str
    ) -> SpellcheckValidationResult:
        """
        Эвристика: если изменено слишком много слов относительно длины запроса,
        считаем, что смысл мог быть искажён — используем оригинал.
        """
        orig_tokens = original.split()
        corr_tokens = corrected.split()
        if not orig_tokens:
            return SpellcheckValidationResult(
                query_to_use=original,
                was_corrupted=False,
                method="fallback_original",
            )

        changed = sum(1 for a, b in zip(orig_tokens, corr_tokens) if a != b)
        # Разная длина — могли добавить/удалить слова
        if len(orig_tokens) != len(corr_tokens):
            changed += abs(len(orig_tokens) - len(corr_tokens))

        ratio = changed / len(orig_tokens)
        # Если изменено > 40% слов — осторожно, используем оригинал
        if ratio > 0.4:
            return SpellcheckValidationResult(
                query_to_use=original,
                was_corrupted=True,
                method="fallback_original",
            )

        # Иначе доверяем spellcheck
        return SpellcheckValidationResult(
            query_to_use=corrected,
            was_corrupted=False,
            method="fallback_heuristic",
        )


def get_safe_query(
    original_query: str,
    corrected_query: str,
    *,
    validator: Optional[SpellcheckValidator] = None,
    use_ollama: bool = True,
) -> tuple[str, bool]:
    """
    Удобная функция: возвращает (query_to_use, was_corrupted).

    Args:
        original_query: Исходный user_query
        corrected_query: Результат spellchecker
        validator: Опционально свой экземпляр SpellcheckValidator
        use_ollama: Использовать Ollama для проверки

    Returns:
        (query_to_use, was_corrupted)
    """
    v = validator or SpellcheckValidator()
    result = v.validate(original_query, corrected_query, use_ollama=use_ollama)
    return result.query_to_use, result.was_corrupted
