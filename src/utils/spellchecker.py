"""
Проверка орфографии пользовательских запросов.

Использует pyspellchecker для исправления опечаток в русскоязычных запросах.
Поддерживает кастомный словарь терминов, которые не должны исправляться.
"""

from spellchecker import SpellChecker
from typing import Set

CUSTOM_WORDS: Set[str] = set()


class QuerySpellChecker:
    """
    Исправляет опечатки в пользовательских запросах.

    Слова из ``CUSTOM_WORDS`` не подвергаются исправлению.
    """

    def __init__(self) -> None:
        self.spell = SpellChecker(language='ru')  # для русского языка
        # Подгружаем кастомные слова
        self.spell.word_frequency.load_words(CUSTOM_WORDS)
        self.custom_words: Set[str] = CUSTOM_WORDS

    def correct_query(self, query: str) -> str:
        """
        Исправляет опечатки в строке запроса, кроме custom_words.

        Args:
            query: пользовательский запрос

        Returns:
            исправленный запрос
        """
        tokens = query.split()
        corrected_tokens = []
        for token in tokens:
            if token in self.custom_words:
                corrected_tokens.append(token)
            else:
                corrected_tokens.append(self.spell.correction(token) or token)
        return " ".join(corrected_tokens)
