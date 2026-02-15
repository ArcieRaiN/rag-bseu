from spellchecker import SpellChecker
from typing import Set

# Сюда прописываем ваши специфичные слова, которые не должны исправляться
CUSTOM_WORDS = {
    # "BSEU",
    # добавляйте сюда всё, что нужно
}

class QuerySpellChecker:
    def __init__(self):
        """
        Инициализация spellchecker с кастомными словами.
        """
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
