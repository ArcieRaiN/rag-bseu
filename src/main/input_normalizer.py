import re
from typing import List

from natasha import MorphVocab, Doc
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pymorphy2"
)


# Инициализация Natasha (один раз)
_segmenter = Segmenter()
_emb = NewsEmbedding()
_morph_tagger = NewsMorphTagger(_emb)
_vocab = MorphVocab()

TOKEN_RE = re.compile(r"[а-яёіў]+(?:[-%][а-яёіў]+)*|\d+", re.IGNORECASE)


def normalize_text_lemmatized(text: str) -> str:
    """
    Нормализация для RAG:
    - лемматизация RU/BY
    - сохранение чисел
    - ничего не склеиваем
    """
    if not text:
        return ""

    text = text.lower()

    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return ""

    # разделяем слова и числа
    words = [t for t in tokens if not t.isdigit()]
    numbers = [t for t in tokens if t.isdigit()]

    lemmas: List[str] = []

    if words:
        doc = Doc(" ".join(words))
        doc.segment(_segmenter)
        doc.tag_morph(_morph_tagger)

        for token in doc.tokens:
            token.lemmatize(_vocab)
            if token.lemma:
                lemmas.append(token.lemma)

    # числа добавляем в конец (стабильно)
    return " ".join(lemmas + numbers)
