import re
from typing import List

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pymorphy2"
)

from natasha import MorphVocab, Doc
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger


# Инициализация Natasha (один раз на процесс)
_segmenter = Segmenter()
_emb = NewsEmbedding()
_morph_tagger = NewsMorphTagger(_emb)
_vocab = MorphVocab()

TOKEN_RE = re.compile(r"[а-яёіў]+(?:[-%][а-яёіў]+)*|\d+", re.IGNORECASE)


def normalize_text_lemmatized(text: str) -> str:
    """
    Нормализация текста для RAG:
    - лемматизация RU/BY
    - сохранение чисел
    - стабильный порядок токенов
    """
    if not text:
        return ""

    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return ""

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

    # ВАЖНО: числа добавляем в конец — стабильность для lexical search
    return " ".join(lemmas + numbers)


def normalize_query(text: str) -> str:
    """
    Семантический alias для query pipeline.
    """
    return normalize_text_lemmatized(text)
