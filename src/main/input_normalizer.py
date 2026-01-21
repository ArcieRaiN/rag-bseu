import re
from typing import List

from natasha import MorphVocab, Doc
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger

# Инициализация Natasha (один раз)
_segmenter = Segmenter()
_emb = NewsEmbedding()
_morph_tagger = NewsMorphTagger(_emb)
_vocab = MorphVocab()

# русские + белорусские буквы
WORD_RE = re.compile(r"[а-яёіў]+", re.IGNORECASE)


def normalize_text_lemmatized(text: str) -> str:
    """
    Приводит текст к леммам.
    НИЧЕГО не выбрасывает кроме мусора и цифр.
    """
    if not text:
        return ""

    text = text.lower()

    # извлекаем слова вручную (важно!)
    words = WORD_RE.findall(text)
    if not words:
        return ""

    doc = Doc(" ".join(words))
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)

    lemmas: List[str] = []
    for token in doc.tokens:
        if not token.pos:
            continue
        token.lemmatize(_vocab)
        if token.lemma:
            lemmas.append(token.lemma)

    return " ".join(lemmas)
