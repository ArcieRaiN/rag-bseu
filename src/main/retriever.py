import json
import re
from pathlib import Path
from typing import List, Dict, Set
import numpy as np

from src.main.vectorizer import SentenceVectorizer
from src.main.input_normalizer import normalize_text_lemmatized


class SemanticRetriever:
    """
    Гибридный поиск: векторный + текстовый по ключевым словам
    """

    def __init__(
        self,
        vectorizer: SentenceVectorizer,
        data_path: Path,
    ):
        self.vectorizer = vectorizer
        self.data_path = Path(data_path)

        self.data = self._load_data()

        # берём ГОТОВУЮ нормализацию из data.json
        self.texts = [item["normalized"] for item in self.data]
        self.titles = [item.get("title", "") for item in self.data]
        self.full_texts = [item.get("text", "") for item in self.data]

        self.embeddings = np.stack(
            [self.vectorizer.embed(t) for t in self.texts]
        ).astype(np.float32)

    def _load_data(self) -> List[Dict]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        query = query / (np.linalg.norm(query) + 1e-9)
        matrix = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        )
        return matrix @ query

    def _expand_query_tokens(self, query_tokens: Set[str]) -> Set[str]:
        """
        Расширяет запрос синонимами и связанными терминами.
        """
        expanded = set(query_tokens)
        
        # Словарь синонимов и связанных терминов
        synonyms = {
            'детсад': {'дошкольн', 'детский', 'садик', 'дошкольный'},
            'дошкольн': {'детсад', 'детский', 'садик'},
            'детский': {'дошкольн', 'детсад'},
            'учрежден': {'организац', 'заведен'},
            'организац': {'учрежден'},
            'число': {'количеств', 'численност'},
            'количеств': {'число', 'численност'},
            'численност': {'число', 'количеств'},
            'нефть': {'нефтян', 'нефтедобыч'},
            'беларусь': {'белорус', 'рб'},
            'минск': {'минский'},
            'человек': {'населен', 'жител'},
            'населен': {'человек', 'жител'},
        }
        
        for token in list(query_tokens):
            if token in synonyms:
                expanded.update(synonyms[token])
        
        return expanded

    def _text_search_score(self, query_tokens: Set[str], normalized_text: str, title: str, full_text: str) -> float:
        """
        Улучшенный текстовый поиск по ключевым словам с расширением запроса.
        Возвращает score на основе совпадений токенов.
        """
        if not query_tokens:
            return 0.0
        
        # Расширяем запрос синонимами
        expanded_tokens = self._expand_query_tokens(query_tokens)
        
        # Нормализуем тексты для поиска
        normalized_title = normalize_text_lemmatized(title).lower()
        normalized_full_text = normalize_text_lemmatized(full_text).lower()
        normalized_text_lower = normalized_text.lower()
        
        # Считаем совпадения в заголовке (более важны) - используем расширенные токены
        title_matches = sum(1 for token in expanded_tokens if token in normalized_title)
        text_matches = sum(1 for token in expanded_tokens if token in normalized_text_lower)
        full_text_matches = sum(1 for token in expanded_tokens if token in normalized_full_text)
        
        # Также считаем точные совпадения (более важны)
        title_exact = sum(1 for token in query_tokens if token in normalized_title)
        text_exact = sum(1 for token in query_tokens if token in normalized_text_lower)
        full_text_exact = sum(1 for token in query_tokens if token in normalized_full_text)
        
        # Взвешенная оценка: точные совпадения важнее, заголовок важнее текста
        exact_score = (title_exact * 5.0 + text_exact * 3.0 + full_text_exact * 2.0) / len(query_tokens)
        expanded_score = (title_matches * 2.0 + text_matches * 1.5 + full_text_matches * 1.0) / len(expanded_tokens)
        
        # Комбинируем с приоритетом точных совпадений
        score = exact_score * 2.0 + expanded_score * 0.5
        
        # Бонус за полное совпадение всех токенов в заголовке
        if title_exact == len(query_tokens):
            score *= 3.0
        elif title_matches >= len(query_tokens) * 0.8:
            score *= 2.0
        elif text_exact == len(query_tokens):
            score *= 2.0
        
        # Бонус за наличие чисел в тексте (для статистических запросов)
        if any(token.isdigit() for token in query_tokens):
            if re.search(r'\d+', full_text):
                score *= 1.2
        
        return min(score, 20.0)  # Увеличиваем максимальный score

    def _is_statistical_query(self, query: str, query_tokens: Set[str]) -> bool:
        """
        Определяет, является ли запрос статистическим (требует конкретных чисел).
        """
        statistical_keywords = {
            'число', 'количеств', 'численност', 'сколько', 'сколько',
            'населен', 'человек', 'жител', 'ввп', 'производств',
            'экспорт', 'импорт', 'объем', 'добыч', 'тонн', 'тонна',
            'поголовь', 'урожайност', 'площад', 'стоимость',
            'заработн', 'доход', 'расход', 'инвестиц', 'бюджет',
            'долг', 'инфляц', 'индекс', 'цена', 'стоимость'
        }
        return bool(query_tokens & statistical_keywords) or any(kw in query.lower() for kw in ['число', 'сколько', 'численность', 'объем', 'производство'])

    def _has_numbers(self, text: str) -> bool:
        """Проверяет наличие чисел в тексте."""
        return bool(re.search(r'\d+[\d\s.,]*', text))

    def _is_noise(self, text: str, title: str) -> bool:
        """
        Определяет, является ли фрагмент шумом (методики, описания без чисел).
        """
        noise_keywords = [
            'методик', 'расчет', 'методолог', 'примечан', 'сноск',
            'источник данн', 'использован', 'приведен', 'учитыва',
            'статистический комитет', 'национальный статистический',
            'редакционн', 'содержан', 'оглавлен'
        ]
        text_lower = (text + " " + title).lower()
        # Если есть шумные ключевые слова И нет чисел - это шум
        if any(keyword in text_lower for keyword in noise_keywords):
            if not self._has_numbers(text):
                return True
        return False

    def search(self, query: str, top_k: int = 5, hybrid_weight: float = 0.5) -> List[Dict]:
        """
        Гибридный поиск: комбинация векторного и текстового поиска.
        С улучшенной фильтрацией для статистических запросов.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            hybrid_weight: Вес векторного поиска (0.0-1.0), остальное - текстовый
        """
        if not query.strip():
            return []

        normalized_query = normalize_text_lemmatized(query)
        if not normalized_query:
            return []

        query_tokens = set(normalized_query.split())
        is_statistical = self._is_statistical_query(query, query_tokens)

        # Векторный поиск
        query_vec = self.vectorizer.embed(normalized_query).astype(np.float32)
        vector_sims = self._cosine_similarity(query_vec)
        
        # Текстовый поиск по ключевым словам
        text_scores = np.array([
            self._text_search_score(query_tokens, self.texts[i], self.titles[i], self.full_texts[i])
            for i in range(len(self.data))
        ])
        
        # Бонус за наличие чисел для статистических запросов
        if is_statistical:
            for i in range(len(self.data)):
                if self._has_numbers(self.full_texts[i]):
                    text_scores[i] *= 1.5  # Увеличиваем score для результатов с числами
        
        # Нормализуем текстовые scores (0-1)
        if text_scores.max() > 0:
            text_scores = text_scores / text_scores.max()
        
        # Нормализуем векторные scores (0-1)
        if vector_sims.max() > vector_sims.min():
            vector_sims_normalized = (vector_sims - vector_sims.min()) / (vector_sims.max() - vector_sims.min() + 1e-9)
        else:
            vector_sims_normalized = vector_sims
        
        # Комбинируем scores
        combined_scores = hybrid_weight * vector_sims_normalized + (1 - hybrid_weight) * text_scores
        
        # Берем больше результатов для фильтрации
        top_idx = np.argsort(combined_scores)[::-1][:top_k * 3]
        
        results = []
        seen_texts = set()  # Для дедупликации
        
        for idx in top_idx:
            item = self.data[idx]
            text = item.get("text", "")
            title = item.get("title", "")
            
            # Пропускаем шумные результаты
            if self._is_noise(text, title):
                continue
            
            # Для статистических запросов требуем наличие чисел
            if is_statistical and not self._has_numbers(text):
                continue
            
            # Дедупликация по первым 100 символам текста
            text_hash = hash(text[:100])
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)
            
            # Используем комбинированный score
            final_score = float(combined_scores[idx])
            
            results.append(
                {
                    "score": final_score,
                    "text": text,
                    "title": title,
                    "source": item.get("source", ""),
                    "page": item.get("page", 0),
                    "vector_score": float(vector_sims[idx]),
                    "text_score": float(text_scores[idx]),
                    "has_numbers": self._has_numbers(text),
                }
            )
            
            if len(results) >= top_k:
                break

        return results


class TableRetriever(SemanticRetriever):
    """
    Compatibility retriever for RagPipeline/tests.
    Returns table rows (if present in data.json) under key 'table'.
    """

    def search(self, query: str, top_k: int = 5, hybrid_weight: float = 0.5) -> List[Dict]:
        base = super().search(query, top_k=top_k, hybrid_weight=hybrid_weight)
        out: List[Dict] = []
        for hit in base:
            # locate original item by matching (source,page,text) (good enough for tests)
            # Note: base hits were created from self.data items in order; but we don't retain id.
            # We'll best-effort re-find.
            matched = None
            for item in self.data:
                if item.get("source") == hit.get("source") and item.get("page") == hit.get("page") and item.get("text") == hit.get("text"):
                    matched = item
                    break
            rows = matched.get("rows") if matched else None
            out.append(
                {
                    "score": hit.get("score"),
                    "title": hit.get("title", ""),
                    "table": rows or [],
                    "source": hit.get("source", ""),
                    "page": hit.get("page", 0),
                }
            )
        return out
