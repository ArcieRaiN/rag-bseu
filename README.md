# RAG BSEU — Система поиска, анализа и выдачи статистических данных 

RAG-система для работы со статистическими сборниками Национального статистического комитета Республики Беларусь (Белстат). Система парсит PDF-документы, обогащает чанки метаданными через LLM, строит гибридный поисковый индекс (FAISS + BM25) и отвечает на запросы в виде структурированных таблиц и графиков через Streamlit.

## Архитектура

Система состоит из четырёх пайплайнов:

```
1. Загрузка       SiteParser → PDF-файлы в usage/documents/
2. Построение БЗ  PDFChunker → LLM-обогащение → FAISS + data.json
3. Запрос          Regex-обогащение → Hybrid Search (Semantic + BM25 + Metadata) → RRF → Top-K
4. Ответ           Top-K чанков → LLM (Ollama) → JSON → DataFrame → Streamlit
```

### Гибридный поиск (Reciprocal Rank Fusion)

Три независимых канала поиска объединяются через RRF — score-agnostic fusion, не требующий нормализации:

| Канал | Описание | Top-K |
|-------|----------|-------|
| Semantic (FAISS) | Cosine similarity по эмбеддингам `context` (intfloat/multilingual-e5-large, 1024d) | 40 |
| Lexical (BM25) | BM25Okapi по `text + context` | 40 |
| Metadata | Скоринг совпадений `geo`, `metrics`, `years` | 30 |

**RRF**: `score(d) = Σ 1/(K + rank_i)`, K=60. Финальный Top-10 передаётся LLM.

### LLM-генерация ответа

OutputPipeline формирует 4-блочный промпт (роль, запрос, чанки, JSON-инструкция) и отправляет в Ollama. Ответ валидируется по JSON-схеме:

- Таблица всегда в **1NF** (первая нормальная форма) — годы/категории как значения, не заголовки
- Единицы измерения включаются в названия столбцов
- Поддержка `no_data: true` при отсутствии данных (вместо галлюцинаций)
- Указание `source_fragment` для определения использованного источника

### Модули

| Модуль | Назначение |
|--------|-----------|
| `src/core/` | Доменные модели (`Chunk`, `EnrichedQuery`, `ScoredChunk`, `PipelineResult`), конфигурация `RetrievalConfig` |
| `src/ingestion/` | Парсинг сайта (`SiteParser`), чанкинг PDF (`PDFChunker`), маппинг секций (`SectionMapper`) |
| `src/enrichers/` | HTTP-клиент Ollama (`ollama_client`), LLM-обогащение чанков (`llm_enricher`) |
| `src/retrieval/` | `FaissSemanticSearcher`, `BM25Search`, `MetadataScorer`, `HybridSearcher` (RRF), `CrossEncoderReranker` |
| `src/vectorstore/` | Эмбеддинги (`SentenceVectorizer`), хранение (`FAISSStore`) |
| `src/pipelines/` | Оркестрация: `parse_documents`, `knowledge_base_builder`, `query`, `output` |
| `src/utils/` | `OutputValidator`, `ChunkValidator`, `RAGLogger` (JSONL), постобработка |
| `tests/` | Тестовая база (182 вопроса), `evaluator.py` (Hit@k, MRR), `compare_embeddings.py`, скрипты запуска |
| `usage/` | Точки входа: `query.py` (CLI + Streamlit), `cli.py`, скрипты загрузки и построения |

## Стек технологий

| Категория | Технология |
|-----------|-----------|
| Язык | Python 3.12 |
| LLM | Ollama (локальный inference, JSON mode) |
| Эмбеддинги | sentence-transformers (`intfloat/multilingual-e5-large`, 1024d) |
| Векторный поиск | FAISS (`IndexFlatIP` + L2-нормализация → cosine similarity) |
| Лексический поиск | rank-bm25 (`BM25Okapi`) |
| Парсинг PDF | LlamaIndex (`PDFReader`) |
| Веб-интерфейс | Streamlit |
| Визуализация | Seaborn, Matplotlib |
| Данные | Pandas, openpyxl |

## Запуск

### 1. Установка

```bash
git clone <url> && cd rag-bseu
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 2. Ollama

```bash
ollama pull llama3-chatqa:latest
ollama serve
```

### 3. Загрузка документов

```bash
python usage/parse_documents.py
```

### 4. Построение базы знаний

```bash
python usage/prepare_vector_store.py
```

### 5. Запуск

```bash
# Streamlit (таблицы + графики + фильтр по источнику)
streamlit run usage/query.py
```
```bash
# CLI
python usage/query.py
```

### 6. Пересборка индекса (без LLM)

```bash
python usage/rebuild_index.py
```

Пересоздаёт FAISS-индекс из существующего `data.json` без повторного LLM-обогащения.

### 7. Тестирование

```bash
# Полный прогон (182 вопроса)
python -m tests.evaluator

# Быстрый прогон (10 вопросов)
python -m tests.evaluator --quick

# Фильтр по категории
python -m tests.evaluator --category prices

# Сравнение embedding-моделей
python -m tests.compare_embeddings
```

Результаты сохраняются в `tests/results/`. Сводный журнал для диплома и повторных замеров: `tests/test_results.md`. Запуск из произвольного каталога: `python tests/run_evaluator.py`.

## Результаты оценки

### v2 (44 вопроса) — исходный набор (`paraphrase-multilingual-MiniLM-L12-v2`, 384d)

| Метрика | Значение |
|---------|----------|
| Hit@1 | 50.0% |
| Hit@3 | 88.6% |
| Hit@5 | 95.5% |
| MRR | 0.693 |
| Avg time | 0.012s |

### v3 (182 вопроса) — расширенный набор (`intfloat/multilingual-e5-large`, 1024d)

| Метрика | Значение |
|---------|----------|
| Hit@1 | 33.5% |
| Hit@3 | 46.2% |
| Hit@5 | 59.3% |
| MRR | 0.432 |
| Avg time | 0.08s |

**По категориям (v3 Hit@5):**

| Категория | Hit@5 | Комментарий |
|-----------|-------|------------|
| tourism | 100% (14/14) | Отличное покрытие |
| trade | 80% (12/15) | Отличное покрытие |
| prices | 76% (16/21) | Хорошо |
| agriculture | 75% (9/12) | Хорошо |
| comparison | 75% (6/8) | Хорошо |
| social | 65% (17/26) | Приемлемо |
| transport | 62% (5/8) | Приемлемо |
| economy | 54% (20/37) | Требует улучшения |
| demographics | 26% (6/23) | Слабо |
| environment | 25% (2/8) | Слабо |
| sdg | 10% (1/10) | Критично |

### Сравнение embedding-моделей (v3, 182 вопроса)

| Модель | Dim | Hit@1 | Hit@3 | Hit@5 | MRR | Embed | Query |
|--------|-----|-------|-------|-------|-----|-------|-------|
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 26.9% | 45.6% | 53.8% | 0.385 | 3.5s | 0.011s |
| BAAI/bge-m3 | 1024 | 29.7% | 46.7% | 58.8% | 0.414 | 89.6s | 0.065s |
| deepvk/USER-bge-m3 | 1024 | 30.8% | 44.5% | 58.2% | 0.415 | 89.3s | 0.076s |
| **intfloat/multilingual-e5-large** | **1024** | **33.5%** | **46.2%** | **59.3%** | **0.432** | 90.5s | 0.076s |

## Структура проекта

```
rag-bseu/
├── src/
│   ├── core/               # Доменные модели, конфигурация, обогащение запросов
│   ├── enrichers/           # ollama_client, llm_enricher, parsers
│   ├── ingestion/           # Парсинг PDF, чанкинг, маппинг секций
│   ├── pipelines/           # query, output, knowledge_base_builder, parse_documents
│   ├── retrieval/           # FAISS, BM25, metadata scoring, RRF, reranker
│   ├── utils/               # chunk_validator, output_validator, logger, post_processor
│   └── vectorstore/         # SentenceVectorizer, FAISSStore
├── tests/
│   ├── test_data.json       # 182 тестовых вопроса (v3)
│   ├── evaluator.py         # Автоматическая оценка retrieval
│   ├── compare_embeddings.py # Сравнение embedding-моделей
│   ├── run_evaluator.py     # Запуск evaluator из произвольного каталога
│   ├── run_rebuild_index.py # Запуск rebuild_index из произвольного каталога
│   └── results/             # JSON-результаты прогонов
├── usage/
│   ├── documents/           # Исходные PDF-сборники
│   ├── vector_store/        # FAISS-индекс + data.json
│   ├── outputs/             # JSON-ответы LLM
│   ├── logs/                # JSONL-логи ошибок
│   ├── query.py             # Streamlit UI + CLI
│   ├── rebuild_index.py     # Пересборка FAISS без LLM-обогащения
│   └── cli.py               # Единый CLI для всех пайплайнов
├── requirements.txt
├── setup.py
└── LICENSE
```

## Автор

**Александр Лебедев**

## Лицензия

Apache License 2.0 — см. [LICENSE](LICENSE).
