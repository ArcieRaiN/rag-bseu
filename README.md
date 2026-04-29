# RAG BSEU — Система поиска, анализа и выдачи статистических данных 

RAG-система для работы со статистическими сборниками Национального статистического комитета Республики Беларусь (Белстат). Система парсит PDF-документы, обогащает чанки метаданными через LLM, строит гибридный поисковый индекс (FAISS + BM25) и отвечает на запросы в виде структурированных таблиц и графиков через Streamlit.

## Архитектура

Система состоит из четырёх пайплайнов:

```
1. Загрузка       SiteParser → PDF-файлы в usage/documents/
2. Построение БЗ  PDFChunker → RuleMetadataExtractor → LLM-обогащение → FAISS + data.json
3. Запрос          Regex/rule-обогащение → Hybrid Search (Semantic + BM25 + Metadata) → RRF → Top-K
4. Ответ           Top-K чанков → LLM (Ollama) → JSON → DataFrame → Streamlit
```

### Гибридный поиск (Reciprocal Rank Fusion)

Три независимых канала поиска объединяются через RRF — score-agnostic fusion, не требующий нормализации:

| Канал | Описание | Top-K |
|-------|----------|-------|
| Semantic (FAISS) | Cosine similarity по эмбеддингам `search_context + text` (intfloat/multilingual-e5-large, 1024d) | 40 |
| Lexical (BM25) | BM25Okapi по `search_context + text` | 40 |
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
| `src/enrichers/` | HTTP-клиент Ollama (`ollama_client`), rule-based и LLM-обогащение чанков (`rule_metadata_extractor`, `llm_enricher`) |
| `src/retrieval/` | `FaissSemanticSearcher`, `BM25Search`, `MetadataScorer`, `HybridSearcher` (RRF), `CrossEncoderReranker` |
| `src/vectorstore/` | Эмбеддинги (`SentenceVectorizer`), хранение (`FAISSStore`) |
| `src/pipelines/` | Оркестрация: `parse_documents`, `knowledge_base_builder`, `query`, `output` |
| `src/utils/` | `OutputValidator`, `ChunkValidator`, `RAGLogger` (JSONL), постобработка |
| `tests/` | Набор тестов v4, `retrieval_eval.py`, `run_v4_experiments.py`, `run_v4_ablation_experiments.py`, `validate_benchmark.py` |
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
# Проверка набора тестов и согласованности с data.json
python -m tests.validate_benchmark

# Полный прогон извлечения (1500 вопросов; долго)
python -m tests.run_v4_experiments

# Быстрая выборка
python -m tests.run_v4_experiments --limit 50

# Ablation: вклад каналов метаданных (тот же индекс)
python -m tests.run_v4_ablation_experiments

# Покрытие полей обогащения
python -m tests.enrichment_quality
```

Результаты: `tests/results/v4/`, сводка в `tests/test_results.md` и `reports/v4/`.

## Результаты оценки

### Набор тестов (1500 вопросов, 12 PDF)

База знаний в `usage/vector_store` (2974 чанка). Набор тестов: `tests/benchmarks/benchmark_v4.json`. Схема чанка — поле `search_context` (поле `context` удалено).

| Метрика | Значение |
|---------|----------|
| Hit@1 | 36.4% |
| Hit@3 | 53.5% |
| Hit@5 | 61.3% |
| MRR | 0.469 |
| Avg time | 0.146s |

Покрытие ключевых полей: `section` 98.42%, `geo` 99.66%, `metrics` 100.00%, `units` 74.55%, `years` 99.19%, `search_context` 100.00%.

Подробности ablation и таймингов построения индекса — в `reports/v4/v4_experiment_report.md` и `tests/test_results.md`.

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
│   ├── benchmarks/         # benchmark_v4.json
│   ├── retrieval_eval.py   # Hit@k / MRR для JSON-наборов
│   ├── run_v4_experiments.py
│   ├── run_v4_ablation_experiments.py
│   ├── validate_benchmark.py
│   ├── generate_benchmark_v4.py
│   ├── run_rebuild_index.py
│   └── results/v4/          # JSON-результаты прогонов
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
