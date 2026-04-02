"""
Пакет src — ядро RAG-системы.

Модули:
  core         Доменные модели (Chunk, ScoredChunk, EnrichedQuery) и конфигурация
  enrichers    HTTP-клиент Ollama, LLM-обогащение чанков метаданными
  ingestion    Парсинг PDF, чанкинг по страницам, маппинг секций
  pipelines    Оркестрация: QueryPipeline, OutputPipeline, KnowledgeBaseBuilder
  retrieval    Гибридный поиск (FAISS + BM25 + Metadata → RRF), reranker
  utils        Валидация, логирование, постобработка
  vectorstore  Генерация эмбеддингов (sentence-transformers), FAISS-хранилище
"""
