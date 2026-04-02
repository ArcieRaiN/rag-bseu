"""
Пакет pipelines — оркестрация этапов RAG-пайплайна.

  parse_documents     Скачивание PDF с сайта Белстата
  knowledge_base_builder  Построение базы знаний (PDF → Chunks → FAISS)
  query               Обработка запроса (enrichment → hybrid search → Top-K)
  output              LLM-генерация табличного вывода → валидация → DataFrame
"""
