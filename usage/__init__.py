"""
Пакет usage — точки входа для запуска RAG-системы.

  cli.py                  Единый CLI для всех пайплайнов
  parse_documents.py      Скачивание документов с Белстата
  prepare_vector_store.py Построение базы знаний и FAISS-индекса
  query.py                RAG-запрос (CLI + Streamlit)
  rebuild_index.py        Пересборка FAISS-индекса без LLM-обогащения
"""
