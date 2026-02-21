"""
Пакет pipelines -- оркестрация этапов RAG-пайплайна.

Пайплайны:
- ParseDocumentsPipeline: скачивание PDF с сайта Белстата
- KnowledgeBaseBuilder: построение базы знаний (PDF -> Chunks -> FAISS)
- QueryPipeline: обработка запроса (spellcheck -> enrichment -> hybrid search)
- OutputPipeline: генерация табличного вывода через LLM
"""
