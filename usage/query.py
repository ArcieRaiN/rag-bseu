"""
USAGE: интерактивный запуск query_pipeline + output_pipeline.

Два режима работы:
- CLI:       python usage/query.py
- Streamlit: streamlit run usage/query.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipelines.query_pipeline import QueryPipeline
from src.pipelines.output_pipeline import OutputPipeline


BASE_DIR = Path(__file__).resolve().parent.parent  # rag-bseu


# ================================================================
# Streamlit detection
# ================================================================

def _is_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# ================================================================
# CLI mode (python usage/query.py)
# ================================================================

def main_cli() -> None:
    pipeline = QueryPipeline(base_dir=BASE_DIR)
    output_pipeline = OutputPipeline(
        output_dir=BASE_DIR / "usage" / "outputs",
    )

    print("Интерактивный режим. Введите запрос (Ctrl+C для выхода).")
    try:
        while True:
            query = input("> ").strip()
            if not query:
                continue

            result = pipeline.run(query)

            if not result.top_chunks:
                print("Ничего не найдено.")
                continue

            print(f"Топ-{min(3, len(result.top_chunks))} чанков:")
            for i, sc in enumerate(result.top_chunks[:3], 1):
                ch = sc.chunk
                meta = []
                if ch.geo:
                    meta.append(f"geo={ch.geo}")
                if ch.years:
                    meta.append(f"years={ch.years}")
                if ch.metrics:
                    meta.append(f"metrics={ch.metrics}")
                if ch.time_granularity:
                    meta.append(f"time={ch.time_granularity}")
                if ch.oked:
                    meta.append(f"oked={ch.oked}")

                print(f"{i}. [source={ch.source}, page={ch.page}, id={ch.id}]")
                if meta:
                    print("   " + "; ".join(meta))
                print(f"   context: {ch.context}")
                print()

            df = output_pipeline.run(result, user_query=query)
            if df is not None:
                print("\n=== ТАБЛИЦА ===")
                print(df.to_string(index=False))
            else:
                print("Не удалось сформировать таблицу.")

    except KeyboardInterrupt:
        print("\nВыход из программы.")


# ================================================================
# Streamlit mode (streamlit run usage/query.py)
# ================================================================

def main_streamlit() -> None:
    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")

    st.set_page_config(page_title="RAG BSEU", layout="wide")
    st.title("RAG BSEU — Статистические данные")

    @st.cache_resource
    def load_pipelines():
        qp = QueryPipeline(base_dir=BASE_DIR)
        op = OutputPipeline(output_dir=BASE_DIR / "usage" / "outputs")
        return qp, op

    def _reset_state():
        if "pipeline_output" not in st.session_state:
            st.session_state["pipeline_output"] = {
                "df": None,
                "title": "",
                "sources": [],
                "error": "",
            }

    _reset_state()
    query_pipeline, output_pipeline = load_pipelines()

    query = st.text_input(
        "Введите запрос",
        placeholder="Например: ВВП Беларуси 2018-2022",
        help="Ключевые слова или фраза по теме: показатель, страна, год. "
             "Примеры: «Численность населения», «Экспорт и импорт 2023», «Цена яблок»",
    )

    all_sources = query_pipeline.get_available_sources()
    selected_source = st.selectbox(
        "Источник",
        all_sources,
        index=None,
        placeholder="Все файлы (по умолчанию)",
    )
    source_filter = selected_source

    if st.button("Найти", type="primary") and query.strip():
        with st.spinner("Поиск по базе знаний..."):
            result = query_pipeline.run(query.strip(), source_filter=source_filter)

        if not result.top_chunks:
            st.warning("Ничего не найдено по запросу.")
            st.session_state["pipeline_output"].update(
                {"df": None, "title": "", "sources": [], "error": "Ничего не найдено."}
            )
        else:
            with st.spinner("Генерация таблицы через LLM..."):
                df = output_pipeline.run(result, user_query=query.strip())

            if df is None:
                st.error("Не удалось сформировать таблицу. Подробности в usage/logs/output_df_fails.json")
                st.session_state["pipeline_output"].update(
                    {"df": None, "title": "", "sources": [], "error": "LLM не вернула корректный JSON."}
                )
            else:
                title = output_pipeline.title or "Результат"
                st.session_state["pipeline_output"].update(
                    {"df": df, "title": title, "sources": output_pipeline.sources, "error": ""}
                )

    ui_state = st.session_state["pipeline_output"]
    if ui_state["df"] is None:
        if ui_state["error"]:
            st.info(ui_state["error"])
        return

    df = ui_state["df"]
    title = ui_state["title"] or "Результат"

    # --- Заголовок ---
    st.subheader(title)

    # --- Seaborn barplot ---
    _render_barplot(df, title)

    # --- Таблица данных ---
    st.markdown("**Таблица данных**")
    st.dataframe(
        df.style.format(_fmt_value),
        use_container_width=True,
    )

    # --- Источники ---
    sources = ui_state["sources"]
    if sources:
        st.markdown(
            "**Источники:** "
            + "; ".join(
                sources
            )
        )

    # --- Скачивание xlsx ---
    xlsx_buffer = io.BytesIO()
    file_name = _sanitize_filename(title)
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    xlsx_buffer.seek(0)

    st.download_button(
        label="Скачать таблицу (.xlsx)",
        data=xlsx_buffer,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _fmt_value(val):
    """Format numbers: no thousands separator, comma for decimals."""
    if isinstance(val, float):
        if val == int(val):
            return str(int(val))
        return f"{val:g}".replace(".", ",")
    if isinstance(val, int):
        return str(val)
    return val


def _sanitize_filename(title: str) -> str:
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip()
    if not safe:
        safe = "output"
    if not safe.lower().endswith(".xlsx"):
        safe = f"{safe}.xlsx"
    return safe


def _render_barplot(df: "pd.DataFrame", title: str) -> None:
    """Seaborn barplot: первый столбец — категории (X), остальные — значения (Y)."""
    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    cols = list(df.columns)
    if len(cols) < 2:
        st.info("Недостаточно столбцов для построения графика.")
        return

    x_col = cols[0]
    value_cols = cols[1:]

    melted = df.melt(id_vars=[x_col], value_vars=value_cols,
                     var_name="Показатель", value_name="Значение")

    melted["Значение"] = pd.to_numeric(melted["Значение"], errors="coerce")
    melted = melted.dropna(subset=["Значение"])

    if melted.empty:
        st.info("Нет числовых данных для графика.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x=x_col, y="Значение", hue="Показатель", ax=ax)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ================================================================
# Entry point
# ================================================================

if _is_streamlit():
    main_streamlit()
elif __name__ == "__main__":
    main_cli()
