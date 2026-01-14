"""Streamlit page for AED analytics outputs."""

from typing import Dict
import os
from pathlib import Path
import sys

import streamlit as st
from streamlit.components.v1 import html

from models.aed_insights.config import AEDConfig
from models.aed_insights.registry import AnalyticsRegistry

def _ensure_libomp_path() -> None:
    if sys.platform != "darwin":
        return
    candidates = [
        Path("/opt/homebrew/opt/libomp/lib"),
        Path("/usr/local/opt/libomp/lib"),
    ]
    for path in candidates:
        if (path / "libomp.dylib").exists():
            current = os.environ.get("DYLD_LIBRARY_PATH", "")
            parts = [p for p in current.split(":") if p]
            if str(path) not in parts:
                parts.insert(0, str(path))
                os.environ["DYLD_LIBRARY_PATH"] = ":".join(parts)
            break


@st.cache_resource
def _load_registry() -> AnalyticsRegistry:
    return AnalyticsRegistry(config=AEDConfig())


@st.cache_resource
def _load_artifacts(xgb_key: str) -> Dict[str, object]:
    # Force Streamlit cache to depend on XGBoost availability/version
    _ = xgb_key
    registry = _load_registry()
    return registry.all_artifacts()

def _xgb_cache_key() -> str:
    _ensure_libomp_path()
    try:
        import xgboost  # type: ignore
    except Exception as exc:
        return f"missing:{exc.__class__.__name__}"
    return f"ok:{xgboost.__version__}"

def _render_tables(tables: Dict[str, object], title: str) -> None:
    st.subheader(title)
    for name, table in tables.items():
        st.markdown(f"**{name.replace('_', ' ').title()}**")
        # if AED_DEBUG and hasattr(table, "columns") and "Model" in table.columns:
        #     st.write({"shape": getattr(table, "shape", None), "name": name})
        #     st.write(table[table["Model"].astype(str).str.contains("XGB", na=False)])
        st.dataframe(table, width='stretch')


def _render_figures(figures: Dict[str, object], title: str) -> None:
    st.subheader(title)
    for name, fig in figures.items():
        st.markdown(f"**{name.replace('_', ' ').title()}**")
        st.pyplot(fig, clear_figure=False, width='stretch')


def _render_tables_compact(tables: Dict[str, object], title: str) -> None:
    st.subheader(title)
    items = list(tables.items())
    for idx in range(0, len(items), 2):
        row_cols = st.columns([2,1])
        for col_idx, target in enumerate(items[idx : idx + 2]):
            name, table = target
            with row_cols[col_idx]:
                st.markdown(f"**{name.replace('_', ' ').title()}**")
                st.dataframe(table, width='stretch')


def _render_table_item(name: str, table: object) -> None:
    st.markdown(f"**{name.replace('_', ' ').title()}**")
    st.dataframe(table, width='stretch')


def _render_figures_expanders(
    figures: Dict[str, object],
    title: str,
    open_count: int = 3,
) -> None:
    st.subheader(title)
    items = list(figures.items())
    for idx in range(0, len(items), 2):
        row_cols = st.columns(2)
        for col_idx, (name, fig) in enumerate(items[idx : idx + 2]):
            label = name.replace("_", " ").title()
            with row_cols[col_idx]:
                with st.expander(label, expanded=(idx + col_idx) < open_count):
                    st.pyplot(fig, clear_figure=False, width='stretch')

def main() -> None:
    st.set_page_config(page_title="AED Insights", layout="wide")
    st.title("AED Insights")
    st.markdown(
        "Descriptive and inferential analyses are based on a reproducible random sample of 400 patients. "
        "Predictive modelling is conducted using the full AED dataset to maximise statistical power and stability."
    )

    xgb_key = _xgb_cache_key()
    artifacts = _load_artifacts(xgb_key)

    descriptive = artifacts["descriptive"]
    inference = artifacts["inference"]
    modelling = artifacts["modelling"]

    descriptive_tab, inference_tab, modelling_tab = st.tabs(
        ["Descriptive", "Inference", "Modelling"]
    )

    with descriptive_tab:
        tables = dict(descriptive.tables)
        categorical_hrg = tables.pop("categorical_hrg", None)
        categorical_investigation = tables.pop("categorical_investigation_cnt_cluster", None)
        breachornot = tables.pop("categorical_breachornot", None)
        breach_day = tables.pop("breach_rate_by_dayofweek", None)
        breach_period = tables.pop("breach_rate_by_period", None)
        _render_tables_compact(tables, "Descriptive Tables")
        if categorical_investigation is not None or categorical_hrg is not None:
            left, right = st.columns([2, 1])
            with left:
                if categorical_investigation is not None:
                    _render_table_item("categorical_investigation_cnt_cluster", categorical_investigation)
            with right:
                if categorical_hrg is not None:
                    _render_table_item("categorical_hrg", categorical_hrg)
        if breachornot is not None:
            _render_table_item("categorical_breachornot", breachornot)
        if breach_day is not None or breach_period is not None:
            left, right = st.columns([2, 1])
            with left:
                if breach_day is not None:
                    _render_table_item("breach_rate_by_dayofweek", breach_day)
            with right:
                if breach_period is not None:
                    _render_table_item("breach_rate_by_period", breach_period)
        _render_figures_expanders(descriptive.figures, "Descriptive Figures")

    with inference_tab:
        _render_tables(inference.tables, "Inference Tables")

    with modelling_tab:
        _render_tables(modelling.tables, "Modelling Tables")


if __name__ == "__main__":
    main()
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.9em; color: gray;'>
            © 2025 <b>Chia-Te Liu</b>. All rights reserved.  
            Made with ❤️ using Streamlit. Backend built by:  
            <a href='https://www.linkedin.com/in/chia-te-liu/' target='_blank'>Chia-Te Liu</a>
            <a href='https://www.linkedin.com/in/ashmi-fathima/' target='_blank'>Ashmi Fathima</a>
            <a href='https://www.linkedin.com/in/%E8%96%87%E5%AE%89-%E9%99%B3-72531b29a/' target='_blank'>Wei-An Chen</a>
            <a href='https://www.linkedin.com/in/akash-somasundaran0713/' target='_blank'>Akash Somasundaran</a>
            <a href='https://www.linkedin.com/in/qutaybah-alowaifeer-88953a150/' target='_blank'>Qutaybah Alowaifeer</a>
            <a href='https://www.linkedin.com/in/raahimsohail/' target='_blank'>Muhammad Raahim Sohail</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    with open("./plerdy.html") as f:
      html_string = f.read()
      html(html_string)