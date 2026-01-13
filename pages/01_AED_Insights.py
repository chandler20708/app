"""Streamlit page for AED analytics outputs."""

from typing import Dict

import streamlit as st

from models.aed_insights.config import AEDConfig
from models.aed_insights.registry import AnalyticsRegistry


@st.cache_resource
def _load_registry() -> AnalyticsRegistry:
    return AnalyticsRegistry(config=AEDConfig())


@st.cache_resource
def _load_artifacts() -> Dict[str, object]:
    registry = _load_registry()
    return registry.all_artifacts()


def _render_tables(tables: Dict[str, object], title: str) -> None:
    st.subheader(title)
    for name, table in tables.items():
        st.markdown(f"**{name.replace('_', ' ').title()}**")
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

    artifacts = _load_artifacts()

    descriptive = artifacts["descriptive"]
    inference = artifacts["inference"]
    modelling = artifacts["modelling"]

    descriptive_tab, inference_tab, modelling_tab = st.tabs(
        ["Descriptive", "Inference", "Modelling"]
    )

    with descriptive_tab:
        tables = dict(descriptive.tables)
        categorical_hrg = tables.pop("categorical_hrg", None)
        breachornot = tables.pop("categorical_breachornot", None)
        breach_day = tables.pop("breach_rate_by_dayofweek", None)
        breach_period = tables.pop("breach_rate_by_period", None)
        _render_tables_compact(tables, "Descriptive Tables")
        if categorical_hrg is not None or breachornot is not None:
            left, right = st.columns([2, 1])
            with left:
                if categorical_hrg is not None:
                    _render_table_item("categorical_hrg", categorical_hrg)
            with right:
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
