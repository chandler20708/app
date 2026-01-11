import streamlit as st
import pandas as pd

def add_title(title: str, subtitle_mk: str = None):
  st.title(title)
  if subtitle_mk:
    st.markdown(subtitle_mk)

WEEKDAY_ORDER = [
    "Mon", "Tue", "Wed",
    "Thu", "Fri", "Sat", "Sun",
]

def order_days(days: list[str]) -> list[str]:
    observed = (
        pd.Series(days)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not observed:
        return []

    # direct canonical match
    canonical = [d for d in WEEKDAY_ORDER if d in observed]
    if canonical:
        return canonical

    # abbreviation match (Mon, Tue, ...)
    abbrev_map = {}
    for v in observed:
        abbrev = v[:3].title()
        abbrev_map.setdefault(abbrev, v)

    abbrev_order = [d[:3] for d in WEEKDAY_ORDER if d[:3] in abbrev_map]
    if abbrev_order:
        return [abbrev_map[a] for a in abbrev_order]

    # final fallback: preserve observed order
    return observed

def card_container():
    return st.markdown(
        """
        <div style="
            background-color: #F9FAFB;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            height: 100%;
        ">
        """,
        unsafe_allow_html=True,
    )

def close_card():
    st.markdown("</div>", unsafe_allow_html=True)
