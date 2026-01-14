import datetime as dt
from typing import List, Dict

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

from utils import add_title, load_log, seed_everything


PAGE_NAME = "Audit Log"


def filter_logs(logs: List[Dict]) -> pd.DataFrame:
    if not logs:
        return pd.DataFrame(columns=["timestamp", "page", "action", "parameters"])
    df = pd.DataFrame(logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def render_filters(df: pd.DataFrame):
    pages = ["All"] + sorted(df["page"].dropna().unique().tolist())
    actions = ["All"] + sorted(df["action"].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        page_filter = st.selectbox("Page", pages, index=0)
    with col2:
        action_filter = st.selectbox("Action type", actions, index=0)

    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    default_start = min_date.date() if pd.notna(min_date) else dt.date.today()
    default_end = max_date.date() if pd.notna(max_date) else dt.date.today()
    start, end = st.date_input(
        "Time range",
        value=(default_start, default_end),
    )
    mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
    if page_filter != "All":
        mask &= df["page"] == page_filter
    if action_filter != "All":
        mask &= df["action"] == action_filter
    return df[mask]


def render_download(df: pd.DataFrame):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export log (CSV)",
        data=csv_bytes,
        file_name="change_log_export.csv",
        mime="text/csv",
        width='stretch',
    )


def main():
    seed_everything()
    add_title(PAGE_NAME)
    logs = load_log()
    df = filter_logs(logs)

    if df.empty:
        st.info("No log entries yet.")
        return

    filtered = render_filters(df)
    filtered = filtered.sort_values("timestamp", ascending=False)
    render_download(filtered)
    st.dataframe(
        filtered,
        width='stretch',
        hide_index=True,
        column_config={
            "timestamp": st.column_config.Column(width="small"),
            "page": st.column_config.Column(width="small"),
            "action": st.column_config.Column(width="small"),
            "parameters": st.column_config.Column(width="large"),
        },
    )


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
    with open("./international_coal/app/plerdy.html") as f:
      html_string = f.read()
      html(html_string)