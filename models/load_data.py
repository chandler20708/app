import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data
def load_aed_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)