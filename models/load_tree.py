import joblib
from pathlib import Path
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

@st.cache_resource
def load_tree_model(path: Path) -> DecisionTreeClassifier:
    with open(path, "rb") as fo:
        model = joblib.load(fo, mmap_mode="r")
    assert isinstance(model, DecisionTreeClassifier)
    return model