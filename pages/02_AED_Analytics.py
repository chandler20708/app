import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from pathlib import Path
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
import re

from models import load_aed_data, load_tree_model, extract_rules
from utils import format_rule_for_management

COLOR_BREACH = "#F1948A"      # soft red
COLOR_NON_BREACH = "#AED6F1"  # soft blue


def init_session():
    defaults = {
        "aed_loaded": False,
        "aed_rules": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def prepare_aed_analytics():
    TREE_PATH = Path("./models/tree_to_plot.joblib")
    DATA_PATH = Path("data/processed_data.parquet")

    tree_model = load_tree_model(TREE_PATH)
    df = load_aed_data(DATA_PATH)

    if df.empty:
        return None

    X = pd.concat(
        [
            df[["noofinvestigation"]],
            pd.get_dummies(df[["HRG_grp"]], drop_first=True),
        ],
        axis=1,
    )

    rules = extract_rules(
        tree_model,
        feature_names=X.columns,
        class_names=["Non-breach", "Breach"],
    )

    high_risk_rules = sorted(
        [r for r in rules if r["breach_rate"] >= 0.5],
        key=lambda x: x["breach_rate"],
        reverse=True,
    )

    return {
        "tree_model": tree_model,
        "feature_names": X.columns,
        "rules": high_risk_rules,
    }


def render_title_section(container):
    with container:
        st.title("AED Breach Risk Analytics")
        st.write(
            "This page summarises key patterns associated with breaches of the "
            "4-hour AED target, using an interpretable decision tree."
        )

def proportion_bar(breach_pct, width=90):
    breach_width = int(breach_pct)
    non_breach_width = width - breach_width

    return f"""
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" WIDTH="{width}">
      <TR>
        <TD WIDTH="{breach_width}" BGCOLOR="#E74C3C"></TD>
        <TD WIDTH="{non_breach_width}" BGCOLOR="#5DADE2"></TD>
      </TR>
    </TABLE>
    """

@st.cache_resource
def render_tree_plot(_analytics):
    tree = _analytics["tree_model"]
    tree_ = tree.tree_
    values = tree_.value[:, 0]          # shape: (n_nodes, n_classes)
    breach_idx = list(tree.classes_).index(1)  # adjust if needed

    feature_names = _analytics["feature_names"]

    breach_prob = values[:, breach_idx] / values.sum(axis=1)


    node_labels = {}
    node_styles = {}

    for i in range(tree_.node_count):
        samples = tree_.n_node_samples[i]
        gini = tree_.impurity[i]

        values_i = values[i]
        pred_idx = values_i.argmax()
        pred_class = tree.classes_[pred_idx]
        pred_class_name = "Breach" if pred_class == 1 else "Non-breach"

        breach_pct = breach_prob[i] * 100
        bar = proportion_bar(breach_pct)

        if tree_.children_left[i] == -1:
            title = "Leaf node"
        else:
            feature = feature_names[tree_.feature[i]]
            threshold = tree_.threshold[i]
            title = f"{feature} ≤ {threshold:.2f}"

        label = label = f"""
<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">
  <TR>
    <TD ALIGN="LEFT"><B>{title}</B></TD>
  </TR>
  <TR>
    <TD ALIGN="LEFT"><FONT POINT-SIZE="6">samples = {samples}</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="LEFT"><FONT POINT-SIZE="6">breach rate = {breach_pct:.1f}%</FONT></TD>
  </TR>
  <TR>
    <TD ALIGN="LEFT"><FONT POINT-SIZE="6">class = {pred_class_name}</FONT></TD>
  </TR>
  <TR>
    <TD>{bar}</TD>
  </TR>
</TABLE>
>
"""


        node_labels[i] = label
        node_styles[i] = COLOR_BREACH if pred_class == 1 else COLOR_NON_BREACH


    # fig = plt.figure(figsize=(60, 26), dpi=250)
    dot_data = export_graphviz(
        tree,
        feature_names=feature_names,
        class_names=["Non-breach", "Breach"],
        filled=False,
        rounded=True,
        node_ids=True,
    )
    dot_lines = dot_data.split("\n")

    for i, line in enumerate(dot_lines):
        match = re.match(r'^(\d+)\s+\[.*\]\s*;$', line)
        if not match:
            continue

        node_id = int(match.group(1))
        if node_id not in node_labels:
            continue

        dot_lines[i] = (
            f'{node_id} [label={node_labels[node_id]}, '
            f'fillcolor="{node_styles[node_id]}", '
            f'style="filled, rounded", shape="box"] ;'
        )


    dot_fixed = "\n".join(dot_lines)


    # plot_tree(
    #     _analytics["tree_model"],
    #     feature_names=_analytics["feature_names"],
    #     class_names=["Non-breach", "Breach"],
    #     filled=False,          # IMPORTANT
    #     max_depth=4,
    #     rounded=True,
    #     fontsize=16,
    # )
    plt.title("Decision Logic Used to Identify High Breach Risk")
    graph = graphviz.Source(dot_fixed)
    st.graphviz_chart(graph, height=650, width='content')

    # st.pyplot(fig, width="content")

def render_tree_section(container, analytics):
    with container:
        st.subheader("Representative Decision Tree")

        render_tree_plot(analytics)

def render_rule_section(container, analytics):
    with container:
        st.subheader("Key Patterns Associated with Breaches")

        rules = analytics["rules"]

        if not rules:
            st.info("No strong high-risk patterns were identified.")
            return

        for rule in rules[:6]:
            st.markdown(f"- {format_rule_for_management(rule)}")

def main():
    init_session()

    analytics = prepare_aed_analytics()

    if not analytics:
        st.info("AED data not available. Please check inputs.")
        st.stop()

    st.session_state.aed_loaded = True
    st.session_state.aed_rules = analytics["rules"]

    # Layout containers
    title_section = st.container()
    tree_section = st.container()
    rule_section = st.container()
    footer_section = st.container()

    render_title_section(title_section)
    render_tree_section(tree_section, analytics)
    render_rule_section(rule_section, analytics)

    with footer_section:
        st.caption("AED analytics module — interpretability-focused view.")


if __name__ == "__main__":
    st.set_page_config(
        page_title="AED Analytics",
        layout="wide",
    )
    main()
