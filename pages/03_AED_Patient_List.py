import io
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils import add_title, log_action, seed_everything
from utils import format_rule_for_management

DATA_PATH = "data/AED4weeks.parquet"
SAMPLE_SIZE = 400
RANDOM_SEED = 42

COLUMN_ALIASES = {
    "id": ["id", "patient_id", "mrn"],
    "age": ["age", "patient_age"],
    "los": ["los", "lengthofstay", "length_of_stay"],
    "noofpatients": ["noofpatient", "noofpatients", "patients_at_arrival", "no_patients", "no_of_patients", "patientswaiting"],
    "noofinvestigation": ["noofinvestigation", "noofinvestigations", "diagnostic_tests", "tests", "no_of_investigations", "investigation_count"],
    "nooftreatment": ["nooftreatment", "nooftreatments", "treatments"],
    "dayofweek": ["dayofweek", "day_of_week", "dow"],
    "hrg": ["hrg", "health_group", "health_related_group"],
    "breachornot": ["breachornot", "breach", "breach_flag"],
}

def init_session():
    defaults = {
        "aed_df": None,
        "aed_colmap": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    colmap: Dict[str, str] = {}
    lower = {
        c.lower().replace(" ", "").replace("_", ""): c for c in df.columns
    }
    for target, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            key = a.replace("_", "").lower()
            if key in lower:
                colmap[target] = lower[key]
                break
    return colmap


def prepare_patient_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    seed_everything()

    try:
        df = pd.read_parquet(DATA_PATH)
    except Exception:
        st.error(
            "Failed to load AED4weeks.parquet. Please ensure dependencies are installed."
        )
        return pd.DataFrame(), {}

    colmap = _normalize_columns(df)

    if not df.empty:
        df = df.sample(
            n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED
        ).reset_index(drop=True)

    return df, colmap


# ---------------------------------------------------------------------
# Filtering Logic (Pure)
# ---------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    filters: Dict[str, Tuple],
) -> pd.DataFrame:
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    def apply_range(key):
        col = colmap.get(key)
        if col and f"{key}_range" in filters:
            lo, hi = filters[f"{key}_range"]
            return (df[col] >= lo) & (df[col] <= hi)
        return None

    for key in [
        "age",
        "los",
        "noofpatients",
        "noofinvestigation",
        "nooftreatment",
    ]:
        rng = apply_range(key)
        if rng is not None:
            mask &= rng

    if "dow_sel" in filters:
        mask &= df[colmap["dayofweek"]].isin(filters["dow_sel"])
    if "hrg_sel" in filters:
        mask &= df[colmap["hrg"]].isin(filters["hrg_sel"])
    if "breach_sel" in filters:
        mask &= df[colmap["breachornot"]].isin(filters["breach_sel"])

    return df[mask]


# ---------------------------------------------------------------------
# Rendering Sections (UI Only)
# ---------------------------------------------------------------------

def render_sidebar_filters(df, colmap) -> Dict[str, Tuple]:
    filters: Dict[str, Tuple] = {}

    with st.sidebar:
        st.markdown("### AED Patient List")
        render_analytics_helper_button()

        if df.empty:
            return filters

        def slider_range(col_key, label):
            col = colmap.get(col_key)
            if not col:
                return
            s = df[col].dropna()
            if s.empty:
                return
            lo, hi = float(s.min()), float(s.max())
            filters[f"{col_key}_range"] = st.slider(
                label, lo, hi, (lo, hi)
            )

        slider_range("age", "Age")
        slider_range("los", "Length of Stay")
        slider_range("noofpatients", "No. of Patients")
        slider_range("noofinvestigation", "No. of Investigations")
        slider_range("nooftreatment", "No. of Treatments")

        if "dayofweek" in colmap:
            vals = sorted(df[colmap["dayofweek"]].dropna().unique())
            filters["dow_sel"] = st.multiselect(
                "Day of Week", vals, default=vals
            )

        if "hrg" in colmap:
            vals = sorted(df[colmap["hrg"]].dropna().unique())
            filters["hrg_sel"] = st.multiselect("HRG", vals, default=vals)

        if "breachornot" in colmap:
            vals = sorted(df[colmap["breachornot"]].dropna().unique())
            filters["breach_sel"] = st.multiselect(
                "Breach or not", vals, default=vals
            )

    return filters


def render_summary(df, colmap):
    st.markdown("#### Summary")

    if df.empty:
        st.info("No data to summarise.")
        return

    cols = [
        colmap[c]
        for c in [
            "age",
            "los",
            "noofpatients",
            "noofinvestigation",
            "nooftreatment",
        ]
        if c in colmap
    ]

    st.write(df[cols].describe())

    breach_col = colmap.get("breachornot")
    if breach_col:
        if pd.api.types.is_numeric_dtype(df[breach_col]):
            st.write("Breach rate:", df[breach_col].mean())
        else:
            st.write(df[breach_col].value_counts(normalize=True))


def render_search(df, colmap):
    st.markdown("#### Search Patient by ID")

    id_col = colmap.get("id")
    if not id_col:
        st.info("No ID column detected.")
        return

    term = st.text_input("Search ID")
    if term:
        hits = df[df[id_col].astype(str).str.contains(term, case=False)]
        if hits.empty:
            st.info("No patients found.")
        else:
            st.dataframe(hits, width='stretch', hide_index=True)


def render_download(df, label, key):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label,
        data=buf.getvalue(),
        file_name=f"{key}.csv",
        mime="text/csv",
        width='stretch',
    )


def _value_changed(old, new) -> bool:
    if pd.isna(old) and pd.isna(new):
        return False
    if pd.isna(old) != pd.isna(new):
        return True
    return old != new


def _match_case(template: str, text: str) -> str:
    if template.isupper():
        return text.upper()
    if template.islower():
        return text.lower()
    return text


def _breach_from_los(los_value, breach_series: pd.Series, current_value):
    try:
        breached = float(los_value) > 240
    except (TypeError, ValueError):
        breached = False

    if pd.api.types.is_bool_dtype(breach_series):
        return bool(breached)
    if pd.api.types.is_numeric_dtype(breach_series):
        return int(breached)

    if isinstance(current_value, str):
        template = current_value.strip()
        lower = template.lower()
        if lower in {"yes", "no"}:
            return _match_case(template, "Yes" if breached else "No")
        if "breach" in lower:
            return _match_case(template, "Breach" if breached else "Non-breach")

    return "Breach" if breached else "Non-breach"


def render_edit_delete(df, colmap) -> pd.DataFrame:
    id_col = colmap.get("id")
    if not id_col:
        return df

    st.markdown("#### Modify Patient")

    ids = df[id_col].astype(str).tolist()
    selected = st.selectbox("Select Patient", [""] + ids)

    if selected:
        idx = df.index[df[id_col].astype(str) == selected][0]
        original = df.loc[idx].copy()
        edits = {}
        breach_col = colmap.get("breachornot")
        los_col = colmap.get("los")
        old_los = df.at[idx, los_col] if los_col else None

        for col in df.columns:
            if breach_col and col == breach_col:
                continue
            value = df.at[idx, col]
            if pd.api.types.is_bool_dtype(df[col]):
                edits[col] = st.checkbox(col, value=bool(value) if pd.notna(value) else False)
            elif pd.api.types.is_numeric_dtype(df[col]):
                edits[col] = st.number_input(
                    col,
                    value=float(value) if pd.notna(value) else 0.0,
                )
            else:
                edits[col] = st.text_input(col, value="" if pd.isna(value) else str(value))

        if st.button("Confirm update"):
            for c, v in edits.items():
                if pd.api.types.is_integer_dtype(df[c]):
                    try:
                        v = int(v)
                    except (TypeError, ValueError):
                        pass
                df.at[idx, c] = v
            if breach_col and los_col in edits and _value_changed(old_los, edits[los_col]):
                df.at[idx, breach_col] = _breach_from_los(
                    edits[los_col],
                    df[breach_col],
                    df.at[idx, breach_col],
                )
            changes = {}
            for col in df.columns:
                if _value_changed(original[col], df.at[idx, col]):
                    changes[col] = {
                        "from": original[col],
                        "to": df.at[idx, col],
                    }
            params = {"id": selected, "changes": changes}
            if changes:
                params["edited_columns"] = list(changes.keys())
            log_action("AED Patient List", "edit", params)
            st.success("Record updated (session only).")

    st.markdown("#### Delete Patient")
    del_id = st.selectbox("Select Patient to Delete", [""] + ids)

    if del_id and st.button("Confirm delete"):
        df = df[df[id_col].astype(str) != del_id].reset_index(drop=True)
        log_action("AED Patient List", "delete", {"id": del_id})
        st.warning(f"Deleted {del_id} (session only).")

    return df


import streamlit as st


@st.dialog("AED Risk Analytics — High-Risk Breach Patterns", width="large")
def show_aed_analytics_popup():
    rules = st.session_state.get("aed_rules")

    if not rules:
        st.info(
            "No AED analytics insights available.\n\n"
            "Please visit the AED Risk Analytics page first."
        )
        return

    st.markdown(
        """
        These insights are derived from an interpretable decision-tree model.
        They summarise **common high-risk patterns** associated with breaches of
        the 4-hour AED target and are intended to help guide filtering and review
        of patient records.
        """
    )

    st.markdown("---")

    for rule in rules[:6]:
        st.markdown(f"- {format_rule_for_management(rule)}")

    st.caption(
        "Insights are advisory only and do not automatically apply filters."
    )

    if st.button("Close"):
        st.rerun()


def render_analytics_helper_button():
    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("🔍 View AED Risk Insights", width='stretch'):
            show_aed_analytics_popup()


def main():
    init_session()
    seed_everything()
    add_title("AED Patient List")

    if st.session_state.aed_df is None:
        df, colmap = prepare_patient_data()
        st.session_state.aed_df = df
        st.session_state.aed_colmap = colmap

    df = st.session_state.aed_df
    colmap = st.session_state.aed_colmap

    filters = render_sidebar_filters(df, colmap)
    filtered = apply_filters(df, colmap, filters)

    st.markdown("#### Filtered Patients")
    st.dataframe(filtered, width='stretch', hide_index=True)
    render_download(filtered, "Download filtered CSV", "filtered-patients")

    render_summary(filtered, colmap)
    render_search(df, colmap)

    df = render_edit_delete(df, colmap)
    st.session_state.aed_df = df

    st.markdown("---")
    st.markdown("#### Full Patients List")
    st.dataframe(df, width='stretch', hide_index=True)
    render_download(df, "Download full CSV", "full-patients")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
