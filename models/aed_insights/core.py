"""Core data access, schema, and feature engineering for AED analytics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models.aed_insights.config import AEDConfig


@dataclass(frozen=True)
class Schema:
    """Canonical AED schema and variable semantics."""
    id_col: str = "id"
    age_col: str = "age"
    day_col: str = "day"
    dayofweek_col: str = "dayofweek"
    period_col: str = "period"
    los_col: str = "los"
    breach_col: str = "breachornot"
    hrg_col: str = "hrg"
    noofinvestigation_col: str = "noofinvestigation"
    nooftreatment_col: str = "nooftreatment"
    noofpatients_col: str = "noofpatients"

    breach_flag_col: str = "breach_flag"
    hrg_group_col: str = "hrg_group"
    investigation_cluster_col: str = "investigation_cnt_cluster"

    @property
    def numeric_cols(self) -> list[str]:
        return [
            self.age_col,
            self.los_col,
            self.noofinvestigation_col,
            self.nooftreatment_col,
            self.noofpatients_col,
        ]

    @property
    def categorical_cols(self) -> list[str]:
        return [
            self.dayofweek_col,
            self.period_col,
            self.hrg_col,
            self.breach_col,
        ]

    @property
    def required_cols(self) -> list[str]:
        return [
            self.id_col,
            self.age_col,
            self.day_col,
            self.dayofweek_col,
            self.period_col,
            self.los_col,
            self.breach_col,
            self.hrg_col,
            self.noofinvestigation_col,
            self.nooftreatment_col,
            self.noofpatients_col,
        ]


def get_schema() -> Schema:
    """Return the default AED schema."""
    return Schema()


@dataclass(frozen=True)
class AEDRepository:
    """Repository for AED dataset access and sampling."""
    config: AEDConfig
    schema: Schema = get_schema()

    def _resolve_path(self) -> Path:
        return Path(self.config.data.data_path)

    def _normalize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        normalized = {
            c.lower().replace(" ", "").replace("_", ""): c for c in df.columns
        }
        mapping: Dict[str, str] = {}
        for canonical in self.schema.required_cols:
            key = canonical.replace("_", "").lower()
            if key in normalized:
                mapping[canonical] = normalized[key]
            else:
                raise ValueError(f"Missing required column: {canonical}")
        renamed = df.rename(columns={v: k for k, v in mapping.items()})
        return renamed, mapping

    def load_raw(self) -> pd.DataFrame:
        """Load the raw dataset from parquet and normalize columns."""
        path = self._resolve_path()
        df = pd.read_parquet(path)
        df, _ = self._normalize_columns(df)
        return df

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic sampling for downstream analytics."""
        n = min(self.config.data.sample_size, len(df))
        if n == len(df):
            return df.reset_index(drop=True)
        return df.sample(n=n, random_state=self.config.data.random_state).reset_index(drop=True)

    def load_sampled(self) -> pd.DataFrame:
        """Load and sample the dataset deterministically."""
        df = self.load_raw()
        return self.sample(df)


@dataclass(frozen=True)
class FeatureBundle:
    """Feature-enhanced dataset with derived columns."""
    data: pd.DataFrame


def add_breach_flag(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> pd.DataFrame:
    """Add a breach flag derived from Breachornot labels or LoS threshold."""
    result = df.copy()
    if schema.breach_col in result.columns:
        labels = result[schema.breach_col].astype(str).str.strip().str.lower()
        positive = set(config.features.breach_positive_labels)
        negative = set(config.features.breach_negative_labels)
        mapped = labels.map(lambda v: 1 if v in positive else 0 if v in negative else None)
        if mapped.notna().any():
            result[schema.breach_flag_col] = mapped.astype("Int64")
        else:
            result[schema.breach_flag_col] = (
                result[schema.los_col] >= config.features.breach_threshold_minutes
            )
    else:
        result[schema.breach_flag_col] = (
            result[schema.los_col] >= config.features.breach_threshold_minutes
        )
    return result


def add_hrg_group(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> pd.DataFrame:
    """Group rare HRG categories into an 'Other' bucket."""
    result = df.copy()
    counts = result[schema.hrg_col].value_counts(dropna=False)
    rare = counts[counts < config.features.hrg_group_min_count].index
    result[schema.hrg_group_col] = result[schema.hrg_col].where(
        ~result[schema.hrg_col].isin(rare),
        config.features.hrg_group_other_label,
    )
    return result


def add_investigation_cluster(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> pd.DataFrame:
    """Bucket investigation counts into simple discrete clusters."""
    result = df.copy()
    if schema.noofinvestigation_col not in result.columns:
        return result
    if not config.features.make_investigation_cluster:
        return result
    values = result[[schema.noofinvestigation_col]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    k_min = config.features.investigation_cluster_k_min
    k_max = config.features.investigation_cluster_k_max
    k_range = range(k_min, k_max + 1)
    inertia = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=config.data.random_state,
            n_init=config.features.investigation_cluster_n_init,
            algorithm=config.features.investigation_cluster_algorithm,
        )
        kmeans.fit(scaled)
        inertia.append(kmeans.inertia_)

    kneedle = KneeLocator(
        list(k_range),
        inertia,
        curve="convex",
        direction="decreasing",
    )
    optimal_k = int(kneedle.elbow or config.features.investigation_cluster_default_k)

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=config.data.random_state,
        n_init=config.features.investigation_cluster_n_init,
        algorithm=config.features.investigation_cluster_algorithm,
    )
    result[schema.investigation_cluster_col] = kmeans.fit_predict(scaled)
    return result


def build_featured_df(df: pd.DataFrame, config: AEDConfig, schema: Schema = None) -> pd.DataFrame:
    """Apply all feature engineering steps in a single, reusable pipeline."""
    schema = schema or get_schema()
    result = add_breach_flag(df, schema, config)
    result = add_hrg_group(result, schema, config)
    result = add_investigation_cluster(result, schema, config)
    return result
