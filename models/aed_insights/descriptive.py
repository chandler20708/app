"""Descriptive analytics tables and figures for AED insights."""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from models.aed_insights.config import AEDConfig
from models.aed_insights.core import Schema, get_schema
from utils import order_days


@dataclass(frozen=True)
class DescriptiveArtifacts:
    """Container for descriptive tables and figures."""
    tables: Dict[str, pd.DataFrame]
    figures: Dict[str, Figure]
    metadata: Dict[str, object]


def numeric_summary(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Compute numeric summary statistics for key variables."""
    summary = df[schema.numeric_cols].describe().T
    return summary.round(3)


def data_quality_summary(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Summarize missingness, duplicates, and basic integrity checks."""
    row_count = len(df)
    col_count = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())
    missing_total = int(df.isna().sum().sum())
    missing_pct = round((missing_total / max(row_count * col_count, 1)) * 100, 3)

    metrics = [
        {"metric": "row_count", "value": row_count},
        {"metric": "column_count", "value": col_count},
        {"metric": "duplicate_row_count", "value": duplicate_rows},
        {"metric": "missing_total", "value": missing_total},
        {"metric": "missing_pct_total", "value": missing_pct},
    ]

    if schema.id_col in df.columns:
        unique_ids = int(df[schema.id_col].nunique(dropna=True))
        duplicate_ids = int(df[schema.id_col].duplicated().sum())
        metrics.extend(
            [
                {"metric": "unique_id_count", "value": unique_ids},
                {"metric": "duplicate_id_count", "value": duplicate_ids},
            ]
        )

    return pd.DataFrame(metrics)


def missingness_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing value counts and percentages by column."""
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / max(len(df), 1) * 100).round(3)
    out = pd.DataFrame(
        {
            "missing_count": missing_counts.astype(int),
            "missing_pct": missing_pct,
        }
    )
    return out


def categorical_proportions(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute proportions for a categorical column."""
    proportions = df[column].value_counts(normalize=True, dropna=False).round(3)
    return proportions.to_frame("proportion")


def breach_share(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Compute overall breach share as percentage."""
    proportions = df[schema.breach_col].value_counts(normalize=True, dropna=False).round(3) * 100
    return proportions.to_frame("share_pct")


def breach_rate_by(df: pd.DataFrame, schema: Schema, column: str) -> pd.DataFrame:
    """Compute breach rate by a categorical column."""
    out = (
        df.groupby(column)[schema.breach_flag_col]
        .mean()
        .reset_index()
        .rename(columns={schema.breach_flag_col: "Breach_Rate"})
        .sort_values("Breach_Rate", ascending=False)
        .reset_index(drop=True)
    )
    out["Breach_Rate"] = out["Breach_Rate"].round(3)
    return out


def correlation_matrix(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for key numeric variables."""
    cols = [
        schema.age_col,
        schema.los_col,
        schema.noofinvestigation_col,
        schema.nooftreatment_col,
        schema.noofpatients_col,
    ]
    present = [col for col in cols if col in df.columns]
    if len(present) < 2:
        return pd.DataFrame()
    return df[present].corr(method="pearson").round(3)


def build_descriptive_tables(
    df: pd.DataFrame,
    config: AEDConfig,
    schema: Schema = None,
) -> Dict[str, pd.DataFrame]:
    """Build descriptive tables aligned with AED EDA outputs."""
    schema = schema or get_schema()
    tables: Dict[str, pd.DataFrame] = {}

    tables["data_quality_summary"] = data_quality_summary(df, schema)
    tables["missingness_by_column"] = missingness_by_column(df)
    tables["numeric_summary"] = numeric_summary(df, schema)
    tables["categorical_day"] = categorical_proportions(df, schema.day_col)
    day_order = _dayofweek_order(df[schema.dayofweek_col], config)
    tables["categorical_dayofweek"] = _reorder_index(
        categorical_proportions(df, schema.dayofweek_col),
        day_order,
    )
    tables["categorical_period"] = categorical_proportions(df, schema.period_col)
    tables["categorical_hrg"] = categorical_proportions(df, schema.hrg_col)
    if schema.hrg_group_col in df.columns:
        tables["categorical_hrg_group"] = categorical_proportions(df, schema.hrg_group_col)
    if schema.investigation_cluster_col in df.columns:
        tables["categorical_investigation_cnt_cluster"] = categorical_proportions(
            df, schema.investigation_cluster_col
        )
    tables["categorical_breachornot"] = categorical_proportions(df, schema.breach_col)
    breach_day = breach_rate_by(df, schema, schema.dayofweek_col)
    if day_order:
        breach_day["_order"] = pd.Categorical(
            breach_day[schema.dayofweek_col],
            categories=day_order,
            ordered=True,
        )
        breach_day = breach_day.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    tables["breach_rate_by_dayofweek"] = breach_day
    tables["breach_rate_by_period"] = breach_rate_by(df, schema, schema.period_col)
    tables["correlation_matrix"] = correlation_matrix(df, schema)

    return tables


def _base_figure(config: AEDConfig, width: float = None, height: float = None) -> Figure:
    fig = plt.figure(
        figsize=(width or config.plots.figure_width, height or config.plots.figure_height)
    )
    return fig


def _observed_order(values: Iterable[str], preferred: Iterable[str]) -> List[str]:
    observed = pd.Series(values).dropna().unique().tolist()
    ordered = [item for item in preferred if item in observed]
    return ordered or sorted(observed)


def _dayofweek_order(values: Iterable[str], _: AEDConfig) -> list[str]:
    return order_days(values)


def _reorder_index(table: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    if table.empty or not order:
        return table
    existing = [idx for idx in table.index if idx not in order]
    return table.reindex(order + existing)


def numeric_histogram(df: pd.DataFrame, schema: Schema, config: AEDConfig, column: str) -> Figure:
    fig = _base_figure(config)
    ax = fig.add_subplot(1, 1, 1)
    sns.histplot(df[column].dropna(), bins=config.plots.hist_bins, kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    return fig


def numeric_boxplot(df: pd.DataFrame, schema: Schema, config: AEDConfig, column: str) -> Figure:
    fig = _base_figure(config)
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(y=df[column], ax=ax)
    ax.set_title(f"Boxplot of {column}")
    ax.set_xlabel("")
    ax.set_ylabel(column)
    return fig


def categorical_countplot(df: pd.DataFrame, schema: Schema, config: AEDConfig, column: str) -> Figure:
    fig = _base_figure(config, width=8.0, height=4.0)
    ax = fig.add_subplot(1, 1, 1)
    if column == schema.dayofweek_col:
        order = _dayofweek_order(df[column], config)
        sns.countplot(data=df, x=column, order=order, ax=ax)
    else:
        sns.countplot(data=df, x=column, ax=ax)
    ax.set_title(f"Count of {column}")
    ax.tick_params(axis="x", rotation=45)
    return fig


def age_vs_los(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        df[schema.age_col],
        df[schema.los_col],
        alpha=config.plots.scatter_alpha,
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Length of Stay (minutes)")
    ax.set_title("Age vs Length of Stay")
    return fig


def age_by_investigations(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=10.0, height=6.0)
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(data=df, x=schema.noofinvestigation_col, y=schema.age_col, ax=ax)
    ax.set_title("Age Distribution by Number of Investigations")
    ax.set_xlabel("Number of Investigations (Discrete)")
    ax.set_ylabel("Patient Age (Continuous)")
    return fig


def los_by_investigations(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=10.0, height=6.0)
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(
        data=df,
        x=schema.noofinvestigation_col,
        y=schema.los_col,
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel("Number of Investigations")
    ax.set_ylabel("Length of Stay (LoS)")
    ax.set_title("Distribution of Length of Stay by Number of Investigations")
    return fig


def age_by_treatments(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=10.0, height=6.0)
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(data=df, x=schema.nooftreatment_col, y=schema.age_col, ax=ax)
    ax.set_title("Age Distribution by Number of Treatments")
    ax.set_xlabel("Number of Treatments (Discrete)")
    ax.set_ylabel("Patient Age (Continuous)")
    return fig


def age_by_breach(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=8.0, height=6.0)
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(data=df, x=schema.breach_col, y=schema.age_col, ax=ax)
    ax.set_title("Age Distribution: Breach vs Non-breach")
    ax.set_xlabel("Breach Status")
    ax.set_ylabel("Patient Age")
    return fig


def age_by_hrg(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=14.0, height=8.0)
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(data=df, x=schema.hrg_col, y=schema.age_col, ax=ax)
    ax.set_title("Patient Age Distribution by HRG Category")
    ax.set_xlabel("HRG Category")
    ax.set_ylabel("Patient Age")
    ax.tick_params(axis="x", rotation=45)
    return fig


def los_by_hrg(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=12.0, height=6.0)
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(data=df, x=schema.hrg_col, y=schema.los_col, ax=ax)
    ax.set_title("Length of Stay (LoS) by HRG")
    ax.tick_params(axis="x", rotation=45)
    return fig


def mean_los_by_period(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=12.0, height=6.0)
    ax = fig.add_subplot(1, 1, 1)
    sns.lineplot(data=df, x=schema.period_col, y=schema.los_col, marker="o", errorbar="sd", ax=ax)
    ax.axhline(
        y=config.features.breach_threshold_minutes,
        linestyle="--",
        label="4-Hour Target",
        linewidth=config.plots.threshold_line_width,
    )
    ax.set_title("Average Length of Stay by Arrival Hour (Period)")
    ax.set_xlabel("Hour of Arrival (0=Midnight, 23=11PM)")
    ax.set_ylabel("Mean LoS (minutes)")
    try:
        ax.set_xticks(list(range(0, 24)))
    except Exception:
        pass
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return fig


def los_by_dayofweek(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=12.0, height=6.0)
    ax = fig.add_subplot(1, 1, 1)
    order = _dayofweek_order(df[schema.dayofweek_col], config)
    sns.boxplot(data=df, x=schema.dayofweek_col, y=schema.los_col, order=order, ax=ax)
    ax.axhline(
        config.features.breach_threshold_minutes,
        linestyle="--",
        label="4-Hour Target",
        linewidth=config.plots.threshold_line_width,
    )
    ax.set_title("Distribution of Length of Stay (LoS) by Day of Week")
    ax.set_ylabel("LoS (minutes)")
    ax.set_xlabel("Day of the Week")
    ax.legend()
    return fig


def correlation_heatmap(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=10.0, height=8.0)
    ax = fig.add_subplot(1, 1, 1)
    cols = [
        schema.age_col,
        schema.los_col,
        schema.noofinvestigation_col,
        schema.nooftreatment_col,
        schema.noofpatients_col,
    ]
    corr = df[cols].corr(method="pearson")
    sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Pearson Correlation Matrix")
    return fig


def hrg_vs_treatments_heatmap(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=10.0, height=8.0)
    ax = fig.add_subplot(1, 1, 1)
    ct = pd.crosstab(df[schema.hrg_col], df[schema.nooftreatment_col])
    sns.heatmap(ct, annot=True, fmt="d", ax=ax)
    ax.set_title("Count of Patients: HRG vs Number of Treatments")
    ax.set_xlabel("Number of Treatments")
    ax.set_ylabel("HRG Category")
    return fig


def hrg_vs_investigations_heatmap(df: pd.DataFrame, schema: Schema, config: AEDConfig) -> Figure:
    fig = _base_figure(config, width=10.0, height=8.0)
    ax = fig.add_subplot(1, 1, 1)
    ct = pd.crosstab(df[schema.hrg_col], df[schema.noofinvestigation_col])
    sns.heatmap(ct, annot=True, fmt="d", ax=ax)
    ax.set_title("Count of Patients: HRG vs Number of Investigations")
    ax.set_xlabel("Number of Investigations")
    ax.set_ylabel("HRG Category")
    return fig


def build_descriptive_figures(
    df: pd.DataFrame,
    config: AEDConfig,
    schema: Schema = None,
) -> Dict[str, Figure]:
    """Generate all descriptive figures aligned with AED EDA outputs."""
    schema = schema or get_schema()
    figures: Dict[str, Figure] = {}

    for col in schema.numeric_cols:
        figures[f"hist_{col}"] = numeric_histogram(df, schema, config, col)
        figures[f"box_{col}"] = numeric_boxplot(df, schema, config, col)

    for col in [schema.day_col, schema.period_col, schema.dayofweek_col, schema.hrg_col, schema.breach_col]:
        figures[f"count_{col}"] = categorical_countplot(df, schema, config, col)

    figures["age_vs_los"] = age_vs_los(df, schema, config)
    figures["los_by_investigations"] = los_by_investigations(df, schema, config)
    figures["age_by_investigations"] = age_by_investigations(df, schema, config)
    figures["age_by_treatments"] = age_by_treatments(df, schema, config)
    figures["age_by_breach"] = age_by_breach(df, schema, config)
    figures["age_by_hrg"] = age_by_hrg(df, schema, config)
    figures["los_by_hrg"] = los_by_hrg(df, schema, config)
    figures["mean_los_by_period"] = mean_los_by_period(df, schema, config)
    figures["los_by_dayofweek"] = los_by_dayofweek(df, schema, config)
    figures["corr_heatmap"] = correlation_heatmap(df, schema, config)
    figures["hrg_vs_treatments_heatmap"] = hrg_vs_treatments_heatmap(df, schema, config)
    figures["hrg_vs_investigations_heatmap"] = hrg_vs_investigations_heatmap(df, schema, config)

    return figures
