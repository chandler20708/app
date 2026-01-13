"""Inference tests and aggregation for AED factors analysis."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import itertools
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models.aed_insights.config import AEDConfig
from models.aed_insights.core import Schema, get_schema


@dataclass(frozen=True)
class InferenceArtifacts:
    """Container for inference tables and figures."""
    tables: Dict[str, pd.DataFrame]
    figures: Dict[str, object]
    metadata: Dict[str, object]


def mw_diagnostics(series: pd.Series) -> Dict[str, float]:
    """Compute Mann-Whitney diagnostic statistics for a series."""
    s = series.dropna().astype(float)
    if s.empty:
        return {
            "n": 0.0,
            "p0": np.nan,
            "tie_rate": np.nan,
            "median": np.nan,
            "iqr": np.nan,
        }
    counts = s.value_counts(normalize=True)
    tie_rate = float(counts.iloc[0])
    iqr = float(np.percentile(s, 75) - np.percentile(s, 25))
    return {
        "n": float(len(s)),
        "p0": float((s == 0).mean()),
        "tie_rate": tie_rate,
        "median": float(np.median(s)),
        "iqr": iqr,
    }


def mw_screen(
    diag_breach: Dict[str, float],
    diag_non: Dict[str, float],
    config: AEDConfig,
) -> Dict[str, object]:
    """Apply screening rules for Mann-Whitney tests."""
    min_n = config.inference.mw_screen_min_n
    small_n_fail = diag_breach["n"] < min_n or diag_non["n"] < min_n

    if config.inference.mw_screen_use_ties:
        tie_fail = (
            diag_breach["tie_rate"] >= config.inference.mw_diag_tie_threshold
            or diag_non["tie_rate"] >= config.inference.mw_diag_tie_threshold
        )
    else:
        tie_fail = False

    zero_fail = (
        diag_breach["p0"] >= config.inference.mw_diag_zero_threshold
        or diag_non["p0"] >= config.inference.mw_diag_zero_threshold
    )

    iqr_b = diag_breach["iqr"]
    iqr_n = diag_non["iqr"]
    if np.isnan(iqr_b) or np.isnan(iqr_n):
        iqr_ratio = np.nan
        disp_fail = False
    else:
        denom = max(min(iqr_b, iqr_n), 1e-9)
        iqr_ratio = max(iqr_b, iqr_n) / denom
        disp_fail = iqr_ratio > config.inference.mw_screen_iqr_ratio_max

    pass_screen = not (small_n_fail or tie_fail or zero_fail or disp_fail)

    return {
        "pass_screen": pass_screen,
        "iqr_ratio": float(iqr_ratio) if not np.isnan(iqr_ratio) else np.nan,
        "screen_failed": float(not pass_screen),
        "fail_small_n": float(small_n_fail),
        "fail_ties": float(tie_fail),
        "fail_zeros": float(zero_fail),
        "fail_dispersion": float(disp_fail),
    }


def mann_whitney_test(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    config: AEDConfig,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run screened Mann-Whitney U test for a numeric feature."""
    breach = df[df[group_col] == 1][feature]
    non_breach = df[df[group_col] == 0][feature]

    diag_breach = mw_diagnostics(breach)
    diag_non = mw_diagnostics(non_breach)
    screen = mw_screen(diag_breach, diag_non, config)

    if screen["pass_screen"]:
        stat, p_value = stats.mannwhitneyu(
            breach.dropna(),
            non_breach.dropna(),
            alternative="two-sided",
        )
    else:
        stat, p_value = np.nan, np.nan

    result = {
        "var": feature,
        "U": stat,
        "p_value": p_value,
        "n_breach": diag_breach["n"],
        "n_nonbreach": diag_non["n"],
        "pass_screen": screen["pass_screen"],
        "screen_failed": screen["screen_failed"],
    }

    diagnostics = {
        "var": feature,
        "n_breach": diag_breach["n"],
        "n_nonbreach": diag_non["n"],
        "p0_breach": diag_breach["p0"],
        "p0_nonbreach": diag_non["p0"],
        "tie_rate_breach": diag_breach["tie_rate"],
        "tie_rate_nonbreach": diag_non["tie_rate"],
        "iqr_breach": diag_breach["iqr"],
        "iqr_nonbreach": diag_non["iqr"],
        "median_breach": diag_breach["median"],
        "median_nonbreach": diag_non["median"],
        "iqr_ratio": screen["iqr_ratio"],
        "screen_failed": screen["screen_failed"],
        "fail_small_n": screen["fail_small_n"],
        "fail_ties": screen["fail_ties"],
        "fail_zeros": screen["fail_zeros"],
        "fail_dispersion": screen["fail_dispersion"],
    }

    return result, diagnostics


def categorical_association_test(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
) -> Dict[str, object]:
    """Run chi-square or Fisher's exact test for categorical association."""
    contingency = pd.crosstab(df[feature], df[target_col])
    if contingency.empty:
        return {
            "feature": feature,
            "test": None,
            "statistic": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
        }

    if contingency.shape == (2, 2):
        odds_ratio, p_value = stats.fisher_exact(contingency)
        return {
            "feature": feature,
            "test": "fisher_exact",
            "statistic": odds_ratio,
            "p_value": p_value,
            "dof": np.nan,
        }

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    return {
        "feature": feature,
        "test": "chi_square",
        "statistic": chi2,
        "p_value": p_value,
        "dof": dof,
    }


def _cramers_v(contingency: pd.DataFrame, chi2: float) -> float:
    n = contingency.to_numpy().sum()
    if n == 0:
        return np.nan
    r, c = contingency.shape
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return np.nan
    phi2 = chi2 / n
    return float(np.sqrt(phi2 / denom))


def k_means_on_others_copy(df: pd.DataFrame, config: AEDConfig, schema: Schema) -> pd.DataFrame:
    """Run K-Means on selected numeric variables and test association with breach."""
    results = []
    k_range = range(config.inference.kmeans_k_min, config.inference.kmeans_k_max + 1)

    for var in config.inference.kmeans_vars:
        if var not in df.columns:
            continue

        X = df[[var]].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        inertia = []
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=config.data.random_state,
                n_init=config.inference.kmeans_n_init,
                algorithm=config.inference.kmeans_algorithm,
            )
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        kneedle = KneeLocator(
            list(k_range),
            inertia,
            curve="convex",
            direction="decreasing",
        )
        optimal_k = int(kneedle.elbow or config.inference.kmeans_k_min)

        kmeans = KMeans(
            n_clusters=optimal_k,
            random_state=config.data.random_state,
            n_init=config.inference.kmeans_n_init,
            algorithm=config.inference.kmeans_algorithm,
        )
        clusters = kmeans.fit_predict(X_scaled)

        ct = pd.crosstab(clusters, df[schema.breach_flag_col])
        chi2, p_value, dof, _ = stats.chi2_contingency(ct)
        cramers_v = _cramers_v(ct, chi2)

        results.append(
            {
                "Variable": var,
                "Optimal_k": optimal_k,
                "Chi_square": chi2,
                "p_value": p_value,
                "Cramers_V": cramers_v,
            }
        )

    return pd.DataFrame(results)


def categorical_comp(df: pd.DataFrame, config: AEDConfig, schema: Schema) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run categorical association tests with Task5-style grouping logic."""
    results = []
    day_band_rows = []

    for var in config.inference.categorical_vars:
        if var not in df.columns:
            continue

        if var == schema.day_col:
            day_band = pd.cut(
                df[var],
                bins=config.inference.day_band_bins,
                labels=config.inference.day_band_labels,
            )
            band_df = df.assign(day_band=day_band)
            bands = list(band_df["day_band"].cat.categories)

            for b1, b2 in itertools.combinations(bands, 2):
                sub = band_df[band_df["day_band"].isin([b1, b2])]
                ct_band = pd.crosstab(sub["day_band"], sub[schema.breach_flag_col])
                if ct_band.shape != (2, 2):
                    continue
                odds_ratio, fisher_p = stats.fisher_exact(ct_band)
                day_band_rows.append(
                    {
                        "Band_A": str(b1),
                        "Band_B": str(b2),
                        "Odds_Ratio": odds_ratio,
                        "p_value": fisher_p,
                    }
                )
            continue

        if var == schema.period_col:
            period_counts = df[var].value_counts()
            rare_periods = period_counts[period_counts < config.inference.period_min_count].index
            period_grp = df[var].replace(rare_periods, "Other")
            ct = pd.crosstab(period_grp, df[schema.breach_flag_col])
            chi2, p_value, dof, _ = stats.chi2_contingency(ct)
            cramers_v = _cramers_v(ct, chi2)
            results.append(
                {
                    "Variable": f"{var}_grouped",
                    "Test": "chi_square",
                    "Chi_square": chi2,
                    "Dof": dof,
                    "p_value": p_value,
                    "Cramers_V": cramers_v,
                    "Notes": "Grouped rare periods",
                }
            )
            continue

        ct = pd.crosstab(df[var], df[schema.breach_flag_col])
        chi2, p_value, dof, _ = stats.chi2_contingency(ct)
        cramers_v = _cramers_v(ct, chi2)
        results.append(
            {
                "Variable": var,
                "Test": "chi_square",
                "Chi_square": chi2,
                "Dof": dof,
                "p_value": p_value,
                "Cramers_V": cramers_v,
                "Notes": "",
            }
        )

    return pd.DataFrame(results), pd.DataFrame(day_band_rows)


def logit_single_predictor(
    df: pd.DataFrame,
    predictor: str,
    target_col: str,
) -> pd.DataFrame:
    """Fit a univariate statsmodels Logit and return odds ratios."""
    data = df[[target_col, predictor]].dropna().copy()
    if data.empty:
        return pd.DataFrame()
    y = data[target_col].astype(int)
    X = sm.add_constant(data[[predictor]])

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
    except (LinAlgError, PerfectSeparationError, ValueError):
        return pd.DataFrame()

    out = result.summary2().tables[1].copy()
    out["odds_ratio"] = np.exp(out["Coef."])
    out["predictor"] = predictor
    return out.reset_index(names="term")


def l2_logistic_odds_ratios(
    df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    extra_cols: List[str],
    target_col: str,
    config: AEDConfig,
) -> pd.DataFrame:
    """Fit L2-regularized logistic regression and return odds ratios."""
    extra_cols = [col for col in extra_cols if col not in cat_cols + numeric_cols]
    use_cols = [c for c in (numeric_cols + cat_cols + extra_cols) if c in df.columns]
    if not use_cols:
        return pd.DataFrame()
    X = pd.get_dummies(df[use_cols], drop_first=True).astype(float)
    y = df[target_col].astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = LogisticRegression(
            penalty="l2",
            solver=config.inference.l2_logreg_solver,
            max_iter=config.inference.l2_logreg_max_iter,
            C=config.inference.l2_logreg_c,
            class_weight=config.inference.l2_logreg_class_weight,
        )
        model.fit(X, y)
    or_df = pd.DataFrame({"Feature": X.columns, "Odds_Ratio": np.exp(model.coef_[0])})
    return or_df.sort_values("Odds_Ratio", ascending=False).reset_index(drop=True)


def compute_vif(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Compute variance inflation factors for selected columns."""
    use_cols = [c for c in columns if c in df.columns]
    if not use_cols:
        return pd.DataFrame()
    X = pd.get_dummies(df[use_cols], drop_first=True).astype(float)
    vif = pd.DataFrame({"Feature": X.columns})
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)


def build_inference_tables(
    df: pd.DataFrame,
    config: AEDConfig,
    schema: Schema = None,
) -> Dict[str, pd.DataFrame]:
    """Build inference tables for numeric and categorical screening."""
    schema = schema or get_schema()
    tables: Dict[str, pd.DataFrame] = {}

    mw_rows: List[dict] = []
    for feature in config.inference.mw_vars:
        if feature not in df.columns:
            continue
        result, diagnostics = mann_whitney_test(df, feature, schema.breach_flag_col, config)
        combined = dict(result)
        for key, value in diagnostics.items():
            if key not in combined:
                combined[key] = value
        mw_rows.append(combined)
    mw_table = pd.DataFrame(mw_rows)
    if not mw_table.empty:
        mw_table = mw_table.set_index("var")[["U", "p_value"]]
    tables["mann_whitney"] = mw_table

    categorical_table, day_band_pairs = categorical_comp(df, config, schema)
    tables["categorical_tests"] = categorical_table
    if not day_band_pairs.empty:
        tables["day_band_pairwise_fisher"] = day_band_pairs

    if config.inference.logit_single_predictor in df.columns:
        logit_table = logit_single_predictor(
            df,
            config.inference.logit_single_predictor,
            schema.breach_flag_col,
        )
    else:
        logit_table = pd.DataFrame()
    tables["logit_single_predictor"] = logit_table

    l2_numeric = list(config.inference.l2_numeric_predictors)
    if schema.investigation_cluster_col in df.columns and schema.investigation_cluster_col not in l2_numeric:
        l2_numeric.append(schema.investigation_cluster_col)
    extra_cols: List[str] = []
    if schema.investigation_cluster_col in df.columns:
        extra_cols.append(schema.investigation_cluster_col)
    if schema.hrg_group_col in df.columns:
        extra_cols.append(schema.hrg_group_col)
    l2_table = l2_logistic_odds_ratios(
        df,
        l2_numeric,
        list(config.inference.l2_categorical_predictors),
        extra_cols,
        schema.breach_flag_col,
        config,
    )
    tables["l2_logistic_odds_ratios"] = l2_table.head(15)

    vif_cols = list(config.inference.vif_predictors)
    if schema.investigation_cluster_col in df.columns and schema.investigation_cluster_col not in vif_cols:
        vif_cols.append(schema.investigation_cluster_col)
    vif_table = compute_vif(df, vif_cols)
    tables["vif"] = vif_table.head(15)

    kmeans_table = k_means_on_others_copy(df, config, schema)
    if not kmeans_table.empty:
        tables["kmeans_on_numeric"] = kmeans_table

    return tables
