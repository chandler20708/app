"""Configuration for AED analytics pipelines and services."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class DataConfig:
    """File location and sampling configuration."""
    data_path: Path = Path("data/AED4weeks.parquet")
    sample_size: int = 400
    random_state: int = 42


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering parameters."""
    breach_threshold_minutes: int = 240
    breach_positive_labels: Tuple[str, ...] = ("breach", "1", "true")
    breach_negative_labels: Tuple[str, ...] = ("non-breach", "non breach", "0", "false")
    hrg_group_min_count: int = 30
    hrg_group_other_label: str = "Other"
    make_investigation_cluster: bool = True
    investigation_cluster_k_min: int = 2
    investigation_cluster_k_max: int = 7
    investigation_cluster_n_init: int = 20
    investigation_cluster_algorithm: str = "lloyd"
    investigation_cluster_default_k: int = 2


@dataclass(frozen=True)
class DescriptiveConfig:
    """EDA variable detection configuration."""
    categorical_threshold: int = 10
    force_categorical: Tuple[str, ...] = ("day", "period", "dayofweek", "hrg", "breachornot")
    force_numeric: Tuple[str, ...] = ("age", "los", "noofinvestigation", "nooftreatment", "noofpatients")
    ignore_columns: Tuple[str, ...] = ("id",)


@dataclass(frozen=True)
class InferenceConfig:
    """Statistical testing parameters."""
    mw_vars: Tuple[str, ...] = ("age", "noofinvestigation", "nooftreatment", "noofpatients")
    mw_screen_min_n: int = 5
    mw_diag_tie_threshold: float = 0.25
    mw_screen_use_ties: bool = False
    mw_diag_zero_threshold: float = 0.30
    mw_screen_iqr_ratio_max: float = 2.0
    categorical_vars: Tuple[str, ...] = ("dayofweek", "period", "day", "hrg_group")
    period_min_count: int = 15
    day_band_bins: Tuple[int, ...] = (0, 7, 14, 21, 28)
    day_band_labels: Tuple[str, ...] = ("Week1", "Week2", "Week3", "Week4")
    kmeans_vars: Tuple[str, ...] = ("age", "noofpatients", "noofinvestigation")
    kmeans_k_min: int = 2
    kmeans_k_max: int = 7
    kmeans_n_init: int = 20
    kmeans_algorithm: str = "lloyd"
    logit_single_predictor: str = "noofinvestigation"
    l2_numeric_predictors: Tuple[str, ...] = ("age", "noofinvestigation", "nooftreatment", "noofpatients")
    l2_categorical_predictors: Tuple[str, ...] = ("hrg_group",)
    l2_logreg_solver: str = "liblinear"
    l2_logreg_max_iter: int = 1000
    l2_logreg_c: float = 1.0
    l2_logreg_class_weight: str | None = None
    vif_predictors: Tuple[str, ...] = ("age", "noofinvestigation", "nooftreatment", "noofpatients", "hrg_group")


@dataclass(frozen=True)
class ModellingConfig:
    """Model training and evaluation parameters."""
    random_state: int = 42
    test_size: float = 0.30
    val_size_within_temp: float = 0.50
    threshold_grid: Tuple[float, ...] = tuple(np.arange(0.05, 0.51, 0.05).round(2))
    f_beta: float = 2.0
    numeric_cols: Tuple[str, ...] = ("noofinvestigation", "noofpatients", "nooftreatment", "age")
    categorical_cols: Tuple[str, ...] = ("day", "dayofweek", "period", "hrg_group")
    main_numeric: Tuple[str, ...] = ("noofinvestigation",)
    main_categorical: Tuple[str, ...] = ("hrg_group",)
    all_numeric: Tuple[str, ...] = ("age", "noofpatients")
    all_discrete_as_str: Tuple[str, ...] = ("noofinvestigation", "nooftreatment")
    all_categorical: Tuple[str, ...] = ("day", "dayofweek", "period", "hrg_group")
    representative_tree_path: Path = Path("models/tree_to_plot.joblib")
    rf_params: Dict | None = None
    logreg_params: Dict | None = None
    logreg_params_all: Dict | None = None
    xgb_params: Dict | None = None

    def __post_init__(self) -> None:
        if self.rf_params is None:
            object.__setattr__(
                self,
                "rf_params",
                dict(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=15,
                    min_samples_split=30,
                    class_weight="balanced",
                    random_state=self.random_state,
                ),
            )

        if self.logreg_params is None:
            object.__setattr__(
                self,
                "logreg_params",
                dict(
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=self.random_state,
                ),
            )

        if self.logreg_params_all is None:
            object.__setattr__(
                self,
                "logreg_params_all",
                dict(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=500,
                    random_state=self.random_state,
                ),
            )

        if self.xgb_params is None:
            object.__setattr__(
                self,
                "xgb_params",
                dict(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                ),
            )

    @property
    def rf_estimators(self) -> int:
        return int(self.rf_params["n_estimators"])

    @property
    def rf_max_depth(self) -> int:
        return int(self.rf_params["max_depth"])

    @property
    def rf_min_samples_leaf(self) -> int:
        return int(self.rf_params["min_samples_leaf"])

    @property
    def rf_min_samples_split(self) -> int:
        return int(self.rf_params["min_samples_split"])

    @property
    def rf_class_weight(self) -> str:
        return str(self.rf_params["class_weight"])

    @property
    def logreg_max_iter(self) -> int:
        return int(self.logreg_params_all.get("max_iter", 0))

    @property
    def logreg_class_weight(self) -> str:
        return str(self.logreg_params.get("class_weight", "balanced"))


@dataclass(frozen=True)
class PlotConfig:
    """Plot-related parameters."""
    day_order: Tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    figure_width: float = 8.0
    figure_height: float = 5.0
    hist_bins: int = 30
    scatter_alpha: float = 0.6
    threshold_line_width: float = 2.0
    tree_max_depth: int = 4
    tree_fig_width: float = 18.0
    tree_fig_height: float = 8.0
    tree_dpi: int = 150


@dataclass(frozen=True)
class AEDConfig:
    """Top-level configuration for AED analytics."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    descriptive: DescriptiveConfig = field(default_factory=DescriptiveConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    modelling: ModellingConfig = field(default_factory=ModellingConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
