"""Configuration for AED analytics pipelines and services."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


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
    investigation_cluster_bins: Tuple[int, ...] = (-1, 0, 1, 2, 10000)
    investigation_cluster_labels: Tuple[int, ...] = (0, 1, 2, 3)


@dataclass(frozen=True)
class InferenceConfig:
    """Statistical testing parameters."""
    mw_vars: Tuple[str, ...] = ("age", "noofinvestigation", "nooftreatment", "noofpatients")
    mw_screen_min_n: int = 5
    mw_diag_tie_threshold: float = 0.25
    mw_screen_use_ties: bool = False
    mw_diag_zero_threshold: float = 0.30
    mw_screen_iqr_ratio_max: float = 2.0
    categorical_vars: Tuple[str, ...] = ("dayofweek", "period", "hrg")
    logit_single_predictor: str = "noofinvestigation"
    l2_numeric_predictors: Tuple[str, ...] = ("age", "noofinvestigation", "nooftreatment", "noofpatients")
    l2_categorical_predictors: Tuple[str, ...] = ("hrg_group",)
    l2_logreg_solver: str = "liblinear"
    l2_logreg_max_iter: int = 1000
    l2_logreg_class_weight: str | None = None
    vif_predictors: Tuple[str, ...] = ("age", "noofinvestigation", "nooftreatment", "noofpatients", "hrg_group")


@dataclass(frozen=True)
class ModellingConfig:
    """Model training and evaluation parameters."""
    test_size: float = 0.30
    val_size_within_temp: float = 0.50
    random_state: int = 42
    threshold_grid: Tuple[float, ...] = tuple(round(x, 2) for x in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    rf_estimators: int = 200
    rf_max_depth: int = 4
    rf_min_samples_leaf: int = 15
    rf_min_samples_split: int = 30
    rf_class_weight: str = "balanced"
    logreg_max_iter: int = 2000
    logreg_class_weight: str = "balanced"
    f_beta: float = 2.0
    main_numeric: Tuple[str, ...] = ("noofinvestigation",)
    main_categorical: Tuple[str, ...] = ("hrg_group",)
    all_numeric: Tuple[str, ...] = ("age", "noofpatients")
    all_discrete_as_str: Tuple[str, ...] = ("noofinvestigation", "nooftreatment")
    all_categorical: Tuple[str, ...] = ("day", "dayofweek", "period", "hrg_group")


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
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    modelling: ModellingConfig = field(default_factory=ModellingConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
