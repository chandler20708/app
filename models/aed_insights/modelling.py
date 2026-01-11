"""Predictive modelling utilities and artifacts for AED insights."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    auc,
)
from sklearn.model_selection import train_test_split

from models.aed_insights.config import AEDConfig
from models.aed_insights.core import get_schema


@dataclass(frozen=True)
class ModellingArtifacts:
    """Container for modelling tables and figures."""
    tables: Dict[str, pd.DataFrame]
    figures: Dict[str, object]
    metadata: Dict[str, Any]


def _filter_existing(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def build_design_matrix(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    discrete_as_str: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Construct X and y with numeric, one-hot categorical, and optional discrete encodings."""
    discrete_as_str = list(discrete_as_str or [])

    numeric_cols = _filter_existing(df, numeric_features)
    categorical_cols = _filter_existing(df, categorical_features)
    discrete_cols = _filter_existing(df, discrete_as_str)

    parts: List[pd.DataFrame] = []
    if numeric_cols:
        parts.append(df[numeric_cols].astype(float))
    if discrete_cols:
        parts.append(pd.get_dummies(df[discrete_cols].astype(str), prefix=discrete_cols))
    if categorical_cols:
        parts.append(pd.get_dummies(df[categorical_cols], drop_first=False))

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    y = df[target_col].astype(int)
    return X, y, list(X.columns)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    f_beta: float,
) -> Dict[str, float]:
    """Compute classification metrics for a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_breach": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_non_breach": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_breach": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_non_breach": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f2_breach": fbeta_score(y_true, y_pred, beta=f_beta, pos_label=1, zero_division=0),
        "f2_non_breach": fbeta_score(y_true, y_pred, beta=f_beta, pos_label=0, zero_division=0),
        "fp": float(fp),
        "fn": float(fn),
        "pr_auc": average_precision_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def _pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: AEDConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=config.modelling.test_size,
        stratify=y,
        random_state=config.modelling.random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=config.modelling.val_size_within_temp,
        stratify=y_temp,
        random_state=config.modelling.random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, config: AEDConfig) -> float:
    best_t = config.modelling.threshold_grid[0]
    best_f2 = -1.0
    for threshold in config.modelling.threshold_grid:
        pred = (y_prob >= threshold).astype(int)
        f2 = fbeta_score(y_true, pred, beta=config.modelling.f_beta, pos_label=1, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_t = threshold
    return float(best_t)


def _evaluate_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    config: AEDConfig,
) -> Dict[str, Any]:
    model.fit(X_train, y_train)

    y_val_prob = model.predict_proba(X_val)[:, 1]
    threshold = _tune_threshold(y_val.values, y_val_prob, config)

    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, digits=6, output_dict=True)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_prob >= threshold).astype(int)
    y_val_pred = (y_val_prob >= threshold).astype(int)

    return {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": report["accuracy"],
        "Recall_Breach": report["1"]["recall"],
        "Recall_NonBreach": report["0"]["recall"],
        "Precision_Breach": report["1"]["precision"],
        "Precision_NonBreach": report["0"]["precision"],
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "F2_Breach": fbeta_score(y_test, y_test_pred, beta=config.modelling.f_beta, pos_label=1, zero_division=0),
        "F2_NonBreach": fbeta_score(y_test, y_test_pred, beta=config.modelling.f_beta, pos_label=0, zero_division=0),
        "Train_PR_AUC": _pr_auc(y_train.values, y_train_prob),
        "Val_PR_AUC": _pr_auc(y_val.values, y_val_prob),
        "Test_PR_AUC": _pr_auc(y_test.values, y_test_prob),
        "Train_BalAcc": balanced_accuracy_score(y_train, y_train_pred),
        "Val_BalAcc": balanced_accuracy_score(y_val, y_val_pred),
        "Test_BalAcc": balanced_accuracy_score(y_test, y_test_pred),
    }


def _evaluate_dummy(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    config: AEDConfig,
) -> Dict[str, Any]:
    dummy = DummyClassifier(strategy="stratified", random_state=config.modelling.random_state)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)

    y_train_prob = dummy.predict_proba(np.zeros((len(y_train), 1)))[:, 1]
    y_val_prob = dummy.predict_proba(np.zeros((len(y_val), 1)))[:, 1]
    y_test_prob = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]

    y_train_pred = dummy.predict(np.zeros((len(y_train), 1)))
    y_val_pred = dummy.predict(np.zeros((len(y_val), 1)))
    y_test_pred = dummy.predict(np.zeros((len(y_test), 1)))

    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, digits=6, output_dict=True)

    return {
        "Model": "Dummy",
        "Threshold": np.nan,
        "Accuracy": report["accuracy"],
        "Recall_Breach": report["1"]["recall"],
        "Recall_NonBreach": report["0"]["recall"],
        "Precision_Breach": report["1"]["precision"],
        "Precision_NonBreach": report["0"]["precision"],
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "F2_Breach": fbeta_score(y_test, y_test_pred, beta=config.modelling.f_beta, pos_label=1, zero_division=0),
        "F2_NonBreach": fbeta_score(y_test, y_test_pred, beta=config.modelling.f_beta, pos_label=0, zero_division=0),
        "Train_PR_AUC": _pr_auc(y_train.values, y_train_prob),
        "Val_PR_AUC": _pr_auc(y_val.values, y_val_prob),
        "Test_PR_AUC": _pr_auc(y_test.values, y_test_prob),
        "Train_BalAcc": balanced_accuracy_score(y_train, y_train_pred),
        "Val_BalAcc": balanced_accuracy_score(y_val, y_val_pred),
        "Test_BalAcc": balanced_accuracy_score(y_test, y_test_pred),
    }


def train_models(
    df: pd.DataFrame,
    config: AEDConfig,
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Train models and assemble the two required modelling tables."""
    schema = get_schema()
    target_col = schema.breach_flag_col

    results: List[Dict[str, Any]] = []
    models: Dict[str, Any] = {}

    X_main, y_main, main_features = build_design_matrix(
        df,
        target_col,
        config.modelling.main_numeric,
        config.modelling.main_categorical,
    )
    X_train, X_val_main, X_test, y_train, y_val, y_test = _split_data(X_main, y_main, config)
    results.append(_evaluate_dummy(X_train, X_val_main, X_test, y_train, y_val, y_test, config))

    rf_main = RandomForestClassifier(
        n_estimators=config.modelling.rf_estimators,
        max_depth=config.modelling.rf_max_depth,
        min_samples_leaf=config.modelling.rf_min_samples_leaf,
        min_samples_split=config.modelling.rf_min_samples_split,
        class_weight=config.modelling.rf_class_weight,
        random_state=config.modelling.random_state,
    )
    results.append(_evaluate_model("RF_Main", rf_main, X_train, X_val_main, X_test, y_train, y_val, y_test, config))
    models["rf_main"] = rf_main

    X_rf_all, y_rf_all, rf_all_features = build_design_matrix(
        df,
        target_col,
        tuple(config.modelling.all_numeric) + tuple(config.modelling.all_discrete_as_str),
        config.modelling.all_categorical,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(X_rf_all, y_rf_all, config)
    rf_all = RandomForestClassifier(
        n_estimators=config.modelling.rf_estimators,
        max_depth=config.modelling.rf_max_depth,
        min_samples_leaf=config.modelling.rf_min_samples_leaf,
        min_samples_split=config.modelling.rf_min_samples_split,
        class_weight=config.modelling.rf_class_weight,
        random_state=config.modelling.random_state,
    )
    results.append(_evaluate_model("RF_All", rf_all, X_train, X_val, X_test, y_train, y_val, y_test, config))
    models["rf_all"] = rf_all

    X_lr_main, y_lr_main, lr_main_features = build_design_matrix(
        df,
        target_col,
        config.modelling.main_numeric,
        config.modelling.main_categorical,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(X_lr_main, y_lr_main, config)
    lr_main = LogisticRegression(
        class_weight=config.modelling.logreg_class_weight,
        solver="liblinear",
        random_state=config.modelling.random_state,
        max_iter=config.modelling.logreg_max_iter,
    )
    results.append(_evaluate_model("LR_Main", lr_main, X_train, X_val, X_test, y_train, y_val, y_test, config))
    models["lr_main"] = lr_main

    X_lr_all, y_lr_all, lr_all_features = build_design_matrix(
        df,
        target_col,
        config.modelling.all_numeric,
        config.modelling.all_categorical,
        discrete_as_str=config.modelling.all_discrete_as_str,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(X_lr_all, y_lr_all, config)
    lr_all = LogisticRegression(
        class_weight=config.modelling.logreg_class_weight,
        solver="liblinear",
        random_state=config.modelling.random_state,
        max_iter=config.modelling.logreg_max_iter,
    )
    results.append(_evaluate_model("LR_All", lr_all, X_train, X_val, X_test, y_train, y_val, y_test, config))
    models["lr_all"] = lr_all

    df_all = pd.DataFrame(results)
    df_sorted = df_all.sort_values(
        by=["Recall_Breach", "F2_Breach", "FN"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    df_full = df_sorted[
        [
            "Model",
            "Threshold",
            "Accuracy",
            "Recall_Breach",
            "Recall_NonBreach",
            "Precision_Breach",
            "Precision_NonBreach",
            "FP",
            "FN",
            "F2_Breach",
            "F2_NonBreach",
        ]
    ].copy()
    df_full.insert(0, "S.No", np.arange(1, len(df_full) + 1))

    df_pr = df_sorted[
        [
            "Model",
            "Train_PR_AUC",
            "Val_PR_AUC",
            "Test_PR_AUC",
            "Train_BalAcc",
            "Val_BalAcc",
            "Test_BalAcc",
        ]
    ].copy()
    df_pr.insert(0, "S.No", np.arange(1, len(df_pr) + 1))

    tables = {
        "full_metrics_table": df_full,
        "pr_auc_balanced_accuracy": df_pr,
    }
    metadata = {
        "rf_main": rf_main,
        "rf_main_features": main_features,
        "rf_main_val": X_val_main,
        "rf_all_features": rf_all_features,
        "lr_main_features": lr_main_features,
        "lr_all_features": lr_all_features,
        "sample_size": len(df),
    }
    return models, tables, metadata
