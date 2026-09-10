#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final strict Flower_rat model rerun for the paper audit.

Outputs are written to paper_audit/final_strict_model and the main
paper_audit/AUDIT_KEY_RESULTS.csv is refreshed from the final rerun.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data/jiaxing")
AUDIT = ROOT / "paper_audit"
OUT = AUDIT / "final_strict_model"
SEED = 20260805
TARGET = "Flower_rat"
TARGET_COLUMNS = {
    "sample_id",
    "x",
    "y",
    "Flower_num",
    "Flower_rat",
    "FHB_num",
    "FHB_rate",
    "FHB_index",
}

DATASETS = {
    "point": {"dir": ROOT / "analysis_point", "token": "rpt", "label": "点采样", "radius_m": 0.0},
    "one_sqft": {"dir": ROOT / "analysis_1sqft_circle", "token": "r0p172", "label": "1 ft²", "radius_m": 0.172},
    "r0p3": {"dir": ROOT / "analysis", "token": "r0p3", "label": "0.3 m", "radius_m": 0.3},
    "r0p5": {"dir": ROOT / "analysis", "token": "r0p5", "label": "0.5 m", "radius_m": 0.5},
    "r1": {"dir": ROOT / "analysis", "token": "r1", "label": "1.0 m", "radius_m": 1.0},
}


def read_dataset(scale: str) -> pd.DataFrame:
    spec = DATASETS[scale]
    return pd.read_csv(spec["dir"] / "all_features_targets.csv").replace([np.inf, -np.inf], np.nan)


def feature_columns(df: pd.DataFrame, scale: str) -> List[str]:
    token = DATASETS[scale]["token"]
    marker = f"_{token}_"
    return [
        col
        for col in df.columns
        if col not in TARGET_COLUMNS
        and not col.endswith("_inside")
        and marker in col
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def feature_sets(cols: Sequence[str]) -> Dict[str, List[str]]:
    return {
        "0510_single_date": [c for c in cols if c.startswith("0510_")],
        "0418_single_date": [c for c in cols if c.startswith("0418_")],
        "change_only": [c for c in cols if c.startswith(("diff_", "ratio_", "rel_"))],
        "all_features": list(cols),
    }


def make_groups(df: pd.DataFrame) -> np.ndarray:
    coords = df[["x", "y"]].to_numpy(float)
    return KMeans(n_clusters=5, random_state=SEED, n_init=50).fit_predict(coords)


def build_pipeline(model_name: str, n_features: int, k: int | None = None) -> Pipeline:
    k = min(k or 10, n_features)
    if model_name == "DummyMean":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="mean"))])
    if model_name == "SingleFeatureLinear":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", SelectKBest(f_regression, k=1)),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if model_name == "Ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", SelectKBest(f_regression, k=k)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
    if model_name == "ElasticNet":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", SelectKBest(f_regression, k=k)),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=20000, random_state=SEED)),
            ]
        )
    if model_name == "PLS":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", SelectKBest(f_regression, k=k)),
                ("scaler", StandardScaler()),
                ("model", PLSRegression(n_components=1)),
            ]
        )
    if model_name == "RandomForest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", SelectKBest(f_regression, k=k)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def selected_feature_names(fitted: Pipeline, features: Sequence[str]) -> List[str]:
    if "selector" not in fitted.named_steps:
        return []
    selector = fitted.named_steps["selector"]
    mask = selector.get_support()
    return [name for name, keep in zip(features, mask) if keep]


def cv_predict(
    x: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    model_name: str,
    feature_names: Sequence[str],
    scale: str,
    feature_set: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    logo = LeaveOneGroupOut()
    pred = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    for fold_i, (train_idx, test_idx) in enumerate(logo.split(x, y, groups), start=1):
        model = build_pipeline(model_name, x.shape[1])
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = np.asarray(model.predict(x.iloc[test_idx])).reshape(-1)
        chosen = selected_feature_names(model, feature_names)
        fold_rows.append(
            {
                "target": TARGET,
                "scale": scale,
                "feature_set": feature_set,
                "model": model_name,
                "fold": fold_i,
                "test_group": int(groups[test_idx][0]),
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "input_feature_count": int(x.shape[1]),
                "selected_feature_count": int(len(chosen)),
                "selected_features": ";".join(chosen),
                "params": json.dumps(model.get_params(deep=False), ensure_ascii=False, default=str),
            }
        )
    return pred, pd.DataFrame(fold_rows)


def safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return np.nan
    if method == "pearson":
        return float(stats.pearsonr(a[mask], b[mask])[0])
    return float(stats.spearmanr(a[mask], b[mask])[0])


def metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "pearson_r": safe_corr(y, pred, "pearson"),
        "spearman_r": safe_corr(y, pred, "spearman"),
    }


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, metric_name: str, n_boot: int = 2000) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED + 11)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            values.append(metrics(y[idx], pred[idx])[metric_name])
        except Exception:
            continue
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def permutation_test(x: pd.DataFrame, y: pd.Series, groups: np.ndarray, model_name: str, observed_r2: float, n_perm: int) -> float:
    rng = np.random.default_rng(SEED + 23)
    ge = 0
    for i in range(n_perm):
        y_perm = pd.Series(rng.permutation(y.to_numpy(float)), index=y.index)
        pred, _ = cv_predict(x, y_perm, groups, model_name, list(x.columns), "perm", "perm")
        perm_r2 = metrics(y_perm.to_numpy(float), pred)["r2"]
        if np.isfinite(perm_r2) and perm_r2 >= observed_r2:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = read_dataset("point")
    valid_mask = pd.to_numeric(base[TARGET], errors="coerce").notna()
    y = pd.to_numeric(base.loc[valid_mask, TARGET], errors="coerce").reset_index(drop=True)
    sample_info = base.loc[valid_mask, ["sample_id", "x", "y"]].reset_index(drop=True)
    groups = make_groups(sample_info)
    sample_info.assign(spatial_group=groups).to_csv(OUT / "STRICT_SPATIAL_GROUPS.csv", index=False)

    result_rows = []
    prediction_rows = []
    fold_rows = []
    permutation_rows = []

    # Scale comparison: same model/rules across all scales.
    for scale in DATASETS:
        df = read_dataset(scale).loc[valid_mask].reset_index(drop=True)
        cols = feature_columns(df, scale)
        sets = feature_sets(cols)
        jobs = [
            ("DummyMean", "all_features", sets["all_features"]),
            ("Ridge", "0510_single_date", sets["0510_single_date"]),
            ("Ridge", "0418_single_date", sets["0418_single_date"]),
            ("Ridge", "change_only", sets["change_only"]),
            ("Ridge", "all_features", sets["all_features"]),
            ("SingleFeatureLinear", "0510_single_date", sets["0510_single_date"]),
        ]
        # Main 1 ft² model comparison.
        if scale == "one_sqft":
            jobs.extend(
                [
                    ("ElasticNet", "0510_single_date", sets["0510_single_date"]),
                    ("PLS", "0510_single_date", sets["0510_single_date"]),
                    ("RandomForest", "0510_single_date", sets["0510_single_date"]),
                    ("RandomForest", "all_features", sets["all_features"]),
                ]
            )
        for model_name, feature_set, selected_cols in jobs:
            selected_cols = list(selected_cols)
            if not selected_cols:
                continue
            x = df[selected_cols].apply(pd.to_numeric, errors="coerce")
            pred, fold_df = cv_predict(x, y, groups, model_name, selected_cols, scale, feature_set)
            m = metrics(y.to_numpy(float), pred)
            ci_l, ci_u = bootstrap_ci(y.to_numpy(float), pred, "r2")
            result_rows.append(
                {
                    "target": TARGET,
                    "scale": scale,
                    "feature_set": feature_set,
                    "model": model_name,
                    "validation_scheme": "5 spatial-cluster LeaveOneGroupOut CV; fixed parameters; selection inside training folds",
                    "n": int(len(y)),
                    "n_features": int(len(selected_cols)),
                    **m,
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                    "permutation_p": np.nan,
                    "leakage_risk": "low_in_final_rerun",
                    "result_status": "可信但需限制表述" if m["r2"] > 0 else "当前数据不支持定量预测",
                    "notes": "Final strict rerun; model permutation p filled for primary scale-comparison Ridge rows.",
                }
            )
            for idx, value in enumerate(pred):
                prediction_rows.append(
                    {
                        "sample_id": sample_info.loc[idx, "sample_id"],
                        "x": sample_info.loc[idx, "x"],
                        "y": sample_info.loc[idx, "y"],
                        "spatial_group": int(groups[idx]),
                        "target": TARGET,
                        "observed": float(y.iloc[idx]),
                        "predicted": float(value),
                        "scale": scale,
                        "feature_set": feature_set,
                        "model": model_name,
                    }
                )
            fold_rows.append(fold_df)

    results = pd.DataFrame(result_rows)

    # Permutation tests for the fixed scale-effect model: Ridge + 0510_single_date on each scale.
    n_perm = 200
    for scale in DATASETS:
        df = read_dataset(scale).loc[valid_mask].reset_index(drop=True)
        cols = feature_sets(feature_columns(df, scale))["0510_single_date"]
        x = df[cols].apply(pd.to_numeric, errors="coerce")
        mask = (results["scale"] == scale) & (results["feature_set"] == "0510_single_date") & (results["model"] == "Ridge")
        observed_r2 = float(results.loc[mask, "r2"].iloc[0])
        p_value = permutation_test(x, y, groups, "Ridge", observed_r2, n_perm)
        results.loc[mask, "permutation_p"] = p_value
        permutation_rows.append(
            {
                "target": TARGET,
                "scale": scale,
                "feature_set": "0510_single_date",
                "model": "Ridge",
                "n_permutations": n_perm,
                "observed_r2": observed_r2,
                "permutation_p": p_value,
                "random_seed": SEED + 23,
                "test_definition": "Permuted Flower_rat labels; each permutation reruns full outer spatial CV pipeline.",
            }
        )

    scale_cmp = results[(results["feature_set"] == "0510_single_date") & (results["model"] == "Ridge")].copy()
    scale_cmp.insert(2, "scale_label", scale_cmp["scale"].map({k: v["label"] for k, v in DATASETS.items()}))
    scale_cmp.insert(3, "radius_m", scale_cmp["scale"].map({k: v["radius_m"] for k, v in DATASETS.items()}))
    scale_cmp = scale_cmp.sort_values("radius_m")

    results.to_csv(OUT / "AUDIT_KEY_RESULTS.csv", index=False)
    scale_cmp.to_csv(OUT / "STRICT_SCALE_COMPARISON.csv", index=False)
    pd.DataFrame(permutation_rows).to_csv(OUT / "MODEL_PERMUTATION_TEST.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "OOF_PREDICTIONS.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(OUT / "FOLD_FEATURE_COUNTS_AND_PARAMS.csv", index=False)

    # Refresh top-level audit key results for the final rerun.
    shutil.copy2(OUT / "AUDIT_KEY_RESULTS.csv", AUDIT / "AUDIT_KEY_RESULTS.csv")
    print(f"Wrote final strict model outputs to {OUT}")


if __name__ == "__main__":
    run()
