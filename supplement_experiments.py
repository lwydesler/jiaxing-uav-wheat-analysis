#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary experiments for UAV multispectral field-sample analysis.

This script reuses existing analysis tables from point, 1-square-foot, and
larger-buffer experiments. It creates paper-ready summary tables and figures
for scale comparison, temporal feature response, feature group interpretation,
target distribution, and disease-presence classification.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import matplotlib
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    print(
        "ERROR: missing dependency. Run with the rs conda environment, for example:\n"
        "  conda run -n rs python supplement_experiments.py",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


LOGGER = logging.getLogger("supplement_experiments")

TARGETS = ["Flower_num", "Flower_rat", "FHB_num", "FHB_rate", "FHB_index"]
ID_COLUMNS = {"sample_id", "x", "y"}
KNOWN_STATS = {"mean", "median", "std", "min", "max", "p25", "p75", "valid_count"}
NAMED_INDICES = {
    "NDVI",
    "GNDVI",
    "NDRE1",
    "NDRE2",
    "RENDVI1",
    "RENDVI2",
    "RVI",
    "DVI",
    "SAVI",
    "EVI",
    "CI_RE1",
    "CI_RE2",
}
DATASET_DEFAULTS = {
    "point": "/data/jiaxing/analysis_point",
    "one_sqft": "/data/jiaxing/analysis_1sqft_circle",
    "multi_scale": "/data/jiaxing/analysis",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    directory: Path
    display_name: str


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build supplementary experiment tables and figures."
    )
    parser.add_argument("--point-dir", default=DATASET_DEFAULTS["point"])
    parser.add_argument("--one-sqft-dir", default=DATASET_DEFAULTS["one_sqft"])
    parser.add_argument("--multi-scale-dir", default=DATASET_DEFAULTS["multi_scale"])
    parser.add_argument("--out-dir", default="/data/jiaxing/supplement_experiments")
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top correlations/features retained in summary tables.",
    )
    parser.add_argument(
        "--class-feature-limit",
        type=int,
        default=25,
        help="Number of top remote-sensing features used in classification.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_matplotlib_fonts() -> None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            font_name = font_manager.FontProperties(fname=path).get_name()
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    LOGGER.warning("No CJK font found; Chinese labels in figures may not render correctly.")


def require_csv(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required table not found: {path}")
    return path


def load_dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    specs = [
        DatasetSpec("point", Path(args.point_dir), "点采样"),
        DatasetSpec("one_sqft", Path(args.one_sqft_dir), "1平方尺"),
        DatasetSpec("multi_scale", Path(args.multi_scale_dir), "0.3/0.5/1.0m"),
    ]
    for spec in specs:
        require_csv(spec.directory / "all_features_targets.csv")
        require_csv(spec.directory / "correlation_summary.csv")
        require_csv(spec.directory / "model_metrics.csv")
    return specs


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)


def feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = set(TARGETS) | ID_COLUMNS
    return [
        col
        for col in df.columns
        if col not in excluded
        and not col.endswith("_inside")
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def parse_feature_name(feature: str) -> Dict[str, str]:
    parts = feature.split("_")
    temporal = parts[0] if parts else "unknown"
    scale = "unknown"
    stat_name = "unknown"

    for part in parts:
        if re.fullmatch(r"r(?:pt|\d+p?\d*)", part):
            scale = part
            break

    if parts and parts[-1] in KNOWN_STATS:
        stat_name = parts[-1]

    core = feature
    if scale != "unknown" and stat_name != "unknown":
        pattern = f"_{scale}_"
        if pattern in feature:
            core = feature.split(pattern, 1)[1]
            core = core[: -(len(stat_name) + 1)]

    if temporal in {"0418", "0510"}:
        temporal_group = temporal
    elif temporal in {"diff", "ratio", "rel"}:
        temporal_group = temporal
    else:
        temporal_group = "other"

    if re.fullmatch(r"B\d+", core):
        feature_type = "raw_band"
    elif core in NAMED_INDICES:
        feature_type = "named_index"
    elif core.startswith("ND_B"):
        feature_type = "pairwise_nd"
    elif core == "valid_count":
        feature_type = "valid_count"
    else:
        feature_type = "other"

    return {
        "feature": feature,
        "temporal": temporal_group,
        "scale": scale,
        "stat": stat_name,
        "feature_core": core,
        "feature_type": feature_type,
    }


def read_correlations(spec: DatasetSpec) -> pd.DataFrame:
    corr = pd.read_csv(spec.directory / "correlation_summary.csv")
    corr.insert(0, "dataset", spec.name)
    corr.insert(1, "dataset_name", spec.display_name)
    meta = pd.DataFrame([parse_feature_name(f) for f in corr["feature"]])
    return pd.concat([corr.reset_index(drop=True), meta.drop(columns=["feature"])], axis=1)


def read_models(spec: DatasetSpec) -> pd.DataFrame:
    models = pd.read_csv(spec.directory / "model_metrics.csv")
    models.insert(0, "dataset", spec.name)
    models.insert(1, "dataset_name", spec.display_name)
    return models


def make_target_distribution(specs: Sequence[DatasetSpec], out_dir: Path) -> pd.DataFrame:
    # Targets are identical across datasets, so use point table as the canonical sample table.
    df = pd.read_csv(specs[0].directory / "all_features_targets.csv")
    rows: List[Dict[str, float]] = []
    for target in TARGETS:
        values = pd.to_numeric(df[target], errors="coerce").dropna()
        row = {
            "target": target,
            "valid_n": int(values.size),
            "missing_n": int(df[target].isna().sum()),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "min": values.min(),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "max": values.max(),
            "zero_n": int((values == 0).sum()),
            "positive_n": int((values > 0).sum()),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "target_distribution_summary.csv", index=False)
    plot_target_distribution(df, out_dir / "figures" / "target_distribution.png")
    return summary


def plot_target_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        values = pd.to_numeric(df[target], errors="coerce").dropna()
        ax.hist(values, bins=min(12, max(5, int(math.sqrt(max(values.size, 1))))), color="#5278a8", edgecolor="white")
        ax.set_title(target)
        ax.set_ylabel("样点数")
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    fig.suptitle("田间指标分布", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_scale_comparison(corr_all: pd.DataFrame, out_dir: Path, top_n: int) -> pd.DataFrame:
    data = corr_all.copy()
    data = data.sort_values(["target", "dataset", "abs_spearman_r"], ascending=[True, True, False])
    top = data.groupby(["target", "dataset"], as_index=False).head(top_n)
    top.to_csv(out_dir / "scale_comparison_top_correlations.csv", index=False)

    best = data.groupby(["target", "dataset", "dataset_name"], as_index=False).first()
    best = best[
        [
            "target",
            "dataset",
            "dataset_name",
            "feature",
            "n",
            "spearman_r",
            "spearman_p",
            "pearson_r",
            "pearson_p",
            "abs_spearman_r",
            "temporal",
            "scale",
            "feature_type",
            "stat",
        ]
    ]
    best.to_csv(out_dir / "best_correlation_by_scale.csv", index=False)
    plot_best_correlation_by_scale(best, out_dir / "figures" / "best_correlation_by_scale.png")
    return best


def plot_best_correlation_by_scale(best: pd.DataFrame, path: Path) -> None:
    piv = best.pivot(index="target", columns="dataset_name", values="abs_spearman_r").reindex(TARGETS)
    ax = piv.plot(kind="bar", figsize=(11, 5), color=["#5c85b1", "#d1884f", "#5e9b76"])
    ax.set_ylabel("|Spearman r|")
    ax.set_xlabel("")
    ax.set_title("不同采样尺度下的最强相关程度")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_model_comparison(models_all: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    models_all = models_all.copy()
    models_all["r2_rank_value"] = models_all["r2"].replace([np.inf, -np.inf], np.nan)
    models_all.to_csv(out_dir / "model_comparison_all.csv", index=False)

    best_rows: List[pd.Series] = []
    for (_, _), group in models_all.groupby(["target", "dataset"]):
        good = group.dropna(subset=["r2_rank_value"])
        if good.empty:
            best_rows.append(group.iloc[0])
        else:
            best_rows.append(good.sort_values("r2_rank_value", ascending=False).iloc[0])
    best = pd.DataFrame(best_rows).drop(columns=["r2_rank_value"], errors="ignore")
    best = best.sort_values(["target", "dataset"])
    best.to_csv(out_dir / "best_model_by_scale.csv", index=False)
    plot_best_model_r2(best, out_dir / "figures" / "best_model_r2_by_scale.png")
    return best


def plot_best_model_r2(best: pd.DataFrame, path: Path) -> None:
    piv = best.pivot(index="target", columns="dataset_name", values="r2").reindex(TARGETS)
    ax = piv.plot(kind="bar", figsize=(11, 5), color=["#5c85b1", "#d1884f", "#5e9b76"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("交叉验证 R2")
    ax.set_xlabel("")
    ax.set_title("不同采样尺度下的最佳回归模型")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_feature_group_summary(corr_all: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        corr_all.groupby(["target", "dataset", "dataset_name", "temporal", "feature_type"], as_index=False)
        .agg(
            feature_count=("feature", "count"),
            mean_abs_spearman=("abs_spearman_r", "mean"),
            max_abs_spearman=("abs_spearman_r", "max"),
        )
        .sort_values(["target", "dataset", "max_abs_spearman"], ascending=[True, True, False])
    )
    summary.to_csv(out_dir / "feature_group_summary.csv", index=False)

    temporal = (
        corr_all.groupby(["target", "dataset", "dataset_name", "temporal"], as_index=False)
        .agg(
            feature_count=("feature", "count"),
            mean_abs_spearman=("abs_spearman_r", "mean"),
            max_abs_spearman=("abs_spearman_r", "max"),
        )
        .sort_values(["target", "dataset", "max_abs_spearman"], ascending=[True, True, False])
    )
    temporal.to_csv(out_dir / "temporal_feature_summary.csv", index=False)
    plot_temporal_summary(temporal, out_dir / "figures" / "temporal_feature_summary.png")
    return summary, temporal


def plot_temporal_summary(temporal: pd.DataFrame, path: Path) -> None:
    order = ["0418", "0510", "diff", "ratio", "rel"]
    focus = temporal[temporal["dataset"].isin(["point", "one_sqft"])].copy()
    rows = []
    for target in TARGETS:
        for dataset in ["point", "one_sqft"]:
            subset = focus[(focus["target"] == target) & (focus["dataset"] == dataset)]
            for temporal_name in order:
                value = subset.loc[subset["temporal"] == temporal_name, "max_abs_spearman"]
                rows.append(
                    {
                        "target": target,
                        "dataset": dataset,
                        "temporal": temporal_name,
                        "value": float(value.iloc[0]) if not value.empty else np.nan,
                    }
                )
    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, dataset, title in zip(axes, ["point", "one_sqft"], ["点采样", "1平方尺"]):
        sub = plot_df[plot_df["dataset"] == dataset]
        x = np.arange(len(TARGETS))
        width = 0.15
        for idx, temporal_name in enumerate(order):
            vals = [
                sub[(sub["target"] == target) & (sub["temporal"] == temporal_name)]["value"].iloc[0]
                for target in TARGETS
            ]
            ax.bar(x + (idx - 2) * width, vals, width=width, label=temporal_name)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("分组内最大 |Spearman r|")
    axes[1].legend(title="时间特征", fontsize=8)
    fig.suptitle("单期与变化特征的响应强度")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def corr_score_for_class(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return 0.0
    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return 0.0
    try:
        score, _ = stats.spearmanr(x[mask], y[mask])
    except Exception:
        return 0.0
    if pd.isna(score):
        return 0.0
    return float(abs(score))


def disease_presence_target(df: pd.DataFrame) -> pd.Series:
    disease_cols = ["FHB_num", "FHB_rate", "FHB_index"]
    available = df[disease_cols].apply(pd.to_numeric, errors="coerce")
    has_any_observed = available.notna().any(axis=1)
    presence = (available.fillna(0) > 0).any(axis=1).astype(float)
    presence.loc[~has_any_observed] = np.nan
    return presence


def evaluate_classifier(
    df: pd.DataFrame,
    dataset: DatasetSpec,
    model_name: str,
    estimator,
    feature_limit: int,
    random_state: int,
) -> Optional[Dict[str, object]]:
    y = disease_presence_target(df)
    mask = y.notna()
    y_valid = y.loc[mask].astype(int)
    if y_valid.nunique() < 2:
        LOGGER.warning("%s: disease presence has only one class; skipped.", dataset.name)
        return None
    class_counts = y_valid.value_counts()
    min_class = int(class_counts.min())
    if min_class < 3:
        LOGGER.warning("%s: too few samples in one disease class; skipped.", dataset.name)
        return None

    candidates = feature_columns(df)
    scores = [(col, corr_score_for_class(pd.to_numeric(df.loc[mask, col], errors="coerce"), y_valid)) for col in candidates]
    top_features = [col for col, score in sorted(scores, key=lambda item: item[1], reverse=True)[:feature_limit] if score > 0]
    if len(top_features) < 2:
        LOGGER.warning("%s: not enough informative features for classification.", dataset.name)
        return None

    x = df.loc[mask, top_features].replace([np.inf, -np.inf], np.nan)
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred = cross_val_predict(estimator, x, y_valid, cv=cv, method="predict")
    proba = None
    try:
        proba = cross_val_predict(estimator, x, y_valid, cv=cv, method="predict_proba")[:, 1]
    except Exception:
        proba = None

    auc = np.nan
    if proba is not None and y_valid.nunique() == 2:
        auc = roc_auc_score(y_valid, proba)

    return {
        "dataset": dataset.name,
        "dataset_name": dataset.display_name,
        "classification_target": "FHB_presence",
        "model": model_name,
        "cv": f"StratifiedKFold_{n_splits}",
        "n": int(y_valid.size),
        "positive_n": int((y_valid == 1).sum()),
        "negative_n": int((y_valid == 0).sum()),
        "n_features": len(top_features),
        "accuracy": accuracy_score(y_valid, pred),
        "balanced_accuracy": balanced_accuracy_score(y_valid, pred),
        "f1": f1_score(y_valid, pred, zero_division=0),
        "roc_auc": auc,
        "features": ";".join(top_features),
    }


def make_classification_experiment(
    specs: Sequence[DatasetSpec],
    out_dir: Path,
    feature_limit: int,
    random_state: int,
) -> pd.DataFrame:
    classifiers = [
        (
            "LogisticRegression",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "RandomForestClassifier",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_features="sqrt",
                            min_samples_leaf=3,
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]

    rows: List[Dict[str, object]] = []
    for spec in specs:
        if spec.name == "multi_scale":
            continue
        df = pd.read_csv(spec.directory / "all_features_targets.csv")
        for model_name, estimator in classifiers:
            row = evaluate_classifier(
                df=df,
                dataset=spec,
                model_name=model_name,
                estimator=estimator,
                feature_limit=feature_limit,
                random_state=random_state,
            )
            if row is not None:
                rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "disease_classification_metrics.csv", index=False)
    if not result.empty:
        plot_classification(result, out_dir / "figures" / "disease_classification_metrics.png")
    return result


def plot_classification(metrics: pd.DataFrame, path: Path) -> None:
    labels = metrics["dataset_name"] + "\n" + metrics["model"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, metrics["balanced_accuracy"], width=0.36, label="Balanced accuracy", color="#5278a8")
    ax.bar(x + 0.18, metrics["roc_auc"], width=0.36, label="ROC AUC", color="#d1884f")
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("分数")
    ax.set_title("病害有无分类实验")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_paper_summary(
    out_dir: Path,
    target_summary: pd.DataFrame,
    best_corr: pd.DataFrame,
    best_models: pd.DataFrame,
    class_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# 补充实验结果摘要",
        "",
        "## 1. 田间指标样本量",
        "",
        "| 指标 | 有效样本数 | 最小值 | 中位数 | 最大值 | 正值样本数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in target_summary.iterrows():
        lines.append(
            f"| {row['target']} | {int(row['valid_n'])} | {row['min']:.3f} | "
            f"{row['median']:.3f} | {row['max']:.3f} | {int(row['positive_n'])} |"
        )

    lines.extend(
        [
            "",
            "## 2. 各尺度最强相关特征",
            "",
            "| 指标 | 尺度 | 最强特征 | Spearman r | p值 |",
            "|---|---|---|---:|---:|",
        ]
    )
    for _, row in best_corr.sort_values(["target", "dataset"]).iterrows():
        lines.append(
            f"| {row['target']} | {row['dataset_name']} | {row['feature']} | "
            f"{row['spearman_r']:.3f} | {row['spearman_p']:.4g} |"
        )

    lines.extend(
        [
            "",
            "## 3. 各尺度最佳回归模型",
            "",
            "| 指标 | 尺度 | 模型 | n | R2 | RMSE | MAE |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in best_models.sort_values(["target", "dataset"]).iterrows():
        lines.append(
            f"| {row['target']} | {row['dataset_name']} | {row['model']} | {int(row['n'])} | "
            f"{row['r2']:.3f} | {row['rmse']:.3f} | {row['mae']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 病害有无分类",
            "",
            "病害有无由 FHB_num、FHB_rate、FHB_index 中任一指标大于 0 判定。",
            "",
            "| 尺度 | 模型 | n | 阳性 | 阴性 | Balanced accuracy | ROC AUC | F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if class_metrics.empty:
        lines.append("| - | - | - | - | - | - | - | - |")
    else:
        for _, row in class_metrics.iterrows():
            lines.append(
                f"| {row['dataset_name']} | {row['model']} | {int(row['n'])} | "
                f"{int(row['positive_n'])} | {int(row['negative_n'])} | "
                f"{row['balanced_accuracy']:.3f} | {row['roc_auc']:.3f} | {row['f1']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## 5. 写作建议",
            "",
            "这些补充实验适合支撑论文中的“尺度效应、时间变化特征、特征类型响应、回归预测和病害分类”五个小节。"
            "当前样本量有限，回归预测宜表述为趋势估计或相对风险制图，不宜表述为高精度业务化反演。",
            "",
        ]
    )
    (out_dir / "paper_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()
    setup_matplotlib_fonts()
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)
    specs = load_dataset_specs(args)

    LOGGER.info("Writing supplementary outputs to %s", out_dir)
    target_summary = make_target_distribution(specs, out_dir)
    corr_all = pd.concat([read_correlations(spec) for spec in specs], ignore_index=True)
    corr_all.to_csv(out_dir / "correlation_summary_with_feature_groups.csv", index=False)
    best_corr = make_scale_comparison(corr_all, out_dir, args.top_n)
    models_all = pd.concat([read_models(spec) for spec in specs], ignore_index=True)
    best_models = make_model_comparison(models_all, out_dir)
    make_feature_group_summary(corr_all, out_dir)
    class_metrics = make_classification_experiment(
        specs=specs,
        out_dir=out_dir,
        feature_limit=args.class_feature_limit,
        random_state=args.random_state,
    )
    make_paper_summary(out_dir, target_summary, best_corr, best_models, class_metrics)

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
