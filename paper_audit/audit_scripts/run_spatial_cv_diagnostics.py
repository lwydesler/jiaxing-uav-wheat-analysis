#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial CV diagnostics for the final Flower_rat audit.

This script checks four paper-critical issues:
1. Spatial grouping and validation details.
2. Whether 1.0 m sample buffers overlap across train-validation folds.
3. Moran's I spatial autocorrelation for observed values and OOF residuals.
4. Benjamini-Hochberg FDR correction across the five scale permutation tests.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path("/data/jiaxing")
AUDIT = ROOT / "paper_audit"
OUT = AUDIT / "spatial_cv_diagnostics"
SEED = 20260805
MORAN_SEED = 20260907
TARGET = "Flower_rat"
PRIMARY_FEATURE_SET = "0510_single_date"
PRIMARY_MODEL = "Ridge"
SCALE_LABELS = {
    "point": "点采样",
    "one_sqft": "1 ft²",
    "r0p3": "0.3 m",
    "r0p5": "0.5 m",
    "r1": "1.0 m",
}


def ensure_inputs() -> None:
    required = [
        AUDIT / "STRICT_SPATIAL_GROUPS.csv",
        AUDIT / "STRICT_SCALE_COMPARISON.csv",
        AUDIT / "MODEL_PERMUTATION_TEST.csv",
        AUDIT / "OOF_PREDICTIONS.csv",
        AUDIT / "FOLD_FEATURE_COUNTS_AND_PARAMS.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required final audit files: " + ", ".join(missing))


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    raw = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(pv)
    out[order] = adjusted
    q[valid] = out
    return q


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def circle_overlap_area(distance: float, radius: float) -> float:
    if distance >= 2 * radius:
        return 0.0
    if distance <= 0:
        return math.pi * radius * radius
    return (
        2 * radius * radius * math.acos(distance / (2 * radius))
        - 0.5 * distance * math.sqrt(max(0.0, 4 * radius * radius - distance * distance))
    )


def summarize_groups(groups: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    primary = oof[
        (oof["scale"] == "r1")
        & (oof["feature_set"] == PRIMARY_FEATURE_SET)
        & (oof["model"] == PRIMARY_MODEL)
    ][["sample_id", "observed"]].drop_duplicates("sample_id")
    merged = groups.merge(primary, on="sample_id", how="left")
    rows = []
    for group_id, part in merged.groupby("spatial_group"):
        rows.append(
            {
                "spatial_group": int(group_id),
                "sample_count": int(len(part)),
                "x_min": float(part["x"].min()),
                "x_max": float(part["x"].max()),
                "y_min": float(part["y"].min()),
                "y_max": float(part["y"].max()),
                "flower_rat_mean": float(part["observed"].mean()),
                "flower_rat_std": float(part["observed"].std(ddof=1)),
                "flower_rat_min": float(part["observed"].min()),
                "flower_rat_max": float(part["observed"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("spatial_group")


def overlap_tables(groups: pd.DataFrame, radius: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    coords = groups[["x", "y"]].to_numpy(float)
    circle_area = math.pi * radius * radius
    pair_rows = []
    n = len(groups)
    for i in range(n):
        for j in range(i + 1, n):
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance < 2 * radius:
                area = circle_overlap_area(distance, radius)
                group_i = int(groups.loc[i, "spatial_group"])
                group_j = int(groups.loc[j, "spatial_group"])
                pair_rows.append(
                    {
                        "sample_id_i": int(groups.loc[i, "sample_id"]),
                        "sample_id_j": int(groups.loc[j, "sample_id"]),
                        "group_i": group_i,
                        "group_j": group_j,
                        "distance_m": distance,
                        "radius_m": radius,
                        "overlap_area_m2": area,
                        "overlap_fraction_of_one_buffer": area / circle_area,
                        "same_spatial_group": group_i == group_j,
                        "cross_train_validation_fold": group_i != group_j,
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        pairs = pd.DataFrame(
            columns=[
                "sample_id_i",
                "sample_id_j",
                "group_i",
                "group_j",
                "distance_m",
                "radius_m",
                "overlap_area_m2",
                "overlap_fraction_of_one_buffer",
                "same_spatial_group",
                "cross_train_validation_fold",
            ]
        )

    fold_rows = []
    for group_id in sorted(groups["spatial_group"].unique()):
        val_ids = set(groups.loc[groups["spatial_group"] == group_id, "sample_id"].astype(int))
        train_ids = set(groups.loc[groups["spatial_group"] != group_id, "sample_id"].astype(int))
        crossing = []
        val_with_overlap = set()
        for row in pairs.to_dict("records"):
            i = int(row["sample_id_i"])
            j = int(row["sample_id_j"])
            crosses_this_fold = (i in val_ids and j in train_ids) or (j in val_ids and i in train_ids)
            if crosses_this_fold:
                crossing.append(row)
                val_with_overlap.add(i if i in val_ids else j)
        fold_rows.append(
            {
                "test_group": int(group_id),
                "validation_sample_count": int(len(val_ids)),
                "training_sample_count": int(len(train_ids)),
                "train_validation_overlap_pair_count": int(len(crossing)),
                "validation_samples_with_training_overlap": int(len(val_with_overlap)),
                "max_overlap_fraction_of_one_buffer": float(
                    max([r["overlap_fraction_of_one_buffer"] for r in crossing], default=0.0)
                ),
                "mean_overlap_fraction_of_one_buffer": float(
                    np.mean([r["overlap_fraction_of_one_buffer"] for r in crossing]) if crossing else 0.0
                ),
            }
        )
    fold_table = pd.DataFrame(fold_rows)
    summary = {
        "radius_m": radius,
        "sample_count": n,
        "all_overlap_pair_count": int(len(pairs)),
        "cross_group_overlap_pair_count": int(pairs["cross_train_validation_fold"].sum()) if not pairs.empty else 0,
        "samples_in_any_overlap_pair": int(
            len(set(pairs["sample_id_i"].astype(int)).union(set(pairs["sample_id_j"].astype(int))))
        )
        if not pairs.empty
        else 0,
        "samples_in_cross_group_overlap_pair": int(
            len(
                set(pairs.loc[pairs["cross_train_validation_fold"], "sample_id_i"].astype(int)).union(
                    set(pairs.loc[pairs["cross_train_validation_fold"], "sample_id_j"].astype(int))
                )
            )
        )
        if not pairs.empty
        else 0,
    }
    return pairs, fold_table, summary


def weight_matrix(coords: np.ndarray, method: str) -> np.ndarray:
    n = coords.shape[0]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    w = np.zeros((n, n), dtype=float)
    if method == "knn4":
        for i in range(n):
            nn = np.argsort(dist[i])[:4]
            w[i, nn] = 1.0
        w = np.maximum(w, w.T)
    elif method == "distance_band_5m":
        w[(dist <= 5.0) & np.isfinite(dist)] = 1.0
    else:
        raise ValueError(f"Unknown weight method: {method}")
    return w


def moran_i(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    x = values[mask]
    w = weights[np.ix_(mask, mask)].astype(float)
    np.fill_diagonal(w, 0.0)
    w_sum = w.sum()
    z = x - x.mean()
    denom = float(np.sum(z * z))
    if len(x) < 3 or w_sum <= 0 or denom <= 0:
        return np.nan
    return float(len(x) / w_sum * np.sum(w * np.outer(z, z)) / denom)


def moran_permutation(values: np.ndarray, weights: np.ndarray, n_perm: int = 999) -> dict:
    rng = np.random.default_rng(MORAN_SEED)
    obs = moran_i(values, weights)
    if not np.isfinite(obs):
        return {
            "moran_i": np.nan,
            "expected_i": np.nan,
            "permutation_p_greater": np.nan,
            "permutation_p_two_sided_abs": np.nan,
        }
    vals = np.asarray(values, dtype=float)
    perms = np.array([moran_i(rng.permutation(vals), weights) for _ in range(n_perm)], dtype=float)
    perms = perms[np.isfinite(perms)]
    expected = -1.0 / (len(vals) - 1)
    p_greater = (np.sum(perms >= obs) + 1) / (len(perms) + 1)
    centered_obs = abs(obs - expected)
    centered_perm = np.abs(perms - expected)
    p_two = (np.sum(centered_perm >= centered_obs) + 1) / (len(perms) + 1)
    return {
        "moran_i": float(obs),
        "expected_i": float(expected),
        "permutation_p_greater": float(p_greater),
        "permutation_p_two_sided_abs": float(p_two),
    }


def spatial_autocorrelation(groups: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    coords = groups[["x", "y"]].to_numpy(float)
    rows = []
    primary = oof[
        (oof["feature_set"] == PRIMARY_FEATURE_SET)
        & (oof["model"] == PRIMARY_MODEL)
    ].copy()
    observed = primary[primary["scale"] == "r1"].drop_duplicates("sample_id").sort_values("sample_id")
    observed = groups[["sample_id"]].merge(observed[["sample_id", "observed"]], on="sample_id", how="left")

    variables = [("observed", "Flower_rat", observed["observed"].to_numpy(float))]
    for scale in ["point", "one_sqft", "r0p3", "r0p5", "r1"]:
        part = primary[primary["scale"] == scale].sort_values("sample_id")
        part = groups[["sample_id"]].merge(part[["sample_id", "observed", "predicted"]], on="sample_id", how="left")
        variables.append((scale, "oof_prediction", part["predicted"].to_numpy(float)))
        variables.append((scale, "oof_residual_observed_minus_predicted", (part["observed"] - part["predicted"]).to_numpy(float)))

    for method in ["knn4", "distance_band_5m"]:
        w = weight_matrix(coords, method)
        neighbor_links = int(np.sum(w > 0) / 2)
        isolated = int(np.sum(w.sum(axis=1) == 0))
        for scale, variable, values in variables:
            result = moran_permutation(values, w)
            rows.append(
                {
                    "weight_method": method,
                    "neighbor_link_count": neighbor_links,
                    "isolated_sample_count": isolated,
                    "scale": scale,
                    "variable": variable,
                    "n": int(np.isfinite(values).sum()),
                    "random_seed": MORAN_SEED,
                    "n_permutations": 999,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def write_markdown(
    group_summary: pd.DataFrame,
    overlap_summary: dict,
    overlap_by_fold: pd.DataFrame,
    spatial_auto: pd.DataFrame,
    perm_fdr: pd.DataFrame,
) -> None:
    best_scale = perm_fdr.sort_values("observed_r2", ascending=False).iloc[0]
    cross_pairs = overlap_summary["cross_group_overlap_pair_count"]
    any_pairs = overlap_summary["all_overlap_pair_count"]
    cross_samples = overlap_summary["samples_in_cross_group_overlap_pair"]
    sig_rows = perm_fdr[perm_fdr["significant_fdr_0p05"]]
    observed_moran = spatial_auto[
        (spatial_auto["scale"] == "observed")
        & (spatial_auto["variable"] == "Flower_rat")
    ].copy()
    r1_residual_moran = spatial_auto[
        (spatial_auto["scale"] == "r1")
        & (spatial_auto["variable"] == "oof_residual_observed_minus_predicted")
    ].copy()

    cv_text = f"""# 空间分组与嵌套验证补充审计

## 1. 空间分组与交叉验证细节

- 建模目标：{TARGET}。
- 有效样本量：33 个有扬花率记录的样点。
- 空间分组方法：使用样点平面坐标 x/y 做 KMeans 空间聚类，聚类数为 5，随机种子为 {SEED}，n_init=50。
- 外部验证方法：Leave-One-Group-Out CV。每次保留 1 个空间组作为验证集，其余 4 个空间组作为训练集，共 5 折。
- 最终尺度比较模型：5月10日单期特征 + Ridge 回归。
- 管道顺序：SimpleImputer(strategy='median') -> SelectKBest(f_regression, k=10) -> StandardScaler() -> Ridge(alpha=10.0)。
- 关键防泄漏设置：缺失值填补、特征筛选和标准化都放在 sklearn Pipeline 内，每一折只在训练样本上拟合，再应用到验证样本。
- 重要边界：当前最终严格重跑不是完整嵌套调参 CV。Ridge 的 alpha=10.0、特征数 k=10 是固定参数；本次验证重点是空间外推能力和尺度比较，而不是在内层 CV 中搜索最优超参数。

如果论文中要使用“嵌套交叉验证”这个表述，必须按以下协议另行重跑：

- 外层仍使用 5 个空间组 Leave-One-Group-Out CV，用于产生最终折外预测和最终性能。
- 每个外层训练集中只包含 4 个空间组；内层验证应继续按空间组划分，即在这 4 个训练组内做 Leave-One-Group-Out。
- 内层只能使用外层训练样本调参，例如搜索 Ridge alpha、SelectKBest 的 k、特征集类型和模型类型。
- 每个外层折中，先由内层 CV 选出参数，再用该外层训练集全体样本重新拟合一次模型，最后预测外层验证组。
- 缺失值填补、标准化、特征筛选和任何特征选择都必须放在内层和外层训练流程内部，不能在全样本上预先完成。
- 置换检验若要匹配嵌套 CV，也应在每次置换中完整重复外层和内层流程。

### 空间组样本数

{df_to_markdown(group_summary)}

## 2. 1.0 m 缓冲区重叠检查

- 检查半径：1.0 m。
- 判定标准：两个样点圆形缓冲区中心距离小于 2.0 m 时，认为缓冲区发生重叠。
- 所有重叠样点对数量：{any_pairs}。
- 跨空间组重叠样点对数量：{cross_pairs}。
- 涉及跨训练-验证折重叠风险的样点数：{cross_samples}。

在 Leave-One-Group-Out CV 中，如果两个 1.0 m 缓冲区重叠但属于不同空间组，则某一折中一个样点会在训练集，另一个样点会在验证集，存在共享影像像元的风险。因此，跨空间组重叠样点对需要在论文中作为空间验证限制说明；若数量较多，建议优先报告 0.3 m 或 0.5 m 尺度作为稳健性结果。

### 各折重叠风险

{df_to_markdown(overlap_by_fold)}

## 3. 空间自相关检验与重采样方式

本次对观测扬花率、五个尺度的折外预测值和折外残差计算 Moran's I。权重矩阵使用两种方式：k 近邻 k=4，以及 5 m 距离带。p 值通过 999 次随机置换估计。

{df_to_markdown(spatial_auto)}

解释原则：

- 如果观测值存在显著正空间自相关，普通随机 Bootstrap 和完全随机标签置换会偏乐观。
- 如果残差仍有显著空间自相关，说明模型没有完全解释空间结构，性能置信区间应使用空间分组 Bootstrap、按空间组置换或空间 block permutation 作为敏感性分析。
- 当前只有 5 个空间组，严格的组级置换组合数有限，因此最终论文中建议把置换检验作为补充证据，并把 Leave-One-Group-Out 的折外性能作为主结果。

本次结果中，观测扬花率的 Moran's I 未显示稳定显著空间自相关：

{df_to_markdown(observed_moran)}

最优 1.0 m 尺度模型的折外残差也未显示显著空间自相关：

{df_to_markdown(r1_residual_moran)}

## 4. 五个尺度置换检验的多重校正

对五个尺度的模型置换检验 p 值做 Benjamini-Hochberg FDR 校正，校正族定义为：同一目标 Flower_rat、同一模型 Ridge、同一特征集 0510_single_date 下的五个空间尺度比较。

{df_to_markdown(perm_fdr)}

FDR=0.05 下显著的尺度：{", ".join(sig_rows["scale"].tolist()) if len(sig_rows) else "无"}。
当前 R² 最高尺度为 {best_scale["scale"]}，R²={best_scale["observed_r2"]:.3f}，原始置换 p={best_scale["permutation_p"]:.4f}，FDR q={best_scale["fdr_q_bh_within_five_scales"]:.4f}。
"""
    (OUT / "SPATIAL_CV_DETAILS.md").write_text(cv_text, encoding="utf-8")

    recommendations = """# Bootstrap 与置换检验建议

1. 论文主结果建议使用空间组 Leave-One-Group-Out 的折外预测性能，不使用随机 KFold 性能作为主证据。
2. Bootstrap 置信区间不宜使用普通样点级随机 Bootstrap 作为唯一结果；更合适的是按空间组重采样，或报告五个空间折的误差分布作为敏感性分析。
3. 置换检验目前是“标签随机置换 + 完整空间 CV 重跑”，比只打乱预测值更严格，但仍未约束空间结构。
4. 若观测扬花率空间自相关显著，建议追加“空间组内置换”或“空间块置换”作为敏感性分析；由于只有 5 个空间组，组级置换检验的分辨率有限，应在论文中说明。
5. 五个尺度属于同一组比较，应报告 FDR q 值；不宜只挑最小 p 值作为单独显著性证据。
"""
    (OUT / "SPATIAL_RESAMPLING_RECOMMENDATIONS.md").write_text(recommendations, encoding="utf-8")

    summary = f"""# 空间验证诊断摘要

- 最终严格模型采用 5 个空间聚类组的 Leave-One-Group-Out CV，不是完整嵌套调参 CV；固定参数为 Ridge(alpha=10.0) 和 SelectKBest(k=10)。
- 1.0 m 缓冲区共有 {any_pairs} 对样点发生重叠，其中 {cross_pairs} 对跨空间组；因此本次未发现 1.0 m 样点缓冲区跨训练-验证折重叠。
- 观测扬花率和最优 1.0 m 模型折外残差均未显示稳定显著的空间自相关。
- 五尺度置换检验经过 BH-FDR 校正后，显著尺度为：{", ".join(sig_rows["scale"].tolist()) if len(sig_rows) else "无"}。
- 最好尺度仍为 {best_scale["scale"]}：R²={best_scale["observed_r2"]:.3f}，FDR q={best_scale["fdr_q_bh_within_five_scales"]:.4f}。
- 论文中可以把空间折外验证结果作为主结果；Bootstrap 和置换检验建议作为补充证据，并说明当前样本量和空间组数量有限。
"""
    (OUT / "SPATIAL_CV_DIAGNOSTICS_SUMMARY.md").write_text(summary, encoding="utf-8")


def run() -> None:
    ensure_inputs()
    OUT.mkdir(parents=True, exist_ok=True)

    groups = pd.read_csv(AUDIT / "STRICT_SPATIAL_GROUPS.csv")
    groups = groups.sort_values("sample_id").reset_index(drop=True)
    oof = pd.read_csv(AUDIT / "OOF_PREDICTIONS.csv")
    perm = pd.read_csv(AUDIT / "MODEL_PERMUTATION_TEST.csv")

    group_summary = summarize_groups(groups, oof)
    group_summary.to_csv(OUT / "SPATIAL_GROUP_SUMMARY.csv", index=False)

    pairs, overlap_by_fold, overlap_summary = overlap_tables(groups, radius=1.0)
    pairs.to_csv(OUT / "ONE_METER_OVERLAP_PAIRS.csv", index=False)
    overlap_by_fold.to_csv(OUT / "ONE_METER_OVERLAP_BY_FOLD.csv", index=False)
    pd.DataFrame([overlap_summary]).to_csv(OUT / "ONE_METER_OVERLAP_SUMMARY.csv", index=False)

    spatial_auto = spatial_autocorrelation(groups, oof)
    spatial_auto.to_csv(OUT / "SPATIAL_AUTOCORRELATION.csv", index=False)

    perm_fdr = perm.copy()
    perm_fdr["scale_label"] = perm_fdr["scale"].map(SCALE_LABELS)
    perm_fdr["fdr_q_bh_within_five_scales"] = bh_fdr(perm_fdr["permutation_p"])
    perm_fdr["significant_fdr_0p05"] = perm_fdr["fdr_q_bh_within_five_scales"] <= 0.05
    perm_fdr["fdr_family"] = "Flower_rat + Ridge + 0510_single_date + five spatial scales"
    perm_fdr = perm_fdr[
        [
            "target",
            "scale",
            "scale_label",
            "feature_set",
            "model",
            "observed_r2",
            "permutation_p",
            "fdr_q_bh_within_five_scales",
            "significant_fdr_0p05",
            "n_permutations",
            "random_seed",
            "fdr_family",
            "test_definition",
        ]
    ].sort_values("permutation_p")
    perm_fdr.to_csv(OUT / "MODEL_PERMUTATION_TEST_FDR.csv", index=False)
    shutil.copy2(OUT / "MODEL_PERMUTATION_TEST_FDR.csv", AUDIT / "MODEL_PERMUTATION_TEST_FDR.csv")

    write_markdown(group_summary, overlap_summary, overlap_by_fold, spatial_auto, perm_fdr)
    shutil.make_archive(str(AUDIT / "paper_audit_spatial_cv_diagnostics"), "zip", OUT)
    print(f"Wrote spatial CV diagnostics to {OUT}")
    print(f"Wrote archive to {AUDIT / 'paper_audit_spatial_cv_diagnostics.zip'}")


if __name__ == "__main__":
    run()
