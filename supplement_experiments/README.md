# 补充实验输出说明

本目录由 `supplement_experiments.py` 生成，用于补充论文中的实验结果。脚本不重新读取遥感影像，只复用已经完成的点采样、1平方尺采样和多尺度缓冲区分析结果。

## 主要表格

- `target_distribution_summary.csv`：田间指标的数据分布，包括有效样本数、缺失数、均值、标准差、四分位数、最大最小值、0值样本数和正值样本数。用于说明样本量和指标分布。
- `best_correlation_by_scale.csv`：每个指标在每个采样尺度下相关性最强的遥感特征。重点看 `spearman_r`、`spearman_p`、`feature` 和 `dataset_name`。
- `scale_comparison_top_correlations.csv`：每个指标、每个尺度下排名靠前的相关特征。用于补充材料或筛选候选变量。
- `model_comparison_all.csv`：所有回归模型的交叉验证结果。重点看 `r2`、`rmse`、`mae`、`pearson_r`、`spearman_r`。
- `best_model_by_scale.csv`：每个指标、每个尺度下表现最好的回归模型。适合放入正文或PPT。
- `correlation_summary_with_feature_groups.csv`：在原相关性表基础上增加了时间类型、尺度、特征类别和统计量解析，便于解释特征来源。
- `feature_group_summary.csv`：按时间类型和特征类型汇总相关性强度，用于回答“哪类遥感特征更有用”。
- `temporal_feature_summary.csv`：比较 0418、0510、diff、ratio、rel 等时间特征的表现，用于回答“单期影像还是变化特征更有用”。
- `disease_classification_metrics.csv`：将病害指标转为“有病/无病”后的分类实验。重点看 `balanced_accuracy`、`roc_auc` 和 `f1`。
- `paper_experiment_summary.md`：面向论文写作的简要结果摘要。

## 图件

图件保存在 `figures/`：

- `target_distribution.png`：田间指标分布图。
- `best_correlation_by_scale.png`：不同采样尺度下最强相关性对比。
- `best_model_r2_by_scale.png`：不同采样尺度下最佳回归模型 R2 对比。
- `temporal_feature_summary.png`：单期与变化特征的响应强度对比。
- `disease_classification_metrics.png`：病害有无分类结果对比。

## 建议写法

正文建议重点使用 `1平方尺` 和 `点采样` 两套结果，因为它们更贴近田间采样面积。`0.3/0.5/1.0m` 多尺度结果可作为尺度效应补充：它能说明扩大邻域后相关性会增强，但不宜直接当作1平方尺采样的正式反演结果。
