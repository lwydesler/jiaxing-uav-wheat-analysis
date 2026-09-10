# 数据与建模审计报告

项目：小麦扬花率与无人机多光谱影像关系分析  
论文主线：无人机多光谱监测小麦扬花率的尺度效应  
审计目标：判断数据、空间尺度、相关性、建模流程和预测制图结果是否足以支撑中文农业期刊论文。

## 一页式执行摘要

1. 当前最高相关系数是否可信：**部分可信**。1 ft²尺度下 `Flower_rat` 最强相关特征为 `0510_r0p172_ND_B2_B3_median`，Spearman r = `-0.659`，FDR q = `0.0116`。但存在大量特征搜索，应同时参考全局FDR、Bootstrap稳定性和最大统计量置换检验。
2. 当前最高模型R2是否可信：**原有模型R2偏乐观风险较高**。原流程在全数据相关性筛选后再交叉验证，存在特征选择泄漏。严格空间CV复核中最佳结果为 `r1` / `0510_single_date` / `Ridge`，R2 = `0.291`。
3. 是否发现数据泄漏：**发现原建模流程存在特征选择泄漏风险**。证据见 `rs_field_analysis.py` 第 598-611 行和第 825-826 行。
4. 最可靠的空间尺度：**1 ft²尺度最适合论文主线**，因为它与田间样方面积一致，且当前空间审计未发现1 ft²样点缓冲区重叠。
5. 当前结果适合“定量估算”还是“趋势监测”：**适合趋势监测或探索性响应分析，不适合作为高精度定量估算**。
6. 当前能否开始写论文结果部分：**可以开始写方法和探索性结果框架，但投稿前必须重跑严格验证结果**。
7. 投稿前必须完成三项工作：**补充影像辐射定标说明；使用训练折内特征筛选和空间/Group CV确认扬花率模型；对最强相关和最佳模型补足高重复Bootstrap和置换检验。**

## 数据和代码概况

数据流关系为：

原始田间数据 `wang/data.shp` → 清洗后的田间数据 `field_samples_clean.csv` → 样点遥感特征 `sample_rs_features.csv` → 特征目标合并表 `all_features_targets.csv` → 相关性结果 `correlation_summary.csv` → 模型结果 `model_metrics.csv` → 补充汇总结果 → 预测制图结果。

各结果来源：

- 点采样结果：`analysis_point/`，由 `rs_field_analysis.py --radii point` 生成。
- 1 ft²结果：`analysis_1sqft_circle/`，由 `rs_field_analysis.py --radii 0.172` 生成。
- 0.3、0.5、1.0 m多尺度结果：`analysis/`，由 `rs_field_analysis.py --radii 0.3,0.5,1.0` 生成。
- 补充汇总结果：`supplement_experiments/`，由 `supplement_experiments.py` 对基础表二次汇总生成。
- 扬花率预测图：`prediction_maps/Flower_rat_RandomForest_1sqft_prediction.*`，由 `predict_flower_rat_map.py` 生成。

项目关键文件清单见 `AUDIT_PROJECT_INVENTORY.csv`。该清单仅记录路径和文件大小，不复制原始影像。

## 目标变量审计

`Flower_num` 和 `Flower_rat` 的分布如下：

| target | valid_n | missing_n | min | p25 | median | mean | p75 | max | std | unique_n | unique_values | duplicate_target_value_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flower_num | 33 | 37 | 1 | 16 | 23 | 20.73 | 28 | 30 | 8.98 | 18 | 1;4;6;10;11;14;15;16;17;18;21;23;25;26;27;28;29;30 | 15 |
| Flower_rat | 33 | 37 | 3.33 | 53.33 | 76.67 | 69.09 | 93.33 | 100 | 29.93 | 18 | 3.33;13.33;20;33.33;36.67;46.67;50;53.33;56.67;60;70;76.67;83.33;86.67;90;93.33;96.67;100 | 15 |

关系审计见 `AUDIT_TARGET_RELATION.csv`。审计重点是判断 `Flower_rat = Flower_num / 30 × 100` 是否成立。如果成立，则二者本质上是同一调查目标的两个表达，论文主线应保留 `Flower_rat`，不要把 `Flower_num` 和 `Flower_rat` 当作两个独立发现重复汇报。对于统计建模，也可以考虑二项、准二项或Beta-binomial思想，但当前脚本未强行引入新依赖。

## 空间与尺度审计

影像像元面积由GDAL GeoTransform计算。各尺度理论面积、像元数、有效像元数和重叠风险如下：

| scale | scale_label | radius_m | area_m2 | theoretical_pixel_count | actual_valid_pixel_median | actual_valid_pixel_min | actual_valid_pixel_max | overlapping_sample_fraction | mean_overlap_fraction | max_overlap_fraction | mean_overlap_neighbor_count | train_test_overlap_risk | boundary_crossing_risk | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point | 点采样 | 0 | 0.00331 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | none | 无法核实：缺少试验小区边界或样方多边形 | Point sampling is a single pixel. |
| one_sqft | 1 ft² | 0.172 | 0.09294 | 28.08 | 29 | 26 | 31 | 0 | 0 | 0 | 0 | low | 无法核实：缺少试验小区边界或样方多边形 | Circle buffer radius in meters; overlap evaluated among Flower_rat-valid samples. |
| r0p3 | 0.3 m | 0.3 | 0.2827 | 85.43 | 85 | 83 | 89 | 0 | 0 | 0 | 0 | low | 无法核实：缺少试验小区边界或样方多边形 | Circle buffer radius in meters; overlap evaluated among Flower_rat-valid samples. |
| r0p5 | 0.5 m | 0.5 | 0.7854 | 237.3 | 238 | 234 | 242 | 0 | 0 | 0 | 0 | low | 无法核实：缺少试验小区边界或样方多边形 | Circle buffer radius in meters; overlap evaluated among Flower_rat-valid samples. |
| r1 | 1.0 m | 1 | 3.142 | 949.2 | 948 | 943 | 955 | 0.06061 | 0.008768 | 0.1447 | 0.06061 | moderate | 无法核实：缺少试验小区边界或样方多边形 | Circle buffer radius in meters; overlap evaluated among Flower_rat-valid samples. |

关键判断：

- 1 ft²按 0.0929 m²理解时，等面积圆半径约为 0.172 m，理论上约对应28个像元。
- 点采样和1 ft²尺度未发现Flower_rat有效样点的缓冲区重叠风险。
- 0.5 m和1.0 m尺度出现缓冲区重叠，尤其1.0 m存在较高训练-测试共享影像像元风险。
- 当前缺少试验小区边界，因此无法核实缓冲区是否跨越小区边界。
- 当前样点表未发现小区、品种、处理或重复区字段，因此无法直接执行设计分组CV；本审计采用空间聚类CV作为严格复核替代方案。

## 影像定标和时间特征审计

影像元数据摘要见 `AUDIT_IMAGE_METADATA.csv`。当前能够确认两期影像的尺寸、波段数、坐标系和像元分辨率，但无法从现有材料确认传感器型号、飞行高度、拍摄时间、太阳高度、曝光参数、校准板信息、辐射定标或反射率转换流程。

因此：

- 0418与0510两期数值是否具有严格辐射可比性：**无法核实**。
- `diff`、`ratio` 和 `rel` 是否能解释为真实作物变化：**只能作为探索性跨期特征**。
- 论文中更稳妥的主结果应优先使用 0510 单期归一化指数和1 ft²尺度结果；跨期特征可作为敏感性分析。

## 特征质量审计

特征质量汇总见 `AUDIT_FEATURE_QUALITY.csv`：

| scale | total_features | constant_features | duplicate_columns | valid_count_feature_count |
| --- | --- | --- | --- | --- |
| point | 378 | 24 | 241 | 12 |
| one_sqft | 378 | 0 | 11 | 12 |
| r0p3 | 378 | 0 | 11 | 12 |
| r0p5 | 378 | 0 | 11 | 12 |
| r1 | 378 | 0 | 11 | 12 |

审计发现：

- 各尺度特征数量较多，存在多重比较问题。
- `valid_count` 特征存在于特征表中，应避免被解释为作物光谱响应变量。
- 当前特征名称能够解析出日期、尺度、统计量和特征类型。
- 是否每个特征公式与名称完全一致，需要结合 `rs_field_analysis.py` 中特征公式继续人工核对。

## 相关性及多重检验结果

本审计以 `Flower_rat` 为主目标，重新计算了各尺度所有候选特征的Pearson和Spearman相关性，并执行：

- 每个尺度内部的Benjamini-Hochberg FDR校正；
- 跨全部尺度和全部特征的全局FDR敏感性分析；
- 对每个尺度前30个特征进行500次Bootstrap置信区间和Top-k稳定性统计；
- 对每个尺度执行300次最大统计量置换检验，置换时重复“从所有候选特征中选择最高相关特征”的完整过程。

说明：附件建议投稿前采用1000次置换和2000次Bootstrap。本次为可在当前环境完成的快速审计版，已经能够判断多重搜索和稳定性风险；投稿定稿前建议用更高重复次数重跑。

各尺度前若干特征见 `AUDIT_TOP_FEATURES.csv`。其中1 ft²尺度的主结果为：

| target | scale | feature | feature_type | date_type | statistic | n | pearson_r | pearson_p | spearman_r | spearman_p | fdr_q_within_scale | fdr_q_global | bootstrap_ci_lower | bootstrap_ci_upper | bootstrap_top1_frequency | bootstrap_top5_frequency | bootstrap_top10_frequency | max_stat_permutation_p | stability_status | abs_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flower_rat | one_sqft | 0510_r0p172_ND_B2_B3_median | pairwise_nd | 0510 | median | 33 | -0.5264 | 0.001649 | -0.6587 | 3.081e-05 | 0.01165 | 0.002875 | -0.8245 | -0.4476 | 0.448 | 0.666 | 0.8 | 0.01993 | stable_top10 | 0.6587 |
| Flower_rat | one_sqft | diff_r0p172_DVI_mean | predefined_index | diff | mean | 33 | -0.547 | 0.0009864 | -0.5959 | 0.0002533 | 0.01234 | 0.007093 | -0.7773 | -0.3682 | 0.07 | 0.306 | 0.454 | 0.01993 | moderate | 0.5959 |
| Flower_rat | one_sqft | ratio_r0p172_DVI_median | predefined_index | ratio | median | 33 | -0.5258 | 0.001675 | -0.5932 | 0.0002745 | 0.01234 | 0.007115 | -0.7679 | -0.3571 | 0.014 | 0.224 | 0.428 | 0.01993 | unstable_or_not_bootstrapped | 0.5932 |
| Flower_rat | one_sqft | rel_r0p172_DVI_median | predefined_index | rel | median | 33 | -0.5258 | 0.001675 | -0.5932 | 0.0002745 | 0.01234 | 0.007115 | -0.7679 | -0.3571 | 0.02 | 0.238 | 0.422 | 0.01993 | unstable_or_not_bootstrapped | 0.5932 |
| Flower_rat | one_sqft | 0510_r0p172_ND_B1_B3_median | pairwise_nd | 0510 | median | 33 | -0.557 | 0.0007607 | -0.5838 | 0.0003619 | 0.01234 | 0.007944 | -0.7993 | -0.2728 | 0.048 | 0.262 | 0.366 | 0.01993 | unstable_or_not_bootstrapped | 0.5838 |
| Flower_rat | one_sqft | 0510_r0p172_ND_B2_B3_mean | pairwise_nd | 0510 | mean | 33 | -0.5151 | 0.002157 | -0.574 | 0.0004775 | 0.01234 | 0.007968 | -0.775 | -0.2828 | 0.006 | 0.18 | 0.304 | 0.01993 | unstable_or_not_bootstrapped | 0.574 |
| Flower_rat | one_sqft | 0510_r0p172_EVI_median | predefined_index | 0510 | median | 33 | -0.5296 | 0.001528 | -0.5714 | 0.0005147 | 0.01234 | 0.007968 | -0.7837 | -0.2458 | 0.006 | 0.09 | 0.24 | 0.01993 | unstable_or_not_bootstrapped | 0.5714 |
| Flower_rat | one_sqft | rel_r0p172_RENDVI1_median | predefined_index | rel | median | 33 | -0.4671 | 0.006134 | -0.564 | 0.0006305 | 0.01234 | 0.007968 | -0.7992 | -0.2221 | 0.022 | 0.172 | 0.282 | 0.01993 | unstable_or_not_bootstrapped | 0.564 |
| Flower_rat | one_sqft | ratio_r0p172_RENDVI1_median | predefined_index | ratio | median | 33 | -0.4671 | 0.006134 | -0.564 | 0.0006305 | 0.01234 | 0.007968 | -0.7992 | -0.2221 | 0.02 | 0.16 | 0.28 | 0.01993 | unstable_or_not_bootstrapped | 0.564 |
| Flower_rat | one_sqft | ratio_r0p172_DVI_mean | predefined_index | ratio | mean | 33 | -0.5393 | 0.001201 | -0.561 | 0.0006841 | 0.01234 | 0.008327 | -0.7685 | -0.2935 | 0.014 | 0.108 | 0.244 | 0.01993 | unstable_or_not_bootstrapped | 0.561 |

解释时必须区分“单个特征显著”和“从大量特征中筛出的最大相关显著”。如果最大统计量置换检验不显著，则只能说候选特征中存在较强响应线索，不能把最高相关特征当作稳健发现。

## 原建模流程的数据泄漏检查

代码审计清单见 `AUDIT_CODE_CHECKLIST.csv`：

| audit_item | status | evidence_file | evidence_lines | impact | required_action |
| --- | --- | --- | --- | --- | --- |
| 使用交叉验证方式 | 是 | rs_field_analysis.py | 642,650 | LOOCV for n<=40; random 5-fold otherwise. | Flower_rat currently uses LOOCV in original workflow. |
| 报告是否为折外预测 | 是 | rs_field_analysis.py | 650,682-694 | cross_val_predict produces out-of-fold predictions. | Keep reporting as CV, not training fit. |
| 缺失值填补在训练折内部 | 是 | rs_field_analysis.py | 546-590,650 | SimpleImputer is inside Pipeline passed to cross_val_predict. | No action for imputation. |
| 标准化在训练折内部 | 是 | rs_field_analysis.py | 546-590,650 | StandardScaler is inside Pipeline for linear/PLS models. | No action for scaling. |
| 特征筛选在训练折内部 | 否 | rs_field_analysis.py | 598-611,825-826 | Features are selected from full-data correlation table before CV. | Rerun with feature selection inside Pipeline/nested CV. |
| 超参数选择在训练折内部 | 部分满足 | rs_field_analysis.py | 553-590 | Most parameters are fixed; no inner tuning is used. | If tuning is reported, use nested CV. |
| 是否先用全数据选择最高相关特征再CV | 是 | rs_field_analysis.py | 598-611,825-826 | select_features_for_target uses corr_top computed on full dataset. | Treat original model R2 as optimistic. |
| 是否使用相同数据选模型并报告最终模型 | 是 | supplement_experiments.py | 327-342 | best_model_by_scale selects best model from same CV result table. | Use an external validation or call it model-selection result. |
| 不同尺度验证划分一致 | 部分满足 | rs_field_analysis.py | 642 | Same deterministic LOOCV or random_state 42; multi-scale table combines scales. | Strict scale audit uses common spatial folds. |
| 最终R2是否由合并折外预测计算 | 是 | rs_field_analysis.py | 650,682-694 | Metrics are computed from cross_val_predict outputs. | Mention leakage risk separately. |
| 是否报告均值预测基线 | 否 | rs_field_analysis.py | 543-594 | Original model list lacks DummyRegressor. | Add DummyRegressor baseline in audit. |
| 预测图模型是否用全数据重训 | 是 | predict_flower_rat_map.py | 104-122,367-383 | Mapping model is trained on all available target samples. | Map should be described as trend map. |
| 预测图是否标注探索性 | 是 | predict_flower_rat_map.py | 4-10,404 | Script notes limited skill and exploratory trend map. | Keep this limitation in paper. |
| 是否有小区/品种/处理/重复区分组字段 | 无法核实 | field_samples_clean.csv/data.shp | columns inspected by audit script | No explicit grouping fields found in current analysis CSV. | Request experimental design metadata. |
| 是否检查缓冲区重叠跨折 | 原流程未检查 | rs_field_analysis.py | 642 | Original random/LOOCV split does not account for spatial overlap. | Use spatial CV or Group CV. |

最重要的问题是：原建模流程先在全数据上计算相关性并筛选Top特征，再将这些特征送入交叉验证模型。因此，缺失值填补和标准化虽然在Pipeline内部完成，但特征筛选不在训练折内部，导致模型性能存在偏乐观风险。

## 严格交叉验证复核

本审计针对 `Flower_rat` 进行了严格复核。由于当前缺少小区、品种、处理和重复区字段，无法进行真正的Group CV；审计采用基于样点坐标的5个空间聚类作为外层验证分组。缺失值填补、标准化和特征筛选均放入训练折内部；为保证本次审计可完成，模型使用固定超参数，未进行内层调参。模型性能由外层折外预测计算。

严格复核结果见 `AUDIT_KEY_RESULTS.csv`。最佳若干结果如下：

| target | scale | feature_set | model | validation_scheme | n | n_features | r2 | rmse | mae | pearson_r | spearman_r | slope | intercept | ci_lower | ci_upper | permutation_p | leakage_risk | result_status | notes | r2_num |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flower_rat | r1 | 0510_single_date | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 102 | 0.2907 | 24.83 | 19.95 | 0.5481 | 0.5615 | 0.8603 | 8.572 | -0.2727 | 0.5362 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.2907 |
| Flower_rat | r1 | change_only | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 174 | 0.2677 | 25.23 | 20.68 | 0.5287 | 0.5727 | 0.8872 | 5.562 | -0.2806 | 0.5014 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.2677 |
| Flower_rat | r0p3 | 0510_single_date | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 102 | 0.1774 | 26.73 | 23.15 | 0.4441 | 0.4471 | 0.7644 | 15.54 | -0.363 | 0.412 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.1774 |
| Flower_rat | r1 | all_features | SingleFeatureLinear | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 378 | 0.1273 | 27.54 | 22.3 | 0.4736 | 0.4883 | 0.7573 | 10.7 | -0.4248 | 0.4124 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.1273 |
| Flower_rat | r0p5 | 0510_single_date | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 102 | 0.112 | 27.78 | 23.33 | 0.4154 | 0.429 | 0.6384 | 23.62 | -0.5859 | 0.3988 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.112 |
| Flower_rat | r0p3 | change_only | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 174 | 0.1001 | 27.96 | 23.26 | 0.3847 | 0.39 | 0.6933 | 18.38 | -0.3937 | 0.3462 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.1001 |
| Flower_rat | point | 0510_single_date | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 102 | 0.08201 | 28.24 | 23.91 | 0.3284 | 0.3798 | 0.672 | 22.91 | -0.4904 | 0.3313 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.08201 |
| Flower_rat | one_sqft | change_only | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 174 | 0.06187 | 28.55 | 23.21 | 0.3348 | 0.3957 | 0.635 | 23.07 | -0.5109 | 0.3209 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.06187 |
| Flower_rat | one_sqft | all_features | SingleFeatureLinear | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 378 | 0.06178 | 28.55 | 23.89 | 0.3295 | 0.3655 | 0.6222 | 24.6 | -0.5038 | 0.2921 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.06178 |
| Flower_rat | r0p5 | change_only | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 174 | 0.05106 | 28.72 | 23.65 | 0.384 | 0.3865 | 0.5827 | 26.36 | -0.5369 | 0.3435 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.05106 |
| Flower_rat | r0p5 | all_features | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 378 | 0.03761 | 28.92 | 24.04 | 0.3743 | 0.3645 | 0.5658 | 27.64 | -0.5787 | 0.319 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.03761 |
| Flower_rat | r0p3 | all_features | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 378 | 0.03161 | 29.01 | 24.06 | 0.3277 | 0.3215 | 0.5832 | 26.31 | -0.4718 | 0.2718 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.03161 |
| Flower_rat | one_sqft | 0510_single_date | Ridge | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 102 | 0.01193 | 29.3 | 24.3 | 0.3325 | 0.3702 | 0.5223 | 31.79 | -0.784 | 0.3537 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.01193 |
| Flower_rat | one_sqft | all_features | PLS | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 378 | 0.01056 | 29.32 | 23.98 | 0.325 | 0.3798 | 0.5291 | 30.87 | -0.7702 | 0.3351 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 可信但需限制表述 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | 0.01056 |
| Flower_rat | one_sqft | all_features | RandomForest | 5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold | 33 | 378 | -0.003483 | 29.53 | 23.75 | 0.2987 | 0.2634 | 0.5362 | 29.5 | -0.6462 | 0.2671 |  | low_for_strict_refit; original_workflow_has_feature_selection_leakage | 当前数据不支持定量预测 | model_permutation_not_run_runtime_limited; rerun before submission; chosen_params={fixed_params}/{fixed_params}/{fixed_params}/{fixed_params}/{fixed_params} | -0.003483 |

需要注意：

- 空间CV通常比随机LOOCV更严格，更接近空间泛化能力。
- 如果严格空间CV结果低于原结果，说明原结果可能受空间邻近、全数据特征筛选或模型选择影响。
- 当前仍建议把模型用途限制在扬花率相对高低和空间趋势监测。
- 本次固定超参数复核主要用于判断原结果是否稳健；投稿前如需报告调参模型，应使用嵌套CV。
- 本次模型级目标变量置换检验因嵌套空间CV计算量较大未完成，已经在 `AUDIT_KEY_RESULTS.csv` 的 `notes` 中标记，投稿前应单独补做。

## 尺度效应复核

尺度效应必须分开报告点采样、1 ft²、0.3 m、0.5 m和1.0 m，不能将0.3、0.5和1.0 m合并成一个“多尺度最优结果”。本审计在 `AUDIT_KEY_RESULTS.csv` 中提供了固定Ridge模型和各模型比较结果，在 `AUDIT_SPATIAL_SCALE.csv` 中提供了各尺度有效像元数和重叠风险。

若1.0 m尺度性能更高，应谨慎解释为“空间平滑增强了信号”，不能直接解释为采样尺度最优，因为1.0 m半径远大于1 ft²地面采样面积，并且有较明显重叠风险。

## 预测制图审计

现有扬花率预测图由 `predict_flower_rat_map.py` 生成，使用 `analysis_1sqft_circle/model_metrics.csv` 中的最佳RandomForest模型，并在全部有效样本上重新训练后应用到整幅影像。脚本自身已注明“exploratory flowering-rate trend map; model skill is limited”。

审计判断：**当前预测图只能作为相对空间趋势图，不适合作为定量扬花率预测图。**

原因：

- 原始最佳模型来自存在特征选择泄漏风险的流程。
- 1 ft²尺度原始模型R2较低。
- 当前没有独立外部验证。
- 未提供预测不确定性。
- 未核实预测区域是否严格限制在小麦区域内。

建议论文图名使用“扬花进程空间趋势图”或“扬花敏感指数空间分布图”，避免写成“扬花率定量预测图”。

## 可信结论与不可信结论

结论分级见 `AUDIT_CONCLUSION_GRADING.csv`：

| conclusion | status | evidence | risk | recommended_wording |
| --- | --- | --- | --- | --- |
| 无人机多光谱能够反映小麦扬花率变化 | 基本可信，但需限制表述 | 1 ft² 最强 Spearman=-0.659, FDR q=0.0116 | 特征搜索较多，需FDR、Bootstrap和置换检验共同支撑 | 可表述为存在显著遥感响应关系。 |
| 5月10日比4月18日更适合监测扬花率 | 基本可信，但需限制表述 | 1 ft² 最优特征来自0510单期ND_B2_B3；严格模型仍需看仅0510特征集 | 两期影像定标无法核实 | 可表述为5月10日特征在当前样本中更敏感。 |
| Green-Red组合指数对扬花率最敏感 | 可信，可进入正文 | 1 ft² 最强特征 0510_r0p172_ND_B2_B3_median | 仍需报告FDR和稳定性，不应只报单一最优值 | 可表述为Green-Red归一化差异指数表现突出。 |
| 1 ft²尺度存在显著相关性 | 可信，可进入正文 | Spearman=-0.659, p=3.08e-05 | 模型R2较低，相关不等于高精度预测 | 可表述为1 ft²尺度相关性显著。 |
| 1 m尺度预测性能最好 | 需要重跑后决定 | 原结果1m相关强，但空间重叠风险高；严格空间CV结果见AUDIT_KEY_RESULTS | 1m有明显缓冲区重叠与平滑效应 | 可作为尺度效应敏感性，不作为最终最优尺度强结论。 |
| 扩大空间邻域能够提高预测稳定性 | 基本可信，但需限制表述 | 原相关性随尺度扩大增强；1m重叠样点比例高 | 可能来自空间平滑或邻域共享像元 | 可表述为扩大邻域增强相关性，但需警惕空间非独立。 |
| 当前模型能够定量预测扬花率 | 当前数据不支持 | 严格空间CV最高R2=0.291 | 样本仅33个，原模型存在特征选择泄漏风险 | 不建议使用定量估算表述。 |
| 当前模型适合扬花率高低趋势监测 | 基本可信，但需限制表述 | 相关性较强，预测图脚本已标注exploratory trend | 趋势监测仍需独立样点验证 | 可表述为空间趋势或相对高低监测。 |
| 当前预测图能够表示扬花率空间分布 | 基本可信，但需限制表述 | 预测图由1 ft² RF模型全样本训练生成 | 模型性能弱，无不确定性，未核实小麦掩膜 | 可作为趋势图，不宜作为精确扬花率图。 |
| 当前研究具有投稿中文农业期刊基础 | 基本可信，但需限制表述 | 数据、尺度效应、相关性和审计流程已形成 | 投稿前需补定标说明、严格验证和论文图表 | 可按尺度效应和探索性监测定位。 |

## 必须补做的实验

1. 使用训练折内特征筛选和空间CV或Group CV重跑 `Flower_rat` 模型。
2. 对最优相关特征进行FDR、Bootstrap和最大统计量置换检验，并在正文中报告校正后结果。
3. 对严格模型流程补做目标变量置换检验。
4. 补充影像辐射定标、飞行参数、调查日期、品种处理和重复区信息。
5. 将0.3、0.5和1.0 m尺度拆开汇报。
6. 对预测图增加外推范围检查、小麦区域掩膜和不确定性说明。

## 投稿可行性判断

当前数据具有投稿中文农业领域期刊的基础，但论文定位应为“尺度效应与遥感响应分析”或“扬花率空间趋势监测”，不宜定位为“高精度定量估算模型”。在完成严格验证、定标说明和尺度拆分汇报后，论文故事是清楚的：5月10日多光谱影像中的Green-Red相关指数对扬花率具有较强响应，1 ft²尺度与田间调查面积匹配，扩大邻域可能增强相关性但也提高空间平滑和重叠风险。
