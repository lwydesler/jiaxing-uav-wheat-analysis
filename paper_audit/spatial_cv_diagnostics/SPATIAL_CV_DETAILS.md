# 空间分组与嵌套验证补充审计

## 1. 空间分组与交叉验证细节

- 建模目标：Flower_rat。
- 有效样本量：33 个有扬花率记录的样点。
- 空间分组方法：使用样点平面坐标 x/y 做 KMeans 空间聚类，聚类数为 5，随机种子为 20260805，n_init=50。
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

| spatial_group | sample_count | x_min | x_max | y_min | y_max | flower_rat_mean | flower_rat_std | flower_rat_min | flower_rat_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 302397 | 302398 | 3.39972e+06 | 3.39973e+06 | 91.665 | 11.7875 | 83.33 | 100 |
| 1 | 10 | 302413 | 302424 | 3.39975e+06 | 3.39977e+06 | 62.334 | 31.8211 | 3.33 | 100 |
| 2 | 7 | 302403 | 302419 | 3.39974e+06 | 3.39975e+06 | 81.4286 | 25.7374 | 36.67 | 100 |
| 3 | 8 | 302411 | 302425 | 3.39971e+06 | 3.39972e+06 | 58.75 | 28.3942 | 13.33 | 93.33 |
| 4 | 6 | 302416 | 302425 | 3.39973e+06 | 3.39974e+06 | 72.2233 | 35.634 | 3.33 | 100 |

## 2. 1.0 m 缓冲区重叠检查

- 检查半径：1.0 m。
- 判定标准：两个样点圆形缓冲区中心距离小于 2.0 m 时，认为缓冲区发生重叠。
- 所有重叠样点对数量：1。
- 跨空间组重叠样点对数量：0。
- 涉及跨训练-验证折重叠风险的样点数：0。

在 Leave-One-Group-Out CV 中，如果两个 1.0 m 缓冲区重叠但属于不同空间组，则某一折中一个样点会在训练集，另一个样点会在验证集，存在共享影像像元的风险。因此，跨空间组重叠样点对需要在论文中作为空间验证限制说明；若数量较多，建议优先报告 0.3 m 或 0.5 m 尺度作为稳健性结果。

### 各折重叠风险

| test_group | validation_sample_count | training_sample_count | train_validation_overlap_pair_count | validation_samples_with_training_overlap | max_overlap_fraction_of_one_buffer | mean_overlap_fraction_of_one_buffer |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 31 | 0 | 0 | 0 | 0 |
| 1 | 10 | 23 | 0 | 0 | 0 | 0 |
| 2 | 7 | 26 | 0 | 0 | 0 | 0 |
| 3 | 8 | 25 | 0 | 0 | 0 | 0 |
| 4 | 6 | 27 | 0 | 0 | 0 | 0 |

## 3. 空间自相关检验与重采样方式

本次对观测扬花率、五个尺度的折外预测值和折外残差计算 Moran's I。权重矩阵使用两种方式：k 近邻 k=4，以及 5 m 距离带。p 值通过 999 次随机置换估计。

| weight_method | neighbor_link_count | isolated_sample_count | scale | variable | n | random_seed | n_permutations | moran_i | expected_i | permutation_p_greater | permutation_p_two_sided_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn4 | 80 | 0 | observed | Flower_rat | 33 | 20260907 | 999 | 0.0717839 | -0.03125 | 0.152 | 0.319 |
| knn4 | 80 | 0 | point | oof_prediction | 33 | 20260907 | 999 | 0.110064 | -0.03125 | 0.09 | 0.176 |
| knn4 | 80 | 0 | point | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | 0.106035 | -0.03125 | 0.086 | 0.161 |
| knn4 | 80 | 0 | one_sqft | oof_prediction | 33 | 20260907 | 999 | 0.170065 | -0.03125 | 0.037 | 0.053 |
| knn4 | 80 | 0 | one_sqft | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | 0.14107 | -0.03125 | 0.05 | 0.078 |
| knn4 | 80 | 0 | r0p3 | oof_prediction | 33 | 20260907 | 999 | 0.0934134 | -0.03125 | 0.126 | 0.229 |
| knn4 | 80 | 0 | r0p3 | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | 0.00765024 | -0.03125 | 0.325 | 0.713 |
| knn4 | 80 | 0 | r0p5 | oof_prediction | 33 | 20260907 | 999 | 0.10091 | -0.03125 | 0.114 | 0.199 |
| knn4 | 80 | 0 | r0p5 | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | 0.075299 | -0.03125 | 0.163 | 0.315 |
| knn4 | 80 | 0 | r1 | oof_prediction | 33 | 20260907 | 999 | 0.0840221 | -0.03125 | 0.147 | 0.26 |
| knn4 | 80 | 0 | r1 | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | 0.0749266 | -0.03125 | 0.151 | 0.316 |
| distance_band_5m | 25 | 9 | observed | Flower_rat | 33 | 20260907 | 999 | -0.0349674 | -0.03125 | 0.476 | 0.978 |
| distance_band_5m | 25 | 9 | point | oof_prediction | 33 | 20260907 | 999 | 0.0335132 | -0.03125 | 0.324 | 0.709 |
| distance_band_5m | 25 | 9 | point | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | 0.00817327 | -0.03125 | 0.373 | 0.842 |
| distance_band_5m | 25 | 9 | one_sqft | oof_prediction | 33 | 20260907 | 999 | 0.187942 | -0.03125 | 0.124 | 0.252 |
| distance_band_5m | 25 | 9 | one_sqft | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | 0.021123 | -0.03125 | 0.361 | 0.792 |
| distance_band_5m | 25 | 9 | r0p3 | oof_prediction | 33 | 20260907 | 999 | -0.191757 | -0.03125 | 0.824 | 0.361 |
| distance_band_5m | 25 | 9 | r0p3 | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | -0.0871432 | -0.03125 | 0.581 | 0.767 |
| distance_band_5m | 25 | 9 | r0p5 | oof_prediction | 33 | 20260907 | 999 | -0.137291 | -0.03125 | 0.716 | 0.568 |
| distance_band_5m | 25 | 9 | r0p5 | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | -0.0187965 | -0.03125 | 0.443 | 0.947 |
| distance_band_5m | 25 | 9 | r1 | oof_prediction | 33 | 20260907 | 999 | -0.0708535 | -0.03125 | 0.588 | 0.83 |
| distance_band_5m | 25 | 9 | r1 | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | -0.0608811 | -0.03125 | 0.534 | 0.865 |

解释原则：

- 如果观测值存在显著正空间自相关，普通随机 Bootstrap 和完全随机标签置换会偏乐观。
- 如果残差仍有显著空间自相关，说明模型没有完全解释空间结构，性能置信区间应使用空间分组 Bootstrap、按空间组置换或空间 block permutation 作为敏感性分析。
- 当前只有 5 个空间组，严格的组级置换组合数有限，因此最终论文中建议把置换检验作为补充证据，并把 Leave-One-Group-Out 的折外性能作为主结果。

本次结果中，观测扬花率的 Moran's I 未显示稳定显著空间自相关：

| weight_method | neighbor_link_count | isolated_sample_count | scale | variable | n | random_seed | n_permutations | moran_i | expected_i | permutation_p_greater | permutation_p_two_sided_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn4 | 80 | 0 | observed | Flower_rat | 33 | 20260907 | 999 | 0.0717839 | -0.03125 | 0.152 | 0.319 |
| distance_band_5m | 25 | 9 | observed | Flower_rat | 33 | 20260907 | 999 | -0.0349674 | -0.03125 | 0.476 | 0.978 |

最优 1.0 m 尺度模型的折外残差也未显示显著空间自相关：

| weight_method | neighbor_link_count | isolated_sample_count | scale | variable | n | random_seed | n_permutations | moran_i | expected_i | permutation_p_greater | permutation_p_two_sided_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn4 | 80 | 0 | r1 | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | 0.0749266 | -0.03125 | 0.151 | 0.316 |
| distance_band_5m | 25 | 9 | r1 | oof_residual_observed_minus_predicted | 33 | 20260907 | 999 | -0.0608811 | -0.03125 | 0.534 | 0.865 |

## 4. 五个尺度置换检验的多重校正

对五个尺度的模型置换检验 p 值做 Benjamini-Hochberg FDR 校正，校正族定义为：同一目标 Flower_rat、同一模型 Ridge、同一特征集 0510_single_date 下的五个空间尺度比较。

| target | scale | scale_label | feature_set | model | observed_r2 | permutation_p | fdr_q_bh_within_five_scales | significant_fdr_0p05 | n_permutations | random_seed | fdr_family | test_definition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flower_rat | r1 | 1.0 m | 0510_single_date | Ridge | 0.290657 | 0.00497512 | 0.0248756 | True | 200 | 20260828 | Flower_rat + Ridge + 0510_single_date + five spatial scales | Permuted Flower_rat labels; each permutation reruns full outer spatial CV pipeline. |
| Flower_rat | r0p3 | 0.3 m | 0510_single_date | Ridge | 0.177448 | 0.00995025 | 0.0248756 | True | 200 | 20260828 | Flower_rat + Ridge + 0510_single_date + five spatial scales | Permuted Flower_rat labels; each permutation reruns full outer spatial CV pipeline. |
| Flower_rat | r0p5 | 0.5 m | 0510_single_date | Ridge | 0.11198 | 0.0149254 | 0.0248756 | True | 200 | 20260828 | Flower_rat + Ridge + 0510_single_date + five spatial scales | Permuted Flower_rat labels; each permutation reruns full outer spatial CV pipeline. |
| Flower_rat | point | 点采样 | 0510_single_date | Ridge | 0.0820138 | 0.0447761 | 0.0559701 | False | 200 | 20260828 | Flower_rat + Ridge + 0510_single_date + five spatial scales | Permuted Flower_rat labels; each permutation reruns full outer spatial CV pipeline. |
| Flower_rat | one_sqft | 1 ft² | 0510_single_date | Ridge | 0.0119265 | 0.0945274 | 0.0945274 | False | 200 | 20260828 | Flower_rat + Ridge + 0510_single_date + five spatial scales | Permuted Flower_rat labels; each permutation reruns full outer spatial CV pipeline. |

FDR=0.05 下显著的尺度：r1, r0p3, r0p5。
当前 R² 最高尺度为 r1，R²=0.291，原始置换 p=0.0050，FDR q=0.0249。
