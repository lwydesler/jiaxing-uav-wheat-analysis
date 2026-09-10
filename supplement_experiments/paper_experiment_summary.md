# 补充实验结果摘要

## 1. 田间指标样本量

| 指标 | 有效样本数 | 最小值 | 中位数 | 最大值 | 正值样本数 |
|---|---:|---:|---:|---:|---:|
| Flower_num | 33 | 1.000 | 23.000 | 30.000 | 33 |
| Flower_rat | 33 | 3.330 | 76.670 | 100.000 | 33 |
| FHB_num | 63 | 0.000 | 1.000 | 21.000 | 34 |
| FHB_rate | 62 | 0.000 | 1.580 | 37.740 | 34 |
| FHB_index | 62 | 0.000 | 0.005 | 0.380 | 31 |

## 2. 各尺度最强相关特征

| 指标 | 尺度 | 最强特征 | Spearman r | p值 |
|---|---|---|---:|---:|
| FHB_index | 0.3/0.5/1.0m | diff_r1_CI_RE2_median | -0.613 | 1.186e-07 |
| FHB_index | 1平方尺 | rel_r0p172_B2_p75 | 0.480 | 7.954e-05 |
| FHB_index | 点采样 | diff_rpt_CI_RE2_mean | -0.502 | 3.231e-05 |
| FHB_num | 0.3/0.5/1.0m | 0418_r1_ND_B1_B2_mean | 0.628 | 3.523e-08 |
| FHB_num | 1平方尺 | ratio_r0p172_B2_p75 | 0.463 | 0.0001314 |
| FHB_num | 点采样 | diff_rpt_CI_RE2_mean | -0.470 | 0.0001001 |
| FHB_rate | 0.3/0.5/1.0m | 0418_r1_ND_B1_B2_mean | 0.612 | 1.241e-07 |
| FHB_rate | 1平方尺 | rel_r0p172_B2_p75 | 0.501 | 3.32e-05 |
| FHB_rate | 点采样 | diff_rpt_CI_RE2_mean | -0.463 | 0.0001497 |
| Flower_num | 0.3/0.5/1.0m | 0510_r1_ND_B2_B3_mean | -0.742 | 7.894e-07 |
| Flower_num | 1平方尺 | 0510_r0p172_ND_B2_B3_median | -0.659 | 3.081e-05 |
| Flower_num | 点采样 | diff_rpt_ND_B1_B5_mean | 0.599 | 0.0002288 |
| Flower_rat | 0.3/0.5/1.0m | 0510_r1_ND_B2_B3_mean | -0.742 | 7.894e-07 |
| Flower_rat | 1平方尺 | 0510_r0p172_ND_B2_B3_median | -0.659 | 3.081e-05 |
| Flower_rat | 点采样 | diff_rpt_ND_B1_B5_mean | 0.599 | 0.0002288 |

## 3. 各尺度最佳回归模型

| 指标 | 尺度 | 模型 | n | R2 | RMSE | MAE |
|---|---|---|---:|---:|---:|---:|
| FHB_index | 0.3/0.5/1.0m | RandomForest | 62 | 0.205 | 0.068 | 0.039 |
| FHB_index | 1平方尺 | ElasticNet | 62 | 0.063 | 0.073 | 0.042 |
| FHB_index | 点采样 | Ridge | 62 | 0.270 | 0.065 | 0.040 |
| FHB_num | 0.3/0.5/1.0m | RandomForest | 63 | -0.019 | 4.532 | 2.862 |
| FHB_num | 1平方尺 | RandomForest | 63 | 0.062 | 4.348 | 2.703 |
| FHB_num | 点采样 | Ridge | 63 | 0.166 | 4.099 | 2.609 |
| FHB_rate | 0.3/0.5/1.0m | RandomForest | 62 | 0.092 | 7.873 | 5.164 |
| FHB_rate | 1平方尺 | RandomForest | 62 | -0.014 | 8.320 | 5.253 |
| FHB_rate | 点采样 | RandomForest | 62 | 0.223 | 7.284 | 4.562 |
| Flower_num | 0.3/0.5/1.0m | RandomForest | 33 | 0.482 | 6.363 | 5.170 |
| Flower_num | 1平方尺 | RandomForest | 33 | 0.096 | 8.406 | 6.846 |
| Flower_num | 点采样 | RandomForest | 33 | 0.010 | 8.801 | 7.547 |
| Flower_rat | 0.3/0.5/1.0m | RandomForest | 33 | 0.485 | 21.153 | 17.207 |
| Flower_rat | 1平方尺 | RandomForest | 33 | 0.096 | 28.020 | 22.813 |
| Flower_rat | 点采样 | RandomForest | 33 | 0.010 | 29.337 | 25.164 |

## 4. 病害有无分类

病害有无由 FHB_num、FHB_rate、FHB_index 中任一指标大于 0 判定。

| 尺度 | 模型 | n | 阳性 | 阴性 | Balanced accuracy | ROC AUC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 点采样 | LogisticRegression | 63 | 34 | 29 | 0.649 | 0.702 | 0.676 |
| 点采样 | RandomForestClassifier | 63 | 34 | 29 | 0.634 | 0.718 | 0.657 |
| 1平方尺 | LogisticRegression | 63 | 34 | 29 | 0.698 | 0.744 | 0.716 |
| 1平方尺 | RandomForestClassifier | 63 | 34 | 29 | 0.710 | 0.683 | 0.743 |

## 5. 写作建议

这些补充实验适合支撑论文中的“尺度效应、时间变化特征、特征类型响应、回归预测和病害分类”五个小节。当前样本量有限，回归预测宜表述为趋势估计或相对风险制图，不宜表述为高精度业务化反演。
