# 1 平方尺采样面积版本 PPT 素材

本版本基于地面 `DATA` 采样面积为 **1 平方尺** 重新分析。  
1 平方尺 = `0.092903 m2`。由于当前脚本使用圆形缓冲区，本次采用等面积圆近似：

```text
等面积圆半径 = sqrt(0.092903 / pi) = 0.172 m
```

新结果目录：

```text
/data/jiaxing/analysis_1sqft_circle/
```

注意：之前 `/data/jiaxing/analysis/` 使用 `0.3 m / 0.5 m / 1.0 m` 缓冲区，其中 `1.0 m` 圆形缓冲约等于 `33.8 平方尺`，明显大于地面调查面积。正式 PPT 建议优先使用本文件中的结果。

## 数据与方法

输入数据：

| 数据 | 路径 | 说明 |
|---|---|---|
| 0418 影像 | `/data/jiaxing/wang/0418c.tif` | 4 月 18 日原始无人机影像 |
| 0510 影像 | `/data/jiaxing/wang/0510c.tif` | 5 月 10 日原始无人机影像 |
| 地面样点 | `/data/jiaxing/wang/data.shp` | 70 个采样点，采样面积 1 平方尺 |

波段顺序：

| 波段 | 含义 |
|---|---|
| B1 | Blue |
| B2 | Green |
| B3 | Red |
| B4 | RedEdge1 |
| B5 | RedEdge2 |
| B6 | NIR |

地面指标有效样本数：

| 指标 | 有效样本数 |
|---|---:|
| Flower_num | 33 |
| Flower_rat | 33 |
| FHB_num | 63 |
| FHB_rate | 62 |
| FHB_index | 62 |

## 1 平方尺尺度下的主要相关性

| 目标变量 | 最敏感特征 | Spearman | Pearson | n |
|---|---|---:|---:|---:|
| Flower_num | `0510_r0p172_ND_B2_B3_median` | -0.659 | -0.526 | 33 |
| Flower_rat | `0510_r0p172_ND_B2_B3_median` | -0.659 | -0.526 | 33 |
| FHB_num | `ratio_r0p172_B2_p75` | 0.463 | 0.399 | 63 |
| FHB_rate | `rel_r0p172_B2_p75` | 0.501 | 0.448 | 62 |
| FHB_index | `rel_r0p172_B2_p75` | 0.480 | 0.403 | 62 |

其中：

```text
ND_B2_B3 = (Green - Red) / (Green + Red)
r0p172 = 0.172 m 等面积圆，约等于 1 平方尺
B2 = Green
p75 = 缓冲区内第 75 百分位数
```

## 杨花指标结果

核心结果：

```text
Flower_num 与 0510_r0p172_ND_B2_B3_median:
Spearman = -0.659
Pearson  = -0.526
n = 33

Flower_rat 与 0510_r0p172_ND_B2_B3_median:
Spearman = -0.659
Pearson  = -0.526
n = 33
```

解释：

```text
在与 1 平方尺采样面积相匹配的尺度下，
杨花数和杨花率仍然与 5 月 10 日 Green-Red 归一化指数存在较强负相关。
```

建议图：

```text
/data/jiaxing/analysis_1sqft_circle/figures/Flower_num_top_feature_scatter.png
/data/jiaxing/analysis_1sqft_circle/figures/Flower_rat_top_feature_scatter.png
```

## 病害指标结果

核心结果：

```text
FHB_num:
ratio_r0p172_B2_p75
Spearman = 0.463

FHB_rate:
rel_r0p172_B2_p75
Spearman = 0.501

FHB_index:
rel_r0p172_B2_p75
Spearman = 0.480
```

解释：

```text
在 1 平方尺尺度下，病害指标与绿色波段 B2 的两期变化关系较突出。
相比 1m 缓冲区结果，红边-近红外变化特征的优势减弱，
说明病害遥感信号对空间尺度比较敏感。
```

建议图：

```text
/data/jiaxing/analysis_1sqft_circle/figures/FHB_num_top_feature_scatter.png
/data/jiaxing/analysis_1sqft_circle/figures/FHB_rate_top_feature_scatter.png
/data/jiaxing/analysis_1sqft_circle/figures/FHB_index_top_feature_scatter.png
```

## 模型结果

| 目标变量 | 最佳模型 | 验证方式 | R2 | RMSE | MAE | Spearman |
|---|---|---|---:|---:|---:|---:|
| Flower_num | RandomForest | LOOCV | 0.096 | 8.41 | 6.85 | 0.464 |
| Flower_rat | RandomForest | LOOCV | 0.096 | 28.02 | 22.81 | 0.449 |
| FHB_num | RandomForest | 5-fold | 0.062 | 4.35 | 2.70 | 0.378 |
| FHB_rate | RandomForest | 5-fold | -0.014 | 8.32 | 5.25 | 0.313 |
| FHB_index | ElasticNet | 5-fold | 0.063 | 0.073 | 0.042 | 0.468 |

解释：

```text
与 1m 缓冲区相比，1 平方尺尺度下模型预测能力明显降低。
这说明单个 1 平方尺样方内的光谱信息可能不足以稳定预测病害和杨花程度，
或者点位误差、冠层空间异质性对结果影响较大。
```

## 与旧版 1m 缓冲结果的比较

| 结论点 | 旧版 1m 缓冲 | 1 平方尺等面积圆 |
|---|---|---|
| 杨花最强相关 | Spearman 约 -0.742 | Spearman 约 -0.659 |
| 病害指数最强相关 | Spearman 约 -0.613 | Spearman 约 0.480 |
| 杨花模型 R2 | 约 0.48 | 约 0.10 |
| 病害指数模型 R2 | 约 0.21 | 约 0.06 |

建议表述：

```text
1m 缓冲区结果显示更强的遥感关系，但其覆盖面积远大于地面样方面积。
1 平方尺尺度下相关性和模型精度下降，说明尺度匹配对遥感-地面关系分析影响显著。
正式结论应以 1 平方尺尺度为主，1m 缓冲结果可作为邻域尺度敏感性分析。
```

## 修正后的主要结论

```text
1. 地面采样面积为 1 平方尺，对应等面积圆半径约 0.172 m。

2. 在尺度匹配后，杨花数和杨花率仍与 5 月 10 日 Green-Red 归一化指数保持较强负相关，
   Spearman 相关系数约为 -0.659。

3. 病害指标与绿色波段 B2 的两期变化关系较明显，
   但相关强度和模型预测能力低于旧版 1m 缓冲结果。

4. 旧版 1m 缓冲区可能反映了样点周边冠层整体状态，
   而不是严格的 1 平方尺样方面积，因此不宜作为唯一正式结论。

5. 后续建议同时报告 1 平方尺尺度和邻域尺度结果，
   用于讨论采样尺度与遥感空间尺度匹配问题。
```

## 新结果文件

```text
/data/jiaxing/analysis_1sqft_circle/field_samples_clean.csv
/data/jiaxing/analysis_1sqft_circle/sample_rs_features.csv
/data/jiaxing/analysis_1sqft_circle/all_features_targets.csv
/data/jiaxing/analysis_1sqft_circle/correlation_summary.csv
/data/jiaxing/analysis_1sqft_circle/model_metrics.csv
/data/jiaxing/analysis_1sqft_circle/feature_importance.csv
/data/jiaxing/analysis_1sqft_circle/figures/
```
