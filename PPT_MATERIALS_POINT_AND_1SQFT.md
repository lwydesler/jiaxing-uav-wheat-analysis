# 点采样与 1 平方尺面积采样对比 PPT 素材

本文档用于把 **点采样结果** 和 **1 平方尺面积采样结果** 放在同一份 PPT 中展示。  
建议正式汇报以 1 平方尺面积采样为主，点采样作为尺度敏感性对比。

## 第 1 页：标题页

标题：

```text
不同遥感采样尺度下杨花与病害指标响应分析
```

副标题：

```text
点采样 vs 1 平方尺等面积采样
```

## 第 2 页：为什么要比较采样尺度

关键背景：

```text
地面 DATA 数据对应 1 平方尺样方面积。
无人机影像空间分辨率约为 0.05753 m。
1 平方尺约等于 0.0929 m2，约对应 28 个影像像素。
```

尺度换算：

| 采样方式 | 对应面积 | 约等于像素数 | 说明 |
|---|---:|---:|---|
| 点采样 | 1 个像元 | 1 像素 | 只取点位所在像元 |
| 1 平方尺等面积圆 | 0.0929 m2 | 约 28 像素 | 更接近地面样方面积 |

建议讲稿：

```text
如果只取点位所在像元，代表面积远小于地面调查样方；
如果使用 1 平方尺等面积区域，则遥感统计尺度与地面调查尺度更加一致。
因此需要比较不同尺度下的结果稳定性。
```

## 第 3 页：分析数据与特征

输入数据：

| 数据 | 路径 | 说明 |
|---|---|---|
| 0418 影像 | `/data/jiaxing/wang/0418c.tif` | 4 月 18 日原始影像 |
| 0510 影像 | `/data/jiaxing/wang/0510c.tif` | 5 月 10 日原始影像 |
| 地面样点 | `/data/jiaxing/wang/data.shp` | 70 个样点 |

波段顺序：

```text
B1 = Blue
B2 = Green
B3 = Red
B4 = RedEdge1
B5 = RedEdge2
B6 = NIR
```

地面指标：

```text
Flower_num, Flower_rat, FHB_num, FHB_rate, FHB_index
```

## 第 4 页：两种采样方式

点采样：

```text
直接提取采样点所在像元值。
优点：简单、避免混入邻域信息。
不足：只代表约 0.00331 m2，远小于 1 平方尺样方。
```

1 平方尺面积采样：

```text
1 平方尺 = 0.092903 m2
等面积圆半径 = 0.172 m
约包含 28 个像素。
```

建议图示：

```text
采样点 → 单像元
采样点 → 0.172 m 等面积圆
```

## 第 5 页：杨花指标相关性对比

| 采样方式 | 目标变量 | 最敏感特征 | Spearman | Pearson | n |
|---|---|---|---:|---:|---:|
| 点采样 | Flower_num | `diff_rpt_ND_B1_B5_mean` | 0.599 | 0.458 | 33 |
| 点采样 | Flower_rat | `diff_rpt_ND_B1_B5_mean` | 0.599 | 0.458 | 33 |
| 1 平方尺 | Flower_num | `0510_r0p172_ND_B2_B3_median` | -0.659 | -0.526 | 33 |
| 1 平方尺 | Flower_rat | `0510_r0p172_ND_B2_B3_median` | -0.659 | -0.526 | 33 |

解释：

```text
两种尺度下，杨花指标均能找到中等偏强相关的遥感特征。
但 1 平方尺面积采样更接近地面调查尺度，相关性略强。
```

建议图：

```text
/data/jiaxing/analysis_point/figures/Flower_num_top_feature_scatter.png
/data/jiaxing/analysis_1sqft_circle/figures/Flower_num_top_feature_scatter.png
```

## 第 6 页：病害指标相关性对比

| 采样方式 | 目标变量 | 最敏感特征 | Spearman | Pearson | n |
|---|---|---|---:|---:|---:|
| 点采样 | FHB_num | `diff_rpt_CI_RE2_mean` | -0.470 | -0.479 | 63 |
| 点采样 | FHB_rate | `diff_rpt_CI_RE2_mean` | -0.463 | -0.490 | 62 |
| 点采样 | FHB_index | `diff_rpt_CI_RE2_mean` | -0.502 | -0.476 | 62 |
| 1 平方尺 | FHB_num | `ratio_r0p172_B2_p75` | 0.463 | 0.399 | 63 |
| 1 平方尺 | FHB_rate | `rel_r0p172_B2_p75` | 0.501 | 0.448 | 62 |
| 1 平方尺 | FHB_index | `rel_r0p172_B2_p75` | 0.480 | 0.403 | 62 |

解释：

```text
病害指标在点采样下更突出红边-近红外变化特征 CI_RE2；
在 1 平方尺面积采样下，更突出绿色波段 B2 的两期变化。
这说明病害遥感响应对采样尺度较敏感。
```

建议图：

```text
/data/jiaxing/analysis_point/figures/FHB_index_top_feature_scatter.png
/data/jiaxing/analysis_1sqft_circle/figures/FHB_index_top_feature_scatter.png
```

## 第 7 页：模型预测对比

| 采样方式 | 目标变量 | 最佳模型 | R2 | RMSE | MAE | Spearman |
|---|---|---|---:|---:|---:|---:|
| 点采样 | Flower_num | RandomForest | 0.010 | 8.80 | 7.55 | 0.294 |
| 点采样 | Flower_rat | RandomForest | 0.010 | 29.34 | 25.16 | 0.289 |
| 点采样 | FHB_num | Ridge | 0.166 | 4.10 | 2.61 | 0.508 |
| 点采样 | FHB_rate | RandomForest | 0.223 | 7.28 | 4.56 | 0.522 |
| 点采样 | FHB_index | Ridge | 0.270 | 0.065 | 0.040 | 0.558 |
| 1 平方尺 | Flower_num | RandomForest | 0.096 | 8.41 | 6.85 | 0.464 |
| 1 平方尺 | Flower_rat | RandomForest | 0.096 | 28.02 | 22.81 | 0.449 |
| 1 平方尺 | FHB_num | RandomForest | 0.062 | 4.35 | 2.70 | 0.378 |
| 1 平方尺 | FHB_rate | RandomForest | -0.014 | 8.32 | 5.25 | 0.313 |
| 1 平方尺 | FHB_index | ElasticNet | 0.063 | 0.073 | 0.042 | 0.468 |

讲稿要点：

```text
点采样对病害指标的模型表现更好，尤其 FHB_index 的 R2 达到 0.270；
1 平方尺面积采样对杨花指标表现略好。
这可能说明杨花更具有样方尺度的冠层响应，而病害局部症状在单像元尺度上更明显。
```

## 第 8 页：尺度效应解释

可直接放 PPT：

```text
1. 点采样只代表一个像元，面积约为 0.00331 m2，仅为 1 平方尺的约 3.6%。

2. 1 平方尺面积采样约包含 28 个像素，更接近地面调查面积。

3. 杨花指标在 1 平方尺面积尺度下相关性更强，说明其更接近冠层整体状态。

4. 病害指标在点采样下模型表现更好，可能与病斑或病株的局部光谱异常有关。

5. 不同尺度得到的敏感特征不同，说明遥感-地面指标关系存在明显空间尺度效应。
```

## 第 9 页：推荐汇报结论

正式结论建议：

```text
本研究同时比较了点采样和 1 平方尺等面积采样两种尺度。
结果表明，杨花指标在面积尺度下与 5 月 10 日 Green-Red 归一化指数关系更稳定；
病害指标在点采样下与红边-近红外变化特征关系更明显。
这说明杨花和病害的遥感响应具有不同空间尺度特征。
```

更谨慎的表述：

```text
由于地面样点数量有限，当前模型预测精度仍然有限。
相关性结果可作为敏感波段和指数筛选依据，
后续需要更多样点和多时相影像进一步验证。
```

## 第 10 页：PPT 图件建议

点采样图件：

```text
/data/jiaxing/analysis_point/figures/Flower_num_top_feature_scatter.png
/data/jiaxing/analysis_point/figures/FHB_index_top_feature_scatter.png
/data/jiaxing/analysis_point/figures/FHB_index_best_model_observed_predicted.png
```

1 平方尺图件：

```text
/data/jiaxing/analysis_1sqft_circle/figures/Flower_num_top_feature_scatter.png
/data/jiaxing/analysis_1sqft_circle/figures/FHB_index_top_feature_scatter.png
/data/jiaxing/analysis_1sqft_circle/figures/Flower_num_best_model_observed_predicted.png
```

结果表：

```text
/data/jiaxing/analysis_point/correlation_summary.csv
/data/jiaxing/analysis_point/model_metrics.csv
/data/jiaxing/analysis_1sqft_circle/correlation_summary.csv
/data/jiaxing/analysis_1sqft_circle/model_metrics.csv
```

## 文件链接

点采样结果：

```text
/data/jiaxing/analysis_point/
```

1 平方尺面积采样结果：

```text
/data/jiaxing/analysis_1sqft_circle/
```

分析脚本：

```text
/data/jiaxing/rs_field_analysis.py
```
