# 遥感影像与杨花/病害地面指标关系分析 PPT 素材

本文档基于 `/data/jiaxing/analysis/` 中已生成的实验结果整理，可直接复制到 PPT/WPS。  
本次分析只使用原始两期无人机影像：`0418c.tif` 和 `0510c.tif`，未使用 PIF 校正后的 0418 影像。

## 第 1 页：标题页

标题：

```text
无人机多光谱遥感影像与杨花及病害指标关系分析
```

副标题：

```text
基于 4 月 18 日与 5 月 10 日两期无人机影像及 70 个地面采样点
```

页脚信息：

```text
研究区域：嘉兴试验区
数据类型：无人机多光谱影像 + 地面采样点
```

## 第 2 页：研究目标

核心问题：

```text
1. 遥感光谱特征是否能够反映杨花数和杨花率？
2. 遥感光谱特征是否能够反映病害株数、发病率和病害指数？
3. 单期影像特征与两期变化特征中，哪些指标更敏感？
4. 红边和近红外指数是否对病害指标具有更强解释能力？
```

建议讲稿：

```text
本研究希望建立无人机多光谱影像与地面调查指标之间的联系，
重点关注杨花程度和病害程度能否通过遥感特征进行定量表达。
```

## 第 3 页：数据概况

表格：

| 数据 | 文件 | 日期/类型 | 说明 |
|---|---|---|---|
| 0418 影像 | `/data/jiaxing/wang/0418c.tif` | 4 月 18 日 | 6 波段无人机多光谱影像 |
| 0510 影像 | `/data/jiaxing/wang/0510c.tif` | 5 月 10 日 | 6 波段无人机多光谱影像 |
| 地面样点 | `/data/jiaxing/wang/data.shp` | 点位数据 | 共 70 个采样点 |

影像参数：

```text
影像尺寸：2777 x 4502
空间分辨率：约 0.0575 m
波段数：6
坐标系：WGS 84 / UTM zone 51N
```

波段顺序：

| 波段 | 含义 |
|---|---|
| B1 | Blue |
| B2 | Green |
| B3 | Red |
| B4 | RedEdge1 |
| B5 | RedEdge2 |
| B6 | NIR |

## 第 4 页：地面指标与样本量

表格：

| 指标 | 含义 | 有效样本数 | 均值 | 范围 |
|---|---|---:|---:|---:|
| Flower_num | 杨花数 | 33 | 20.73 | 1.00 - 30.00 |
| Flower_rat | 杨花率 | 33 | 69.09 | 3.33 - 100.00 |
| FHB_num | 病害株数 | 63 | 2.52 | 0.00 - 21.00 |
| FHB_rate | 发病率 | 62 | 4.74 | 0.00 - 37.74 |
| FHB_index | 病害指数 | 62 | 0.036 | 0.00 - 0.38 |

建议图：

```text
/data/jiaxing/analysis/figures/Flower_num_hist.png
/data/jiaxing/analysis/figures/FHB_index_hist.png
```

讲稿要点：

```text
杨花相关指标有效样本数为 33，病害相关指标有效样本数约 62-63。
因此后续分析中，每个目标变量均单独使用其非缺失样本，不强行取所有字段的交集。
```

## 第 5 页：技术路线

流程图文字：

```text
两期无人机影像
      ↓
影像对齐检查
      ↓
采样点缓冲区特征提取：0.3 m / 0.5 m / 1.0 m
      ↓
构建波段统计、两期差值、比值、相对变化、植被指数
      ↓
相关性分析：Pearson / Spearman
      ↓
建模分析：Linear / Ridge / Lasso / ElasticNet / PLS / RandomForest
      ↓
筛选敏感遥感指标并解释结果
```

提取统计量：

```text
mean, median, std, min, max, p25, p75, valid_count
```

最终特征规模：

```text
70 个样点 x 1144 个字段
```

## 第 6 页：构建的植被与红边指数

命名指数：

| 指数 | 公式 | 主要意义 |
|---|---|---|
| NDVI | `(NIR - Red) / (NIR + Red)` | 植被活力 |
| GNDVI | `(NIR - Green) / (NIR + Green)` | 叶绿素/冠层活力 |
| NDRE1 | `(NIR - RE1) / (NIR + RE1)` | 红边敏感指数 |
| NDRE2 | `(NIR - RE2) / (NIR + RE2)` | 红边敏感指数 |
| RENDVI1 | `(RE1 - Red) / (RE1 + Red)` | 红边-红光差异 |
| RENDVI2 | `(RE2 - Red) / (RE2 + Red)` | 红边-红光差异 |
| CI_RE1 | `NIR / RE1 - 1` | 红边叶绿素指数 |
| CI_RE2 | `NIR / RE2 - 1` | 红边叶绿素指数 |
| EVI | `2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1)` | 增强植被指数 |
| SAVI | `1.5*(NIR-Red)/(NIR+Red+0.5)` | 土壤调节植被指数 |

同时构建：

```text
0418 指数
0510 指数
diff = 0510 - 0418
ratio = 0510 / 0418
rel = (0510 - 0418) / 0418
```

## 第 7 页：杨花指标相关性结果

核心结果：

| 目标变量 | 最敏感特征 | Spearman | Pearson | n |
|---|---|---:|---:|---:|
| Flower_num | `0510_r1_ND_B2_B3_mean` | -0.742 | -0.642 | 33 |
| Flower_rat | `0510_r1_ND_B2_B3_mean` | -0.742 | -0.642 | 33 |

其中：

```text
ND_B2_B3 = (Green - Red) / (Green + Red)
r1 = 1.0 m 缓冲区
```

命名指数中较强的特征：

| 特征 | 目标 | Spearman |
|---|---|---:|
| `ratio_r1_EVI_mean` | Flower_num / Flower_rat | -0.702 |
| `rel_r1_EVI_mean` | Flower_num / Flower_rat | -0.702 |
| `diff_r1_EVI_mean` | Flower_num / Flower_rat | -0.699 |
| `0510_r1_EVI_mean` | Flower_num / Flower_rat | -0.682 |

建议图：

```text
/data/jiaxing/analysis/figures/Flower_num_top_feature_scatter.png
/data/jiaxing/analysis/figures/Flower_rat_top_feature_scatter.png
```

结论句：

```text
杨花数和杨花率与 5 月 10 日 Green-Red 归一化指数关系最强，
同时 EVI 的两期变化也表现出较强相关性。
```

## 第 8 页：病害指标相关性结果

核心结果：

| 目标变量 | 最敏感特征 | Spearman | Pearson | n |
|---|---|---:|---:|---:|
| FHB_num | `0418_r1_ND_B1_B2_mean` | 0.628 | 0.282 | 63 |
| FHB_rate | `0418_r1_ND_B1_B2_mean` | 0.612 | 0.279 | 62 |
| FHB_index | `diff_r1_CI_RE2_median` | -0.613 | -0.539 | 62 |

红边/近红外相关特征：

| 特征 | 指标 | Spearman |
|---|---|---:|
| `diff_r1_CI_RE2_median` | FHB_index | -0.613 |
| `diff_r1_CI_RE2_median` | FHB_rate | -0.611 |
| `diff_r1_NDRE2_median` | FHB_index | -0.607 |
| `diff_r1_ND_B5_B6_median` | FHB_index | 0.607 |

建议图：

```text
/data/jiaxing/analysis/figures/FHB_index_top_feature_scatter.png
/data/jiaxing/analysis/figures/FHB_rate_top_feature_scatter.png
```

结论句：

```text
病害指标与红边-近红外指数变化关系明显，
尤其是 CI_RE2 和 NDRE2 的两期变化对发病率和病害指数较敏感。
```

## 第 9 页：模型预测结果

最佳模型均为 RandomForest。

| 目标变量 | 验证方式 | n | R2 | RMSE | MAE | Spearman |
|---|---|---:|---:|---:|---:|---:|
| Flower_num | LOOCV | 33 | 0.482 | 6.36 | 5.17 | 0.691 |
| Flower_rat | LOOCV | 33 | 0.485 | 21.15 | 17.21 | 0.691 |
| FHB_num | 5-fold | 63 | -0.019 | 4.53 | 2.86 | 0.561 |
| FHB_rate | 5-fold | 62 | 0.092 | 7.87 | 5.16 | 0.612 |
| FHB_index | 5-fold | 62 | 0.205 | 0.068 | 0.039 | 0.604 |

建议图：

```text
/data/jiaxing/analysis/figures/Flower_num_best_model_observed_predicted.png
/data/jiaxing/analysis/figures/FHB_index_best_model_observed_predicted.png
```

讲稿要点：

```text
杨花指标具有较好的遥感预测潜力，R2 约为 0.48。
病害指数有一定预测能力，但 FHB_num 和 FHB_rate 的回归效果偏弱，
说明病害指标可能受到空间异质性、样本分布偏斜或调查误差影响。
```

## 第 10 页：重要特征解释

RandomForest 重要特征摘要：

| 目标 | 重要特征示例 | 解释 |
|---|---|---|
| Flower_num | `ratio_r0p5_B6_p25`, `rel_r1_EVI_mean` | NIR 和 EVI 的时序变化与杨花程度有关 |
| Flower_rat | `rel_r0p5_B6_p25`, `ratio_r1_DVI_mean` | 与杨花数结果一致 |
| FHB_num | `diff_r1_ND_B5_B6_median`, `diff_r1_CI_RE2_median` | 红边 2 与 NIR 的变化反映病害差异 |
| FHB_rate | `ratio_r1_NDRE2_median`, `diff_r1_ND_B5_B6_median` | 红边指数变化较敏感 |
| FHB_index | `ratio_r1_CI_RE2_median`, `diff_r1_NDRE2_median` | CI_RE2 / NDRE2 是病害指数的关键特征 |

结论句：

```text
杨花指标更依赖 5 月 10 日单期特征和 EVI 变化；
病害指标更依赖红边-近红外指数的两期变化。
```

## 第 11 页：主要结论

可直接放 PPT：

```text
1. 两期无人机多光谱影像能够反映杨花和病害地面指标的变化。

2. 杨花数和杨花率与 5 月 10 日 Green-Red 归一化指数关系最强，
   Spearman 相关系数达到 -0.742。

3. EVI 的两期变化与杨花指标关系明显，
   说明杨花过程可能伴随冠层光谱结构变化。

4. 病害指数与红边-近红外变化特征关系较强，
   其中 diff_r1_CI_RE2_median 与 FHB_index 的 Spearman 相关为 -0.613。

5. 随机森林模型对杨花指标预测效果较好，
   Flower_num 和 Flower_rat 的交叉验证 R2 约为 0.48。

6. 病害指标中 FHB_index 的预测效果优于 FHB_num 和 FHB_rate，
   说明病害指数可能比单纯病株数更适合与遥感连续变量建立联系。
```

## 第 12 页：不足与后续工作

可直接放 PPT：

```text
1. 杨花指标有效样本数仅 33 个，样本量偏小。

2. FHB_num 分布偏斜，低值样本较多、少数高值样本影响模型稳定性。

3. 当前仅使用两期影像，难以完整刻画杨花和病害发展的时间过程。

4. 后续可增加更多时相影像、更多地面样点，并尝试分区建模或空间交叉验证。

5. 可进一步结合冠层高度、纹理特征、植株行间结构等信息，
   提高对病害程度的解释能力。
```

## 推荐插图清单

地面指标分布：

```text
/data/jiaxing/analysis/figures/Flower_num_hist.png
/data/jiaxing/analysis/figures/Flower_rat_hist.png
/data/jiaxing/analysis/figures/FHB_num_hist.png
/data/jiaxing/analysis/figures/FHB_rate_hist.png
/data/jiaxing/analysis/figures/FHB_index_hist.png
```

最强相关散点图：

```text
/data/jiaxing/analysis/figures/Flower_num_top_feature_scatter.png
/data/jiaxing/analysis/figures/Flower_rat_top_feature_scatter.png
/data/jiaxing/analysis/figures/FHB_num_top_feature_scatter.png
/data/jiaxing/analysis/figures/FHB_rate_top_feature_scatter.png
/data/jiaxing/analysis/figures/FHB_index_top_feature_scatter.png
```

观测值-预测值图：

```text
/data/jiaxing/analysis/figures/Flower_num_best_model_observed_predicted.png
/data/jiaxing/analysis/figures/Flower_rat_best_model_observed_predicted.png
/data/jiaxing/analysis/figures/FHB_num_best_model_observed_predicted.png
/data/jiaxing/analysis/figures/FHB_rate_best_model_observed_predicted.png
/data/jiaxing/analysis/figures/FHB_index_best_model_observed_predicted.png
```

## 结果文件链接

```text
/data/jiaxing/analysis/correlation_summary.csv
/data/jiaxing/analysis/model_metrics.csv
/data/jiaxing/analysis/feature_importance.csv
/data/jiaxing/analysis/all_features_targets.csv
```
