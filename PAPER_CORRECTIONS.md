# 论文分析修正与运行命令

本次修正不重新训练回归模型。新的相关性结果独立保存，原有模型结果和历史审计报告保留。

## 1. 更新代码与环境

以下假设仓库位于 `/data/jiaxing`，原始影像位于 `/data/jiaxing/wang`。

```bash
cd /data/jiaxing
git pull --ff-only origin main
conda install -n rs -c conda-forge numpy pandas scipy rasterio
```

已有依赖时可跳过安装。独立相关性脚本仅需 NumPy、Pandas、SciPy；新增制图脚本需 Rasterio、NumPy。旧的整体分析流程仍使用原有 GDAL 等依赖。

## 2. 按5月10日同期口径重新计算相关性

```bash
conda run --no-capture-output -n rs python paper_audit/audit_scripts/rerun_correlation_audit.py \
  --data-root /data/jiaxing \
  --feature-scope 0510 \
  --n-bootstrap 2000 \
  --n-permutation 1000 \
  --seed 20260803
```

默认输出到 `paper_audit/correlation_corrected_0510/`：

- `CORRELATION_ALL.csv`：所有入选候选特征的相关系数、FDR、置信区间和稳定性。
- `TOP_FEATURES.csv`：每个尺度按绝对 Spearman 相关系数排名前30的特征。
- `RUN_MANIFEST.json`：样本编号、候选特征清单、输入及脚本哈希、软件版本、次数与随机种子。

脚本读取仓库已有的三份 `all_features_targets.csv`，分别分析五种尺度。保留扬花率非缺失的样点，并核验不同尺度的样本编号和响应值完全一致；不以预测表现筛选样本。排除有效像元计数、常数及全缺失列。候选光谱列保留重复项，并列排名按输入列顺序处理，因此排名频率取决于候选集合。FDR的校正范围在运行记录中明确保存。

重复运行同一输出目录需显式添加 `--overwrite`，也可用 `--out-dir` 指定另一目录。

若保留双时相补充分析，单独执行：

```bash
conda run --no-capture-output -n rs python paper_audit/audit_scripts/rerun_correlation_audit.py \
  --data-root /data/jiaxing --feature-scope all \
  --n-bootstrap 2000 --n-permutation 1000 --seed 20260803
```

结果写入 `paper_audit/correlation_corrected_all/`。两种范围的FDR值、最大统计量置换检验及排名频率不可混用。

### 统计修正的含义

旧实现对全样本秩进行重采样，并将每次抽样相关系数强制赋予原始相关方向，会影响置信区间和稳定性估计。现在对原始观测成对抽样，在每次抽样内部重新计算平均秩，保留实际正负号；存在缺失值时，对每个特征使用共同有效观测对重新计算两变量的秩。无定义的抽样结果不进入置信区间或排名，另报告有效抽样次数。

最大统计量置换检验也使用成对有效观测上的准确 Spearman 相关。其p值检验一个尺度内候选特征的最大绝对相关性，不是每个特征各自的置换p值。Bootstrap采用样点成对独立重采样，置换采用不受限制的标签置换；它们依赖独立性/可交换性假设，不能代替空间交叉验证或空间置信区间。

`run_paper_audit.py` 的相关性部分同步调用修正后的函数；该旧脚本仍保留自己的候选集合和默认次数。论文更新请使用上面的独立脚本，避免同时覆盖历史综合报告和模型结果。

## 3. 绘制敏感光谱指数图

```bash
conda run --no-capture-output -n rs python map_sensitive_index.py \
  --image /data/jiaxing/wang/0510c.tif \
  --out /data/jiaxing/sensitive_index_maps/0510_green_red_median_r0p172.tif \
  --radius 0.172 --tile-size 256
```

输出为带地理坐标的单波段 GeoTIFF，可在 QGIS/ArcGIS 中按连续色带排版。图名使用“小麦扬花敏感光谱指数空间分布”，图例单位为无量纲指数。

计算严格遵循样点特征 `0510_r0p172_ND_B2_B3_median` 的顺序：

```text
G = 圆形窗口内B2有效像元的中位数
R = 圆形窗口内B3有效像元的中位数
指数 = (G - R) / (G + R)
```

这不是“先计算逐像元指数，再对指数取中位数”。论文方法和表头建议称为“基于波段窗口中位数构建的绿光–红光归一化差异指数”。

窗口按影像实际分辨率计算，不硬编码0.04 m或0.0575 m。输入须为以米为单位的无旋转投影影像，B2为绿光、B3为红光。每个输出像元以其中心为圆心；原始RTK样点可能位于像元中心之外，因此相同半径下包含的像元数可能不同，地图像元值不保证与最近样点特征完全相同。

分块处理包含完整邻域，避免块边界接缝；影像外侧窗口截断，各波段独立排除无效像元。任一波段窗口完全无有效像元或分母绝对值不大于1e-6时输出NoData。边缘或缺测区域使用的有效像元可能较少，应在正式插图中结合种植区范围检查。

有与原图完全对齐的单波段小麦区域掩膜时，添加 `--mask /path/to/wheat_mask.tif`，仅保留掩膜值大于0的输出位置。掩膜不改变邻域波段统计。未提供掩膜时输出包含影像中所有具有有效邻域的地物，正式小麦区域插图需要另行裁剪。

该脚本输出敏感指数，不加载回归模型、不将指数换算成扬花率百分比。旧 `predict_flower_rat_map.py` 属于历史随机森林制图流程，不作为本次论文指数制图入口。

## 4. 验证

```bash
conda run --no-capture-output -n rs python -m unittest discover -s tests -v
```

测试以SciPy的逐次计算校验带并列值/缺失值的相关系数和Bootstrap区间，并检查相关方向反转；用合成GeoTIFF逐像元对照圆形窗口计算，检查NoData、影像边缘、不同分块大小及掩膜处理。

正式论文中应以新输出替换旧的置信区间、FDR及稳定性数字。回归模型的R²、RMSE等指标不由本次相关性重算更新。试验处理、样点与小区对应关系、田间扬花判定标准仍需依据田间记录填写，代码不能反推这些事实。
