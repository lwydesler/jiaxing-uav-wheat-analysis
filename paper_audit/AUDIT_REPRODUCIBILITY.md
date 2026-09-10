# 审计复现说明

## 软件环境

```json
{
  "python": "3.10.20 | packaged by conda-forge | (main, Mar  5 2026, 16:42:22) [GCC 14.3.0]",
  "platform": "Linux-6.17.0-29-generic-x86_64-with-glibc2.39",
  "numpy": "2.2.6",
  "pandas": "2.3.3",
  "geopandas": "1.1.3",
  "gdal": "GDAL 3.10.3, released 2025/04/01",
  "scikit_learn": "1.7.2",
  "scipy": "1.15.2"
}
```

## 随机种子

- 主随机种子：`20260803`
- 相关性 Bootstrap：`20260803`
- 严格模型 Bootstrap：`20260880`
- 模型置换检验：`20260974`

## 运行命令

```bash
cd /data/jiaxing
conda run -n rs python paper_audit/audit_scripts/run_paper_audit.py
```

## 主要输入文件及大小

- `/data/jiaxing/analysis_point/field_samples_clean.csv`: 0.004 MB
- `/data/jiaxing/analysis_point/all_features_targets.csv`: 0.396 MB
- `/data/jiaxing/analysis_1sqft_circle/all_features_targets.csv`: 0.424 MB
- `/data/jiaxing/analysis/all_features_targets.csv`: 1.266 MB
- `/data/jiaxing/wang/0418c.tif`: 188.136 MB
- `/data/jiaxing/wang/0510c.tif`: 198.723 MB
- `/data/jiaxing/wang/data.shp`: 0.002 MB

## 新增审计脚本

- `paper_audit/audit_scripts/run_paper_audit.py`

## 未能完全核实的项目

- 传感器型号、飞行高度、具体拍摄时间、太阳高度、曝光参数和校准板信息：当前项目材料中未找到。
- 两期影像是否完成辐射定标或反射率转换：当前元数据和代码无法证明。
- 试验小区边界、品种、处理、重复区和样点从属关系：当前样点表中未发现明确字段。
- 缓冲区是否跨越小区边界：缺少小区边界或样方多边形，无法核实。

## 最短复现步骤

1. 进入项目根目录 `/data/jiaxing`。
2. 使用已有 `rs` conda 环境运行上方命令。
3. 查看 `paper_audit/AUDIT_REPORT.md`、`paper_audit/AUDIT_KEY_RESULTS.csv`、`paper_audit/AUDIT_TOP_FEATURES.csv` 和 `paper_audit/AUDIT_SPATIAL_SCALE.csv`。
