# 资源目录说明 (resources/weights_cluster_eval)

本目录用于存放项目运行所需的模型权重、聚类分析结果以及测评报告。

## 1. weights/
存放训练生成的 LightGBM 模型、路由专家模型及其它权重文件。
- `lightgbm_model.txt`: 主分类模型。
- `lightgbm_model_normal.txt`: 普通样本专家模型。
- `lightgbm_model_packed.txt`: 加壳样本专家模型。
- `gating_model.pth`: 路由专家系统的门控模型权重。

## 2. cluster/
存放 HDBSCAN 聚类结果、家族分类器以及特征持久化文件。
- `family_classifier.pkl`: 家族分类器模型。
- `extracted_features.pkl`: 特征缓存，用于加速聚类与训练流程。
- `cluster_statistics.json`: 簇分布统计信息。

## 3. eval/
存放模型性能评估报告、可视化图表及扫描结果。
- `model_evaluation.png`: 混淆矩阵与性能指标可视化。
- `model_auc_curve.png`: ROC-AUC 曲线。
- `routing_evaluation_report.txt`: 路由门控系统的测评报告。
- `scan_results/`: 包含单次扫描任务的详细 JSON/CSV 输出。

## 4. 注意事项
- 严禁将大体积权重文件（>50MB）直接提交至 Git 仓库。
- 资源文件应按日期或版本号进行划分子目录，以便进行多版本比对。
