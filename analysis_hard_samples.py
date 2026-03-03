import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler


def resolve_json_path(json_arg: str | None, cluster_dir: Path, prefixes: list[str], arg_name: str) -> Path:
    if json_arg:
        path = Path(json_arg)
        if path.exists():
            return path
        matches = []
        for prefix in prefixes:
            matches.extend(cluster_dir.glob(f"{prefix}_*.json"))
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            raise FileNotFoundError(f"{arg_name} 指定文件不存在: {path}。可用最新文件: {matches[0]}")
        raise FileNotFoundError(f"{arg_name} 指定文件不存在: {path}，且未在 {cluster_dir} 找到匹配文件")

    matches = []
    for prefix in prefixes:
        matches.extend(cluster_dir.glob(f"{prefix}_*.json"))
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"未在 {cluster_dir} 找到匹配文件，前缀: {prefixes}")


def load_samples(json_path: Path, sample_type: str) -> pd.DataFrame:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for item in records:
        row = {
            "sample_type": sample_type,
            "sample_id": str(item.get("sample_id", "")),
            "true_label": int(item.get("true_label", 0)),
            "predicted_label": int(item.get("predicted_label", 0)),
            "prediction_probability": float(item.get("prediction_probability", 0.0)),
        }
        feature_importance = item.get("feature_importance", {}) or {}
        for k, v in feature_importance.items():
            row[str(k)] = float(v)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["sample_type", "sample_id", "true_label", "predicted_label", "prediction_probability"])
    return pd.DataFrame(rows)


def summarize_file(df: pd.DataFrame, output_dir: Path, file_name: str, topn: int = 20) -> dict:
    feature_cols = [c for c in df.columns if re.match(r"^Column_\d+$", c)]
    feature_cols = sorted(feature_cols, key=lambda x: int(x.split("_")[1]))
    stats = {
        "file_name": file_name,
        "sample_count": int(len(df)),
        "label_distribution_true": df["true_label"].value_counts(dropna=False).to_dict(),
        "label_distribution_predicted": df["predicted_label"].value_counts(dropna=False).to_dict(),
        "probability_summary": df["prediction_probability"].describe().to_dict(),
        "feature_count": int(len(feature_cols)),
    }
    if feature_cols:
        desc = df[feature_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        desc["range"] = desc["max"] - desc["min"]
        desc = desc.sort_values("range", ascending=False)
        desc.head(topn).to_csv(output_dir / f"{Path(file_name).stem}_top{topn}_feature_ranges.csv", encoding="utf-8-sig")
        stats["top_feature_ranges"] = desc.head(topn)[["min", "max", "range", "mean", "std"]].to_dict(orient="index")
    return stats


def build_feature_matrix(df_all: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [c for c in df_all.columns if re.match(r"^Column_\d+$", c)]
    feature_cols = sorted(feature_cols, key=lambda x: int(x.split("_")[1]))
    feature_df = df_all[feature_cols].fillna(0.0).astype(np.float32)
    return feature_df, feature_cols


def plot_boxplots(df_all: pd.DataFrame, feature_cols: list[str], output_dir: Path, topn: int = 12) -> None:
    if not feature_cols:
        return
    mean_by_group = df_all.groupby("sample_type")[feature_cols].mean()
    if len(mean_by_group) < 2:
        return
    score = (mean_by_group.max(axis=0) - mean_by_group.min(axis=0)).sort_values(ascending=False)
    selected = score.head(topn).index.tolist()
    melted = df_all[["sample_type"] + selected].melt(id_vars=["sample_type"], var_name="feature", value_name="value")
    plt.figure(figsize=(max(12, topn * 1.2), 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="sample_type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_boxplot_top_diff.png", dpi=220)
    plt.close()


def plot_group_heatmap(df_all: pd.DataFrame, feature_cols: list[str], output_dir: Path, topn: int = 40) -> None:
    if not feature_cols:
        return
    group_mean = df_all.groupby("sample_type")[feature_cols].mean()
    if group_mean.empty:
        return
    span = (group_mean.max(axis=0) - group_mean.min(axis=0)).sort_values(ascending=False)
    selected = span.head(topn).index.tolist()
    plt.figure(figsize=(max(12, topn * 0.28), 4))
    sns.heatmap(group_mean[selected], cmap="YlOrRd")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_group_mean_heatmap.png", dpi=220)
    plt.close()


def plot_pca_scatter(feature_df: pd.DataFrame, df_all: pd.DataFrame, output_dir: Path) -> None:
    if feature_df.empty or feature_df.shape[1] < 2:
        return
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)
    if np.allclose(np.var(X_scaled, axis=0).sum(), 0.0):
        return
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X_scaled)
    vis_df = pd.DataFrame({"pc1": emb[:, 0], "pc2": emb[:, 1], "sample_type": df_all["sample_type"].values})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=vis_df, x="pc1", y="pc2", hue="sample_type", alpha=0.8, s=30)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_scatter_by_sample_type.png", dpi=220)
    plt.close()


def rank_misclassification_features(df_all: pd.DataFrame, feature_df: pd.DataFrame, topk: int) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    y_mis = (df_all["true_label"].astype(int) != df_all["predicted_label"].astype(int)).astype(int).values
    if len(np.unique(y_mis)) < 2:
        importance = feature_df.var(axis=0).sort_values(ascending=False)
        return pd.DataFrame({"feature": importance.index, "importance": importance.values}).head(topk)
    model = ExtraTreesClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(feature_df.values, y_mis)
    imp = pd.Series(model.feature_importances_, index=feature_df.columns).sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "importance": imp.values}).head(topk)


def compute_threshold_metrics(df_all: pd.DataFrame, baseline_threshold: float = 0.98) -> dict:
    y_true = df_all["true_label"].astype(int).to_numpy()
    y_prob = df_all["prediction_probability"].astype(float).to_numpy()

    def _metrics(th: float) -> dict:
        y_pred = (y_prob > th).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        acc = float((tp + tn) / len(y_true)) if len(y_true) else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
        return {"threshold": float(th), "accuracy": acc, "fpr": fpr, "fnr": fnr, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    baseline = _metrics(baseline_threshold)
    best = baseline
    for th in np.linspace(0.5, 0.995, 200):
        cur = _metrics(float(th))
        if cur["accuracy"] + 1e-12 < baseline["accuracy"]:
            continue
        if cur["fpr"] <= baseline["fpr"] * 0.9 + 1e-12 and cur["fnr"] <= baseline["fnr"] * 0.9 + 1e-12:
            if cur["fpr"] + cur["fnr"] < best["fpr"] + best["fnr"] - 1e-12:
                best = cur
    if best is baseline:
        for th in np.linspace(0.5, 0.995, 200):
            cur = _metrics(float(th))
            if cur["accuracy"] + 1e-12 < baseline["accuracy"]:
                continue
            if 0.6 * cur["fpr"] + 0.4 * cur["fnr"] < 0.6 * best["fpr"] + 0.4 * best["fnr"] - 1e-12:
                best = cur
    return {"baseline": baseline, "optimized": best}


def generate_diff_report(output_dir: Path, top_features_df: pd.DataFrame, main_py: Path) -> Path:
    feature_lines = "\n".join([f"- {r.feature}: {r.importance:.6f}" for r in top_features_df.itertuples(index=False)])
    if not feature_lines:
        feature_lines = "- 无可用特征"
    existing = main_py.read_text(encoding="utf-8", errors="ignore")
    hit_scaler = "StandardScaler" in existing
    hit_ohem = "OHEM" in existing or "hard_example" in existing
    report = [
        "# main.py 自动调整建议diff报告",
        "",
        "## Top-K误判关键特征",
        feature_lines,
        "",
        "## 建议改动片段",
        "```diff",
        "- pretrain.main: 训练前后未统一持久化标准化器",
        "+ pretrain.main: 加入StandardScaler拟合/转换并保存到模型目录",
        "- 预测阈值固定为配置常量",
        "+ 在验证集上自动搜索阈值并持久化阈值报告，扫描时加载",
        "- 训练仅按统一样本权重",
        "+ 引入代价敏感权重 + OHEM难例再加权训练",
        "```",
        "",
        "## main.py命中状态",
        f"- StandardScaler关键字存在: {hit_scaler}",
        f"- OHEM关键字存在: {hit_ohem}",
        "",
        "## 建议落地点",
        "- training.train_lightgbm.train_lightgbm_model",
        "- pretrain.main",
        "- scanner.MalwareScanner._predict_malware_from_features",
        "- scanner.MalwareScanner._predict_malware_batch",
    ]
    path = output_dir / "main_adjustment_diff_report.md"
    path.write_text("\n".join(report), encoding="utf-8")
    return path


def render_summary_report(output_dir: Path, per_file_stats: list[dict], top_features_df: pd.DataFrame, diff_report_path: Path, threshold_metrics: dict) -> Path:
    lines = ["# 困难样本分析报告", ""]
    for info in per_file_stats:
        lines.extend(
            [
                f"## {info['file_name']}",
                f"- 样本数量: {info['sample_count']}",
                f"- 真实标签分布: {info['label_distribution_true']}",
                f"- 预测标签分布: {info['label_distribution_predicted']}",
                f"- 概率统计: {info['probability_summary']}",
                f"- 特征维度数: {info['feature_count']}",
                "",
            ]
        )
    lines.append("## Top-K误判关键特征")
    if top_features_df.empty:
        lines.append("- 无")
    else:
        for row in top_features_df.itertuples(index=False):
            lines.append(f"- {row.feature}: {row.importance:.6f}")
    lines.extend(
        [
            "",
            "## 阈值优化前后指标",
            f"- baseline: {threshold_metrics['baseline']}",
            f"- optimized: {threshold_metrics['optimized']}",
            "",
            "## 可视化输出",
            "- feature_boxplot_top_diff.png",
            "- feature_group_mean_heatmap.png",
            "- pca_scatter_by_sample_type.png",
            "",
            f"## 自动diff报告\n- {diff_report_path}",
        ]
    )
    out = output_dir / "analysis_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parent
    default_cluster_dir = project_root / "resources" / "weights_cluster_eval" / "cluster"
    default_output_dir = project_root / "resources" / "weights_cluster_eval" / "eval" / "hard_samples_analysis"
    default_main_py = project_root / "src" / "python" / "kvd_detector" / "main.py"

    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-json", default=None)
    parser.add_argument("--fp-json", default=None)
    parser.add_argument("--fn-json", default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--cluster-dir", default=str(default_cluster_dir))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--main-py", default=str(default_main_py))
    args = parser.parse_args()

    cluster_dir = Path(args.cluster_dir)
    hard_json_path = resolve_json_path(args.hard_json, cluster_dir, ["hard_samples"], "--hard-json")
    fp_json_path = resolve_json_path(args.fp_json, cluster_dir, ["false_positives"], "--fp-json")
    fn_json_path = resolve_json_path(args.fn_json, cluster_dir, ["false_negitives", "false_negatives"], "--fn-json")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_hard = load_samples(hard_json_path, "hard_samples")
    df_fp = load_samples(fp_json_path, "false_positives")
    df_fn = load_samples(fn_json_path, "false_negatives")
    df_all = pd.concat([df_hard, df_fp, df_fn], ignore_index=True)

    per_file_stats = [
        summarize_file(df_hard, output_dir, hard_json_path.name),
        summarize_file(df_fp, output_dir, fp_json_path.name),
        summarize_file(df_fn, output_dir, fn_json_path.name),
    ]
    (output_dir / "per_file_stats.json").write_text(json.dumps(per_file_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    feature_df, feature_cols = build_feature_matrix(df_all)
    plot_boxplots(df_all, feature_cols, output_dir)
    plot_group_heatmap(df_all, feature_cols, output_dir)
    plot_pca_scatter(feature_df, df_all, output_dir)

    top_features_df = rank_misclassification_features(df_all, feature_df, args.top_k)
    top_features_df.to_csv(output_dir / "topk_misclassification_features.csv", index=False, encoding="utf-8-sig")
    threshold_metrics = compute_threshold_metrics(df_all, baseline_threshold=0.98)
    (output_dir / "threshold_metrics.json").write_text(json.dumps(threshold_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    diff_report_path = generate_diff_report(output_dir, top_features_df, Path(args.main_py))
    summary_path = render_summary_report(output_dir, per_file_stats, top_features_df, diff_report_path, threshold_metrics)
    print(f"分析完成: {summary_path}")


if __name__ == "__main__":
    main()
