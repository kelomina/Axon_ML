import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from config.config import HDBSCAN_SAVE_DIR, RESOURCES_DIR

LABEL_TO_ID = {"hard_samples": 0, "false_positives": 1, "false_negatives": 2}
ID_TO_LABEL = {0: "hard_samples", 1: "false_positives", 2: "false_negatives"}

def _latest(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def _norm_name(v: str) -> str:
    return str(v).replace("\\\\", "/").lower()

def load_dataset():
    cluster_dir = Path(HDBSCAN_SAVE_DIR)
    pkl_path = cluster_dir / "extracted_features.pkl"
    hard_path = _latest(cluster_dir, "hard_samples_*.json")
    fp_path = _latest(cluster_dir, "false_positives_*.json")
    fn_path = _latest(cluster_dir, "false_negatives_*.json")
    if not pkl_path.exists() or hard_path is None or fp_path is None or fn_path is None:
        raise RuntimeError("缺少数据文件，无法执行模型试验")
    df = pd.read_pickle(pkl_path)
    feature_cols = [c for c in df.columns if str(c).startswith("feature_")]
    if "filename" not in df.columns or not feature_cols:
        raise RuntimeError("特征文件结构不完整")
    filename_norm = df["filename"].astype(str).map(_norm_name)
    full_map = {}
    base_map = {}
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    for i, n in enumerate(filename_norm.tolist()):
        full_map[n] = i
        b = n.split("/")[-1]
        if b not in base_map:
            base_map[b] = []
        base_map[b].append(i)
    data = [("hard_samples", hard_path), ("false_positives", fp_path), ("false_negatives", fn_path)]
    X = []
    y = []
    for label, path in data:
        records = json.loads(path.read_text(encoding="utf-8"))
        for item in records:
            sid = _norm_name(item.get("sample_id", ""))
            idx = full_map.get(sid)
            if idx is None:
                b = sid.split("/")[-1]
                cand = base_map.get(b, [])
                if len(cand) == 1:
                    idx = cand[0]
            if idx is None:
                continue
            X.append(feat[idx])
            y.append(LABEL_TO_ID[label])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(y) < 50:
        raise RuntimeError("有效样本不足")
    return X, y

def eval_scores(y_true, scores, fn_threshold=0.4, fn_margin=-0.02):
    scores = np.asarray(scores, dtype=np.float32)
    preds = np.argmax(scores, axis=1).astype(np.int64)
    fn_idx = LABEL_TO_ID["false_negatives"]
    other = np.delete(scores, fn_idx, axis=1)
    other_max = np.max(other, axis=1)
    force = (scores[:, fn_idx] >= float(fn_threshold)) & ((scores[:, fn_idx] - other_max) >= float(fn_margin))
    preds[force] = fn_idx
    report = classification_report(y_true, preds, target_names=[ID_TO_LABEL[i] for i in range(3)], output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "recall_hard_samples": float(report["hard_samples"]["recall"]),
        "recall_false_positives": float(report["false_positives"]["recall"]),
        "recall_false_negatives": float(report["false_negatives"]["recall"]),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1, 2]).tolist(),
        "cascade_triggered": int(np.sum(force)),
    }

def run_plan_a(X_train, y_train, X_val, y_val):
    lgb_ovr = OneVsRestClassifier(
        lgb.LGBMClassifier(
            objective="binary",
            n_estimators=260,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
    )
    xgb_ovr = OneVsRestClassifier(
        xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=260,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=4,
        )
    )
    lgb_ovr.fit(X_train, y_train)
    xgb_ovr.fit(X_train, y_train)
    s1 = lgb_ovr.predict_proba(X_val)
    s2 = xgb_ovr.predict_proba(X_val)
    scores = (np.asarray(s1, dtype=np.float32) + np.asarray(s2, dtype=np.float32)) / 2.0
    return eval_scores(y_val, scores, fn_threshold=0.40, fn_margin=-0.02)

def run_plan_b(X_train, y_train, X_val, y_val):
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.18, random_state=42, stratify=y_train)
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    counts[counts <= 0] = 1.0
    class_weights = (np.mean(counts) / counts).tolist()
    catboost_dir = Path(RESOURCES_DIR) / "eval" / "catboost_info"
    catboost_dir.mkdir(parents=True, exist_ok=True)
    base = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=350,
        depth=8,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        class_weights=class_weights,
        train_dir=str(catboost_dir),
    )
    base.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    scores = calibrated.predict_proba(X_val)
    return eval_scores(y_val, scores, fn_threshold=0.40, fn_margin=-0.02)

def run_plan_c(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=120,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    gbdt = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    mlp.fit(X_train_s, y_train)
    gbdt.fit(X_train, y_train)
    s_mlp = mlp.predict_proba(X_val_s)
    s_gbdt = gbdt.predict_proba(X_val)
    scores = 0.45 * np.asarray(s_mlp, dtype=np.float32) + 0.55 * np.asarray(s_gbdt, dtype=np.float32)
    return eval_scores(y_val, scores, fn_threshold=0.38, fn_margin=-0.02)

def run_trials():
    X, y = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    out = {
        "dataset": {
            "total_samples": int(len(y)),
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "class_counts": {
                "hard_samples": int(np.sum(y == 0)),
                "false_positives": int(np.sum(y == 1)),
                "false_negatives": int(np.sum(y == 2)),
            },
        },
        "plan_a_lgb_xgb_ovr_cascade": run_plan_a(X_train, y_train, X_val, y_val),
        "plan_b_catboost_calibrated": run_plan_b(X_train, y_train, X_val, y_val),
        "plan_c_mlp_gbdt_softvote": run_plan_c(X_train, y_train, X_val, y_val),
    }
    eval_dir = Path(RESOURCES_DIR) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "hardcase_plan_abc_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[plan-abc] saved: {out_path}")
    for k in ["plan_a_lgb_xgb_ovr_cascade", "plan_b_catboost_calibrated", "plan_c_mlp_gbdt_softvote"]:
        v = out[k]
        print(f"[plan-abc] {k} acc={v['accuracy']:.4f} macro_f1={v['macro_f1']:.4f} hard_recall={v['recall_hard_samples']:.4f} fp_recall={v['recall_false_positives']:.4f} fn_recall={v['recall_false_negatives']:.4f}")
    return out

def main(args=None):
    return run_trials()
