import os
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from config.config import PACKED_SECTIONS_RATIO_THRESHOLD, PACKER_KEYWORD_HITS_THRESHOLD
from dataset_v2 import load_dataset_v2
from feature_enhancer import get_packed_feature_indices, get_feature_names
from ensemble import train_ensemble, EnsembleModel
from calibration import ProbabilityCalibrator
from threshold import choose_threshold
from hard_negative import update_pool, load_pool, sample_pool
from gating_v2 import train_gating_model, load_gating_model, predict_gating
from settings import MODEL_DIR, REPORT_DIR, CALIBRATION_METHOD, THRESHOLD_FP_WEIGHT, THRESHOLD_FN_WEIGHT, THRESHOLD_MAX_FPR, GATING_THRESHOLD, ONNX_ENABLED, ONNX_MODEL_DIR, ONNX_PROVIDERS, DATA_DIR, TIME_SPLIT_TEST_RATIO, TIME_SPLIT_VAL_RATIO, RANDOM_STATE

def _index_map(files):
    return {f: i for i, f in enumerate(files)}

def _slice_by_files(X, y, files, target_files):
    idx_map = _index_map(files)
    indices = [idx_map[f] for f in target_files if f in idx_map]
    return X[indices], y[indices]

def _file_path(name):
    if name.endswith(".npz"):
        return os.path.join(DATA_DIR, name)
    return os.path.join(DATA_DIR, f"{name}.npz")

def _get_mtime(name):
    path = _file_path(name)
    if os.path.exists(path):
        return os.path.getmtime(path)
    return None

def _split_by_indices(files, labels, train_idx, val_idx, test_idx):
    X_train = [files[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_val = [files[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    X_test = [files[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test

def random_split(files, labels, test_size=TIME_SPLIT_TEST_RATIO, val_size=TIME_SPLIT_VAL_RATIO, random_state=RANDOM_STATE):
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        files, labels, list(range(len(files))), test_size=test_size, random_state=random_state, stratify=labels if len(np.unique(labels)) > 1 else None
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp, test_size=val_size, random_state=random_state, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def time_based_split(files, labels, test_size=TIME_SPLIT_TEST_RATIO, val_size=TIME_SPLIT_VAL_RATIO, random_state=RANDOM_STATE):
    times = [(_get_mtime(f), i) for i, f in enumerate(files)]
    valid = [(t, i) for t, i in times if t is not None]
    if len(valid) < max(10, int(0.3 * len(files))):
        return random_split(files, labels, test_size, val_size, random_state)
    valid.sort(key=lambda x: x[0])
    indices = [i for _, i in valid]
    n_total = len(indices)
    n_test = max(1, int(n_total * test_size))
    n_train_val = n_total - n_test
    n_val = max(1, int(n_train_val * val_size))
    train_indices = indices[:n_train_val - n_val]
    val_indices = indices[n_train_val - n_val:n_train_val]
    test_indices = indices[n_train_val:]
    return _split_by_indices(files, labels, train_indices, val_indices, test_indices)

def _make_gating_labels(X_base):
    packed_idx, packer_idx = get_packed_feature_indices()
    if packed_idx is None or packer_idx is None:
        return np.zeros(X_base.shape[0], dtype=np.int64)
    packed_ratio = X_base[:, packed_idx]
    packer_hits = X_base[:, packer_idx]
    labels = ((packed_ratio > PACKED_SECTIONS_RATIO_THRESHOLD) | (packer_hits > PACKER_KEYWORD_HITS_THRESHOLD)).astype(np.int64)
    return labels

def _train_one_route(X_train, y_train, X_val, y_val, base_dim, total_dim, prefix, files_train=None, hard_negative_set=None):
    if len(np.unique(y_train)) < 2:
        y_train = np.asarray(y_train)
        y_train[0] = 1 - y_train[0]
    ensemble = train_ensemble(X_train, y_train, X_val, y_val, base_dim, total_dim, prefix, files_train=files_train, hard_negative_set=hard_negative_set)
    y_val_prob = ensemble.predict_proba(X_val)
    calibrator = ProbabilityCalibrator(CALIBRATION_METHOD).fit(y_val, y_val_prob)
    y_cal = calibrator.predict(y_val_prob)
    threshold, stats = choose_threshold(y_val, y_cal, fp_weight=THRESHOLD_FP_WEIGHT, fn_weight=THRESHOLD_FN_WEIGHT, max_fpr=THRESHOLD_MAX_FPR)
    with open(os.path.join(MODEL_DIR, f"{prefix}_threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "stats": stats}, f, ensure_ascii=False, indent=2)
    import joblib
    joblib.dump(calibrator, os.path.join(MODEL_DIR, f"{prefix}_calibrator.joblib"))
    return ensemble, calibrator, threshold

def _evaluate_system(X, y, gating_model, base_dim, normal_artifacts, packed_artifacts):
    normal_model, normal_cal, normal_th = normal_artifacts
    packed_model, packed_cal, packed_th = packed_artifacts
    X_base = X[:, :base_dim]
    gating_probs = predict_gating(gating_model, X_base)
    routing = (gating_probs >= GATING_THRESHOLD).astype(np.int64)
    probs = np.zeros(len(X), dtype=np.float32)
    normal_idx = np.where(routing == 0)[0]
    packed_idx = np.where(routing == 1)[0]
    if normal_idx.size > 0:
        p = normal_model.predict_proba(X[normal_idx])
        p = normal_cal.predict(p)
        probs[normal_idx] = p
    if packed_idx.size > 0:
        p = packed_model.predict_proba(X[packed_idx])
        p = packed_cal.predict(p)
        probs[packed_idx] = p
    thresholds = np.where(routing == 0, normal_th, packed_th)
    preds = (probs >= thresholds).astype(np.int64)
    acc = accuracy_score(y, preds)
    pre = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    fp = int(np.sum((y == 0) & (preds == 1)))
    tn = int(np.sum((y == 0) & (preds == 0)))
    fpr = fp / max(1, fp + tn)
    return {"accuracy": float(acc), "precision": float(pre), "recall": float(rec), "fpr": float(fpr)}

def train_pipeline(fast_dev_run=False, use_cache=True):
    X, y, files, base_dim = load_dataset_v2(use_cache=use_cache, fast_dev_run=fast_dev_run)
    os.makedirs(MODEL_DIR, exist_ok=True)
    feature_names = get_feature_names()
    if len(feature_names) != X.shape[1]:
        feature_names = feature_names[:X.shape[1]]
        while len(feature_names) < X.shape[1]:
            feature_names.append(f"feature_{len(feature_names)}")
    with open(os.path.join(MODEL_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    train_files, y_train_list, val_files, y_val_list, test_files, y_test_list = time_based_split(files, y)
    X_train, y_train = _slice_by_files(X, y, files, train_files)
    X_val, y_val = _slice_by_files(X, y, files, val_files)
    X_test, y_test = _slice_by_files(X, y, files, test_files)
    X_base_train = X_train[:, :base_dim]
    X_base_val = X_val[:, :base_dim]
    gating_labels_train = _make_gating_labels(X_base_train)
    gating_labels_val = _make_gating_labels(X_base_val)
    gating_path = os.path.join(MODEL_DIR, "gating_model.pt")
    train_gating_model(X_base_train, gating_labels_train, X_base_val, gating_labels_val, gating_path, base_dim)
    gating_model = load_gating_model(gating_path, base_dim)
    routing_train = (predict_gating(gating_model, X_base_train) >= GATING_THRESHOLD).astype(np.int64)
    routing_val = (predict_gating(gating_model, X_base_val) >= GATING_THRESHOLD).astype(np.int64)
    normal_train_idx = np.where(routing_train == 0)[0]
    packed_train_idx = np.where(routing_train == 1)[0]
    normal_val_idx = np.where(routing_val == 0)[0]
    packed_val_idx = np.where(routing_val == 1)[0]
    if normal_train_idx.size == 0:
        normal_train_idx = np.arange(len(X_train))
        normal_val_idx = np.arange(len(X_val))
    if packed_train_idx.size == 0:
        packed_train_idx = np.arange(len(X_train))
        packed_val_idx = np.arange(len(X_val))
    pool = load_pool()
    hard_negative_set = set(sample_pool(pool))
    normal_files_train = [train_files[i] for i in normal_train_idx]
    packed_files_train = [train_files[i] for i in packed_train_idx]
    normal_artifacts = _train_one_route(X_train[normal_train_idx], y_train[normal_train_idx], X_val[normal_val_idx], y_val[normal_val_idx], base_dim, X.shape[1], "normal", files_train=normal_files_train, hard_negative_set=hard_negative_set)
    packed_artifacts = _train_one_route(X_train[packed_train_idx], y_train[packed_train_idx], X_val[packed_val_idx], y_val[packed_val_idx], base_dim, X.shape[1], "packed", files_train=packed_files_train, hard_negative_set=hard_negative_set)
    normal_pred = normal_artifacts[0].predict_proba(X_val[normal_val_idx])
    normal_fp = [val_files[i] for i, p in zip(normal_val_idx, normal_pred) if y_val[i] == 0 and p >= normal_artifacts[2]]
    packed_pred = packed_artifacts[0].predict_proba(X_val[packed_val_idx])
    packed_fp = [val_files[i] for i, p in zip(packed_val_idx, packed_pred) if y_val[i] == 0 and p >= packed_artifacts[2]]
    update_pool(normal_fp + packed_fp)
    metrics = _evaluate_system(X_test, y_test, gating_model, base_dim, normal_artifacts, packed_artifacts)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

def load_system():
    normal_model = EnsembleModel.load(MODEL_DIR, "normal")
    packed_model = EnsembleModel.load(MODEL_DIR, "packed")
    import joblib
    normal_cal = joblib.load(os.path.join(MODEL_DIR, "normal_calibrator.joblib"))
    packed_cal = joblib.load(os.path.join(MODEL_DIR, "packed_calibrator.joblib"))
    with open(os.path.join(MODEL_DIR, "normal_threshold.json"), "r", encoding="utf-8") as f:
        normal_th = json.load(f)["threshold"]
    with open(os.path.join(MODEL_DIR, "packed_threshold.json"), "r", encoding="utf-8") as f:
        packed_th = json.load(f)["threshold"]
    gating_path = os.path.join(MODEL_DIR, "gating_model.pt")
    gating_model = load_gating_model(gating_path, normal_model.base_dim)
    if ONNX_ENABLED:
        try:
            from onnx_backend import OnnxPredictor
            features_path = os.path.join(MODEL_DIR, "features.json")
            with open(features_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
            normal_onnx = os.path.join(ONNX_MODEL_DIR, "normal.onnx")
            packed_onnx = os.path.join(ONNX_MODEL_DIR, "packed.onnx")
            n_pred = OnnxPredictor(normal_onnx, feature_names, providers=ONNX_PROVIDERS)
            p_pred = OnnxPredictor(packed_onnx, feature_names, providers=ONNX_PROVIDERS)
            if n_pred.available():
                normal_model = n_pred
            if p_pred.available():
                packed_model = p_pred
        except Exception:
            pass
    return gating_model, (normal_model, normal_cal, normal_th), (packed_model, packed_cal, packed_th)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    if args.train:
        metrics = train_pipeline(fast_dev_run=args.fast_dev_run, use_cache=args.use_cache)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
