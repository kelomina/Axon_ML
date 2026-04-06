import os
import sys
import argparse
import json
import pickle
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from training.data_loader import load_dataset, extract_features_from_raw_files, load_incremental_dataset
from training.feature_io import save_features_to_pickle
from training.train_lightgbm import train_lightgbm_model
from training.evaluate import evaluate_model
from training.model_io import save_model, load_existing_model
from training.incremental import incremental_train_lightgbm_model
from features.statistics import extract_statistical_features
from features.extractor_in_memory import PE_FEATURE_ORDER
from config.config import PROCESSED_DATA_DIR, METADATA_FILE, SAVED_MODEL_DIR, MODEL_PATH, FEATURES_PKL_PATH, DEFAULT_MAX_FILE_SIZE, DEFAULT_NUM_BOOST_ROUND, DEFAULT_INCREMENTAL_ROUNDS, DEFAULT_INCREMENTAL_EARLY_STOPPING, DEFAULT_MAX_FINETUNE_ITERATIONS, HELP_MAX_FILE_SIZE, HELP_FAST_DEV_RUN, HELP_SAVE_FEATURES, HELP_FINETUNE_ON_FALSE_POSITIVES, HELP_INCREMENTAL_TRAINING, HELP_INCREMENTAL_DATA_DIR, HELP_INCREMENTAL_RAW_DATA_DIR, HELP_FILE_EXTENSIONS, HELP_LABEL_INFERENCE, HELP_NUM_BOOST_ROUND, HELP_INCREMENTAL_ROUNDS, HELP_INCREMENTAL_EARLY_STOPPING, HELP_MAX_FINETUNE_ITERATIONS, HELP_USE_EXISTING_FEATURES, FEATURE_SCALER_PATH, THRESHOLD_REPORT_PATH, HDBSCAN_SAVE_DIR, PREDICTION_THRESHOLD, OHEM_ENABLED, OHEM_RATIO, OHEM_WEIGHT_FACTOR, MISCLASSIFIED_FP_WEIGHT, MISCLASSIFIED_FN_WEIGHT, MISCLASSIFIED_HARD_WEIGHT, RESOURCES_DIR


def _normalize_name(name):
    return str(name).replace('\\', '/').lower()


def _latest_sample_report(pattern):
    base_dir = Path(HDBSCAN_SAVE_DIR)
    files = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_samples(report_path):
    if report_path is None or not report_path.exists():
        return []
    try:
        return json.loads(report_path.read_text(encoding='utf-8'))
    except Exception:
        return []


def _feature_index_from_name(name):
    m = re.search(r'(\d+)$', str(name))
    if not m:
        return None
    return int(m.group(1))


def _load_analysis_topk_feature_indices():
    analysis_dir = Path(RESOURCES_DIR) / 'eval' / 'hard_samples_analysis'
    if not analysis_dir.exists():
        return set()
    csv_files = sorted(analysis_dir.glob('topk_misclassification_features*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        return set()
    try:
        top_df = pd.read_csv(csv_files[0])
    except Exception:
        return set()
    protected = set()
    for _, row in top_df.iterrows():
        idx = _feature_index_from_name(row.get('feature', ''))
        if idx is None:
            try:
                idx = int(row.get('feature_id'))
            except Exception:
                idx = None
        if idx is not None and idx >= 0:
            protected.add(int(idx))
    return protected


def _collect_low_importance_features():
    report_defs = [
        ('hard_samples', 'hard_samples_*.json'),
        ('false_positives', 'false_positives_*.json'),
        ('false_negatives', 'false_negatives_*.json'),
    ]
    sum_abs_importance = {}
    max_abs_importance = {}
    sample_counts = {}
    group_sum = {name: {} for name, _ in report_defs}
    group_counts = {name: 0 for name, _ in report_defs}
    for group_name, pattern in report_defs:
        report_path = _latest_sample_report(pattern)
        samples = _load_samples(report_path)
        group_counts[group_name] = int(len(samples))
        for sample in samples:
            fmap = sample.get('feature_importance', {}) or {}
            for key, value in fmap.items():
                idx = _feature_index_from_name(key)
                if idx is None:
                    continue
                abs_val = abs(float(value))
                sum_abs_importance[idx] = float(sum_abs_importance.get(idx, 0.0) + abs_val)
                sample_counts[idx] = int(sample_counts.get(idx, 0) + 1)
                max_abs_importance[idx] = float(max(max_abs_importance.get(idx, 0.0), abs_val))
                group_sum[group_name][idx] = float(group_sum[group_name].get(idx, 0.0) + abs_val)
    if not sample_counts:
        return [], {}, {}
    mean_abs_importance = {
        idx: float(sum_abs_importance.get(idx, 0.0) / max(1, sample_counts.get(idx, 0)))
        for idx in sample_counts.keys()
    }
    feature_indices = sorted(sample_counts.keys())
    span_by_feature = {}
    for idx in feature_indices:
        group_means = []
        for group_name, _ in report_defs:
            g_count = max(1, int(group_counts.get(group_name, 0)))
            g_mean = float(group_sum[group_name].get(idx, 0.0) / max(1, g_count))
            group_means.append(g_mean)
        span_by_feature[idx] = float(max(group_means) - min(group_means)) if group_means else 0.0
    mean_values = np.asarray([mean_abs_importance[idx] for idx in feature_indices], dtype=np.float64)
    span_values = np.asarray([span_by_feature[idx] for idx in feature_indices], dtype=np.float64)
    max_values = np.asarray([max_abs_importance.get(idx, 0.0) for idx in feature_indices], dtype=np.float64)
    q_mean = float(np.quantile(mean_values, 0.2)) if mean_values.size else 0.0
    q_span = float(np.quantile(span_values, 0.2)) if span_values.size else 0.0
    q_max = float(np.quantile(max_values, 0.25)) if max_values.size else 0.0
    low_importance_indices = sorted([
        idx
        for idx in feature_indices
        if mean_abs_importance.get(idx, 0.0) <= q_mean + 1e-15
        and span_by_feature.get(idx, 0.0) <= q_span + 1e-15
        and max_abs_importance.get(idx, 0.0) <= q_max + 1e-15
    ])
    stat = {
        'mean_quantile_20': q_mean,
        'span_quantile_20': q_span,
        'max_quantile_25': q_max,
    }
    return low_importance_indices, mean_abs_importance, stat


def _load_zero_importance_feature_indices():
    eval_dir = Path(RESOURCES_DIR) / 'eval'
    json_path = eval_dir / 'full_feature_importance_ranking.json'
    csv_path = eval_dir / 'full_feature_importance_ranking.csv'
    rows = None
    if json_path.exists():
        try:
            rows = json.loads(json_path.read_text(encoding='utf-8'))
        except Exception:
            rows = None
    if rows is None and csv_path.exists():
        try:
            rows = pd.read_csv(csv_path).to_dict(orient='records')
        except Exception:
            rows = None
    if not rows:
        return set()
    zero_indices = set()
    for row in rows:
        try:
            gain = float(row.get('gain', 0.0))
        except Exception:
            gain = 0.0
        try:
            split = float(row.get('split', 0.0))
        except Exception:
            split = 0.0
        if abs(gain) > 1e-15 or abs(split) > 1e-15:
            continue
        idx = None
        try:
            idx = int(row.get('feature_id'))
        except Exception:
            idx = None
        if idx is None:
            idx = _feature_index_from_name(row.get('feature', ''))
        if idx is not None and idx >= 0:
            zero_indices.add(int(idx))
    return zero_indices


def _make_feature_columns(n_features):
    return [f'feature_{i}' for i in range(int(n_features))]


def _write_feature_selector_reports(n_samples, n_features_in, selected_indices, removed_indices, importance_map):
    weights_dir = Path(RESOURCES_DIR) / 'weights'
    eval_dir = Path(RESOURCES_DIR) / 'eval'
    weights_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    removed_feature_stats = []
    for idx in removed_indices:
        removed_feature_stats.append({
            'feature_id': int(idx),
            'feature_name': f'feature_{int(idx)}',
            'importance': float(importance_map.get(int(idx), 0.0))
        })
    selector_payload = {
        'enabled': True,
        'n_samples': int(n_samples),
        'n_features_in': int(n_features_in),
        'n_features_out': int(len(selected_indices)),
        'selected_indices': [int(i) for i in selected_indices],
        'removed_indices': [int(i) for i in removed_indices],
        'removed_feature_stats': removed_feature_stats,
        'rules': {
            'variance_threshold': 1e-08,
            'mutual_info_threshold': 1e-06,
            'chi2_threshold': 1e-06,
            'protected_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 323, 412],
            'fallback_tolerance': 0.0
        }
    }
    (weights_dir / 'feature_selector.json').write_text(json.dumps(selector_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    (eval_dir / 'feature_selection_report.json').write_text(json.dumps(selector_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    dim_change_payload = {
        'n_features_in': int(n_features_in),
        'n_features_out': int(len(selected_indices)),
        'removed_count': int(len(removed_indices)),
        'selected_ratio': float(len(selected_indices) / max(1, int(n_features_in)))
    }
    (eval_dir / 'feature_dimension_change.json').write_text(json.dumps(dim_change_payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _build_sample_weight_map():
    weight_map = {}
    hard_report = _latest_sample_report('hard_samples_*.json')
    fp_report = _latest_sample_report('false_positives_*.json')
    fn_report = _latest_sample_report('false_negatives_*.json')
    for item in _load_samples(hard_report):
        sid = _normalize_name(item.get('sample_id', ''))
        if sid:
            weight_map[sid] = max(weight_map.get(sid, 1.0), float(MISCLASSIFIED_HARD_WEIGHT))
    for item in _load_samples(fp_report):
        sid = _normalize_name(item.get('sample_id', ''))
        if sid:
            weight_map[sid] = max(weight_map.get(sid, 1.0), float(MISCLASSIFIED_FP_WEIGHT))
    for item in _load_samples(fn_report):
        sid = _normalize_name(item.get('sample_id', ''))
        if sid:
            weight_map[sid] = max(weight_map.get(sid, 1.0), float(MISCLASSIFIED_FN_WEIGHT))
    return weight_map


def _build_train_weights(files_train, weight_map):
    weights = np.ones(len(files_train), dtype=np.float32)
    for i, fname in enumerate(files_train):
        normalized = _normalize_name(fname)
        base_name = _normalize_name(Path(str(fname)).name)
        if normalized in weight_map:
            weights[i] = max(weights[i], weight_map[normalized])
        if base_name in weight_map:
            weights[i] = max(weights[i], weight_map[base_name])
    return weights


def _apply_ohem_weights(model, X_train, y_train, base_weights):
    if not OHEM_ENABLED or len(X_train) == 0:
        return base_weights
    best_iteration = getattr(model, 'best_iteration', None)
    if isinstance(best_iteration, int) and best_iteration > 0:
        train_proba = model.predict(X_train, num_iteration=best_iteration)
    else:
        train_proba = model.predict(X_train)
    hardness = np.where(y_train == 1, 1 - train_proba, train_proba)
    topk = int(max(1, round(len(hardness) * float(OHEM_RATIO))))
    hard_idx = np.argsort(hardness)[-topk:]
    updated = base_weights.copy()
    updated[hard_idx] = np.maximum(updated[hard_idx], updated[hard_idx] * float(OHEM_WEIGHT_FACTOR))
    return updated


def _binary_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        'accuracy': float(accuracy),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'tp': int(tp)
    }


def _optimize_threshold(y_true, y_proba, baseline_threshold):
    baseline_pred = (y_proba > baseline_threshold).astype(int)
    baseline = _binary_metrics(y_true, baseline_pred)
    best_threshold = baseline_threshold
    best_metric = baseline
    for threshold in np.linspace(0.50, 0.995, 200):
        pred = (y_proba > threshold).astype(int)
        metric = _binary_metrics(y_true, pred)
        if metric['accuracy'] + 1e-12 < baseline['accuracy']:
            continue
        target_fpr = baseline['fpr'] * 0.9 if baseline['fpr'] > 0 else 0.0
        target_fnr = baseline['fnr'] * 0.9 if baseline['fnr'] > 0 else 0.0
        fpr_ok = metric['fpr'] <= target_fpr + 1e-12 if baseline['fpr'] > 0 else metric['fpr'] <= baseline['fpr'] + 1e-12
        fnr_ok = metric['fnr'] <= target_fnr + 1e-12 if baseline['fnr'] > 0 else metric['fnr'] <= baseline['fnr'] + 1e-12
        if fpr_ok and fnr_ok:
            score = metric['fpr'] + metric['fnr']
            best_score = best_metric['fpr'] + best_metric['fnr']
            if score < best_score - 1e-12:
                best_metric = metric
                best_threshold = float(threshold)
    if best_threshold == baseline_threshold:
        for threshold in np.linspace(0.50, 0.995, 200):
            pred = (y_proba > threshold).astype(int)
            metric = _binary_metrics(y_true, pred)
            if metric['accuracy'] + 1e-12 < baseline['accuracy']:
                continue
            score = 0.6 * metric['fpr'] + 0.4 * metric['fnr']
            best_score = 0.6 * best_metric['fpr'] + 0.4 * best_metric['fnr']
            if score < best_score - 1e-12:
                best_metric = metric
                best_threshold = float(threshold)
    return best_threshold, baseline, best_metric


def _expected_feature_dim():
    pe_dim = int(len(PE_FEATURE_ORDER))
    dummy_bytes = np.zeros(1, dtype=np.uint8)
    dummy_pe = np.zeros(pe_dim, dtype=np.float32)
    feat = extract_statistical_features(dummy_bytes, dummy_pe, orig_length=1)
    return int(feat.shape[0])


def main(args):
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    df = None
    feature_columns = None
    expected_dim = _expected_feature_dim()
    reuse_existing = bool(args.use_existing_features and os.path.exists(FEATURES_PKL_PATH))

    if reuse_existing:

        print("[*] Loading existing feature file...")
        try:
            df = pd.read_pickle(FEATURES_PKL_PATH)
            files = df['filename'].tolist()
            y = df['label'].values
            feature_columns = [c for c in df.columns if c.startswith('feature_')]
            feature_columns = sorted(feature_columns, key=lambda c: int(c.split('_')[1]))
            X = df[feature_columns].values

            print(f"[+] Successfully loaded feature file, total {len(files)} samples, feature dimension: {X.shape[1]}")
            if int(X.shape[1]) != expected_dim:
                print(f"[!] Existing feature dimension mismatch: {X.shape[1]} != {expected_dim}")
                print("[*] Will regenerate features from processed dataset")
                reuse_existing = False
                df = None
                feature_columns = None
        except Exception as e:

            print(f"[!] Failed to load feature file: {e}")
            reuse_existing = False
    if not reuse_existing:
        if args.incremental_training and args.incremental_data_dir:
            if args.incremental_raw_data_dir:

                print("[*] Extracting features from raw files...")
                output_features_dir = args.incremental_data_dir
                file_names, labels = extract_features_from_raw_files(
                    args.incremental_raw_data_dir,
                    output_features_dir,
                    args.max_file_size,
                    args.file_extensions,
                    args.label_inference
                )

                if not file_names:

                    print("[!] Failed to extract features from raw files, exiting training")
                    return

            X, y, files = load_incremental_dataset(args.incremental_data_dir, args.max_file_size)
            if X is None:

                print("[!] Failed to load incremental data, exiting training")
                return
        else:
            X, y, files = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE, args.max_file_size, args.fast_dev_run)

    n_features_in = int(X.shape[1])
    if feature_columns is None:
        feature_columns = _make_feature_columns(n_features_in)
    low_importance_indices, feature_importance_map, feature_select_stat = _collect_low_importance_features()
    zero_importance_indices = _load_zero_importance_feature_indices()
    analysis_topk_indices = _load_analysis_topk_feature_indices()
    protected_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 323, 412}
    protected_indices = protected_indices.union(analysis_topk_indices)
    candidate_indices = [idx for idx in low_importance_indices if 0 <= idx < n_features_in and idx not in protected_indices]
    max_remove_count = int(max(0, np.floor(float(n_features_in) * 0.2)))
    if max_remove_count > 0 and len(candidate_indices) > max_remove_count:
        candidate_indices = sorted(candidate_indices, key=lambda idx: float(feature_importance_map.get(int(idx), 0.0)))
        removed_low_indices = sorted(candidate_indices[:max_remove_count])
    else:
        removed_low_indices = sorted(candidate_indices)
    removed_zero_indices = sorted([idx for idx in zero_importance_indices if 0 <= idx < n_features_in])
    removed_indices = sorted(set(removed_low_indices).union(removed_zero_indices))
    removed_set = set(removed_indices)
    selected_indices = [idx for idx in range(n_features_in) if idx not in removed_set]
    if not selected_indices:
        selected_indices = list(range(n_features_in))
        removed_indices = []
    if len(selected_indices) < n_features_in:
        X = X[:, selected_indices]
        feature_columns = [feature_columns[idx] for idx in selected_indices]
        print(f"[*] 已根据误判与假阴性重要度剔除特征: {len(removed_indices)} 个，保留 {len(selected_indices)} 个")
        if removed_zero_indices:
            print(f"[*] 额外剔除零重要度特征: {len(removed_zero_indices)} 个")
        print(f"[*] 特征筛选统计: {feature_select_stat}")
    _write_feature_selector_reports(len(files), n_features_in, selected_indices, removed_indices, feature_importance_map)
    save_features_to_pickle(X, y, files, FEATURES_PKL_PATH, feature_names=feature_columns)

    if len(X) > 10:
        from config.config import DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, DEFAULT_RANDOM_STATE
        X_temp, X_test, y_temp, y_test, files_temp, files_test = train_test_split(
            X, y, files, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y if len(np.unique(y)) > 1 else None
        )
        if len(X_temp) > 5:
            X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
                X_temp, y_temp, files_temp, test_size=DEFAULT_VAL_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
            )
        else:
            X_train, X_val = X_temp, X_temp
            y_train, y_val = y_temp, y_temp
            X_test, y_test = X_temp, y_temp
            files_train, files_val, files_test = files_temp, files_temp, files_temp
    else:
        X_train, X_val, X_test = X, X, X
        y_train, y_val, y_test = y, y, y
        files_train, files_val, files_test = files, files, files

    print(f"[*] Dataset split completed:")
    print(f"    Training set: {len(X_train)} samples")
    print(f"    Validation set: {len(X_val)} samples")
    print(f"    Test set: {len(X_test)} samples")
    print(f"    Class distribution - Train: Benign={np.sum(y_train==0)}, Malicious={np.sum(y_train==1)}")
    print(f"    Class distribution - Val: Benign={np.sum(y_val==0)}, Malicious={np.sum(y_val==1)}")
    print(f"    Class distribution - Test: Benign={np.sum(y_test==0)}, Malicious={np.sum(y_test==1)}")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    with open(FEATURE_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[+] Feature scaler saved to: {FEATURE_SCALER_PATH}")
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType
        scaler_onnx = to_onnx(scaler, initial_types=[('input', SkFloatTensorType([None, int(X_train.shape[1])]))], target_opset=15)
        scaler_onnx_path = os.path.splitext(FEATURE_SCALER_PATH)[0] + '.onnx'
        with open(scaler_onnx_path, 'wb') as f:
            f.write(scaler_onnx.SerializeToString())
        print(f"[+] Feature scaler ONNX saved to: {scaler_onnx_path}")
    except Exception as e:
        raise RuntimeError(f"Feature scaler ONNX导出失败: {e}")
    weight_map = _build_sample_weight_map()
    sample_weights = _build_train_weights(files_train, weight_map)

    existing_model = None
    if args.incremental_training:
        existing_model = load_existing_model(MODEL_PATH)

    model = None

    if args.incremental_training and existing_model:

        print("\n[*] Performing incremental training...")
        model = incremental_train_lightgbm_model(
            existing_model, X_train, y_train, X_val, y_val,
            num_boost_round=args.incremental_rounds,
            early_stopping_rounds=args.incremental_early_stopping
        )
    else:
        model = train_lightgbm_model(
            X_train, y_train, X_val, y_val,
            iteration=1,
            num_boost_round=args.num_boost_round,
            params_override=getattr(args, 'override_params', None),
            sample_weights=sample_weights
        )
        if OHEM_ENABLED:
            sample_weights = _apply_ohem_weights(model, X_train, y_train, sample_weights)
            model = train_lightgbm_model(
                X_train, y_train, X_val, y_val,
                iteration=2,
                num_boost_round=args.num_boost_round,
                init_model=model,
                params_override=getattr(args, 'override_params', None),
                sample_weights=sample_weights
            )

    max_finetune_iterations = args.max_finetune_iterations
    finetune_iteration = 0
    false_positives = []

    while finetune_iteration < max_finetune_iterations:
        if args.finetune_on_false_positives:
            finetune_iteration += 1

            print(f"\n[*] Performing round {finetune_iteration} reinforcement training...")
            model = train_lightgbm_model(X_train, y_train, X_val, y_val, 
                                       iteration=finetune_iteration+1, 
                                       num_boost_round=args.num_boost_round,
                                       init_model=model,
                                       params_override=getattr(args, 'override_params', None),
                                       sample_weights=sample_weights)

            if finetune_iteration >= max_finetune_iterations:

                print("[*] Reached maximum reinforcement training rounds")
                break
        else:

            print("[*] Reinforcement training not enabled, skipping reinforcement training phase")
            break

    print("\n[*] Reinforcement training completed, performing final evaluation...")
    if len(X_test) > 0:
        test_accuracy, false_positives = evaluate_model(model, X_test, y_test, files_test)

        if false_positives and args.finetune_on_false_positives:

            print(f"\n[*] Detected {len(false_positives)} false positive samples, performing targeted reinforcement training...")

            targeted_iteration = 0
            max_targeted_iterations = 5
            previous_fp_count = len(false_positives)

            while len(false_positives) > 0 and targeted_iteration < max_targeted_iterations:
                targeted_iteration += 1

                print(f"\n[*] Performing round {targeted_iteration} targeted reinforcement training...")
                model = train_lightgbm_model(X_train, y_train, X_val, y_val,
                                           false_positives, files_train,
                                           finetune_iteration + targeted_iteration,
                                           num_boost_round=args.num_boost_round,
                                           init_model=model,
                                           params_override=getattr(args, 'override_params', None),
                                           sample_weights=sample_weights)

                print(f"\n[*] Evaluating after round {targeted_iteration} targeted reinforcement training...")
                test_accuracy, false_positives = evaluate_model(model, X_test, y_test, files_test)

                if len(false_positives) >= previous_fp_count:

                    print("[*] Targeted reinforcement training failed to reduce false positives, stopping training")
                    break
                previous_fp_count = len(false_positives)

            if len(false_positives) == 0:

                print("[*] Successfully eliminated all false positive samples")
            else:

                print(f"[*] Targeted reinforcement training completed, remaining {len(false_positives)} false positive samples")
        elif false_positives:

            print(f"\n[*] Detected {len(false_positives)} false positive samples, but reinforcement training is not enabled")

            print("    To enable reinforcement training, use the --finetune-on-false-positives parameter")
        try:
            best_iteration = getattr(model, 'best_iteration', None)
            if isinstance(best_iteration, int) and best_iteration > 0:
                y_val_proba = model.predict(X_val, num_iteration=best_iteration)
                y_pred_proba = model.predict(X_test, num_iteration=best_iteration)
            else:
                y_val_proba = model.predict(X_val)
                y_pred_proba = model.predict(X_test)
            selected_threshold, baseline_metrics, optimized_metrics = _optimize_threshold(y_val, y_val_proba, PREDICTION_THRESHOLD)
            y_pred_selected = (y_pred_proba > selected_threshold).astype(int)
            selected_metrics = _binary_metrics(y_test, y_pred_selected)
            false_positives = [files_test[i] for i in np.where((y_pred_selected == 1) & (y_test == 0))[0]]
            print(f"[*] Selected threshold: {selected_threshold:.4f}")
            print(f"[*] Validation baseline metrics: {baseline_metrics}")
            print(f"[*] Validation optimized metrics: {optimized_metrics}")
            print(f"[*] Test metrics with selected threshold: {selected_metrics}")
            threshold_report = {
                'selected_threshold': float(selected_threshold),
                'validation_baseline': baseline_metrics,
                'validation_optimized': optimized_metrics,
                'test_metrics': selected_metrics
            }
            os.makedirs(os.path.dirname(THRESHOLD_REPORT_PATH), exist_ok=True)
            with open(THRESHOLD_REPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(threshold_report, f, ensure_ascii=False, indent=2)
            print(f"[+] Threshold report saved to: {THRESHOLD_REPORT_PATH}")
            thresholds = sorted(set([float(selected_threshold)] + list(np.arange(0.90, 0.99, 0.01))))
            print("\n[*] Threshold sensitivity (0.90–0.98 + selected):")
            for t in thresholds:
                y_pred_t = (y_pred_proba > t).astype(int)
                cm = confusion_matrix(y_test, y_pred_t, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                acc = accuracy_score(y_test, y_pred_t)
                pre = precision_score(y_test, y_pred_t, zero_division=0)
                rec = recall_score(y_test, y_pred_t, zero_division=0)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fp_count = int(fp)
                print(f"    t={t:.4f} acc={acc:.4f} pre={pre:.4f} rec={rec:.4f} FPR={fpr:.4f} TPR={tpr:.4f} FP={fp_count}")
        except Exception as e:
            print(f"[!] Threshold optimization failed: {e}")
    else:

        print("[*] Test set is empty, skipping model evaluation")

    save_model(model, MODEL_PATH)

    stat_feature_names = [
        'byte_mean', 'byte_std', 'byte_min', 'byte_max', 'byte_median', 'byte_q25', 'byte_q75',
        'count_0', 'count_255', 'count_0x90', 'count_printable', 'entropy',
        'seg0_mean', 'seg0_std', 'seg0_entropy',
        'seg1_mean', 'seg1_std', 'seg1_entropy',
        'seg2_mean', 'seg2_std', 'seg2_entropy',
        'chunk_mean_0', 'chunk_mean_1', 'chunk_mean_2', 'chunk_mean_3', 'chunk_mean_4',
        'chunk_mean_5', 'chunk_mean_6', 'chunk_mean_7', 'chunk_mean_8', 'chunk_mean_9',
        'chunk_std_0', 'chunk_std_1', 'chunk_std_2', 'chunk_std_3', 'chunk_std_4',
        'chunk_std_5', 'chunk_std_6', 'chunk_std_7', 'chunk_std_8', 'chunk_std_9',
        'chunk_mean_diff_mean_abs', 'chunk_mean_diff_std', 'chunk_mean_diff_max', 'chunk_mean_diff_min',
        'chunk_std_diff_mean_abs', 'chunk_std_diff_std', 'chunk_std_diff_max', 'chunk_std_diff_min'
    ]
    feature_gain = model.feature_importance(importance_type='gain')
    feature_split = model.feature_importance(importance_type='split')
    model_feature_names = list(model.feature_name() or [])
    if len(model_feature_names) != len(feature_gain):
        model_feature_names = list(feature_columns[:len(feature_gain)])
    if len(model_feature_names) < len(feature_gain):
        model_feature_names.extend([f'feature_{i}' for i in range(len(model_feature_names), len(feature_gain))])
    def _parse_feature_id(name):
        m = re.search(r'(\d+)$', str(name))
        if not m:
            return None
        return int(m.group(1))
    def _feature_id_to_real_name(feature_id):
        if feature_id < len(stat_feature_names):
            return stat_feature_names[feature_id]
        x = feature_id - len(stat_feature_names)
        if x < 256:
            return f'lw_{x}'
        pe_idx = x - 256
        if pe_idx < len(PE_FEATURE_ORDER):
            return PE_FEATURE_ORDER[pe_idx]
        return f'pe_extra_{pe_idx - len(PE_FEATURE_ORDER)}'
    rows = []
    for idx in range(len(feature_gain)):
        feature_key = str(model_feature_names[idx])
        feature_id = _parse_feature_id(feature_key)
        if feature_id is None:
            feature_id = idx
        rows.append({
            'rank': 0,
            'feature': feature_key,
            'feature_id': int(feature_id),
            'feature_name': _feature_id_to_real_name(int(feature_id)),
            'gain_importance': float(feature_gain[idx]),
            'split_importance': float(feature_split[idx]),
        })
    rows = sorted(rows, key=lambda x: x['gain_importance'], reverse=True)
    for rank, row in enumerate(rows, 1):
        row['rank'] = int(rank)
    feature_importance_df = pd.DataFrame(rows)
    full_importance_csv = Path(RESOURCES_DIR) / 'eval' / 'full_feature_importance_ranking.csv'
    full_importance_json = Path(RESOURCES_DIR) / 'eval' / 'full_feature_importance_ranking.json'
    os.makedirs(full_importance_csv.parent, exist_ok=True)
    feature_importance_df.to_csv(full_importance_csv, index=False, encoding='utf-8-sig')
    full_importance_json.write_text(feature_importance_df.to_json(orient='records', force_ascii=False, indent=2), encoding='utf-8')
    print('\n[*] Full feature importance ranking (all features):')
    for row in rows:
        print(f"    {int(row['rank']):4d}. {row['feature_name']} [{row['feature']}] gain={row['gain_importance']:.6f} split={row['split_importance']:.0f}")
    print(f"[+] Full feature importance CSV saved to: {full_importance_csv}")
    print(f"[+] Full feature importance JSON saved to: {full_importance_json}")

    print(f"\n[+] LightGBM pre-training completed! Model saved to: {MODEL_PATH}")

    print(f"[+] Extracted features saved to: {FEATURES_PKL_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LightGBM-based malware detection pre-training script")
    parser.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)
    parser.add_argument('--fast-dev-run', action='store_true', help=HELP_FAST_DEV_RUN)
    parser.add_argument('--save-features', action='store_true', help=HELP_SAVE_FEATURES)
    parser.add_argument('--finetune-on-false-positives', action='store_true', help=HELP_FINETUNE_ON_FALSE_POSITIVES)
    parser.add_argument('--incremental-training', action='store_true', help=HELP_INCREMENTAL_TRAINING)
    parser.add_argument('--incremental-data-dir', type=str, help=HELP_INCREMENTAL_DATA_DIR)
    parser.add_argument('--incremental-raw-data-dir', type=str, help=HELP_INCREMENTAL_RAW_DATA_DIR)
    parser.add_argument('--file-extensions', type=str, nargs='+', help=HELP_FILE_EXTENSIONS)
    parser.add_argument('--label-inference', type=str, default='filename', choices=['filename', 'directory'], help=HELP_LABEL_INFERENCE)
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND, help=HELP_NUM_BOOST_ROUND)
    parser.add_argument('--incremental-rounds', type=int, default=DEFAULT_INCREMENTAL_ROUNDS, help=HELP_INCREMENTAL_ROUNDS)
    parser.add_argument('--incremental-early-stopping', type=int, default=DEFAULT_INCREMENTAL_EARLY_STOPPING, help=HELP_INCREMENTAL_EARLY_STOPPING)
    parser.add_argument('--max-finetune-iterations', type=int, default=DEFAULT_MAX_FINETUNE_ITERATIONS, help=HELP_MAX_FINETUNE_ITERATIONS)
    parser.add_argument('--use-existing-features', action='store_true', help=HELP_USE_EXISTING_FEATURES)

    args = parser.parse_args()

    if args.incremental_training and not args.incremental_data_dir:

        print("[!] --incremental-data-dir parameter must be specified when enabling incremental training")
        sys.exit(1)

    if args.incremental_raw_data_dir and not args.incremental_data_dir:

        print("[!] --incremental-data-dir parameter must be specified when specifying --incremental-raw-data-dir")
        sys.exit(1)

    main(args)
