#   Copyright 2025-2026 KoloStudio & Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import argparse
import re
import importlib.util
import asyncio
import sys
import json
import ctypes
import csv


from config.config import (
    MODEL_PATH,
    FAMILY_CLASSIFIER_PATH,
    FEATURES_PKL_PATH,
    FEATURE_SCALER_PATH,
    PROCESSED_DATA_DIR,
    METADATA_FILE,
    SAVED_MODEL_DIR,
    RESOURCES_DIR,
    THRESHOLD_REPORT_PATH,
    ROUTING_EVAL_REPORT_PATH,
    BENIGN_SAMPLES_DIR,
    MALICIOUS_SAMPLES_DIR,
    PROJECT_ROOT,
    DEFAULT_MAX_FILE_SIZE,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_INCREMENTAL_ROUNDS,
    DEFAULT_INCREMENTAL_EARLY_STOPPING,
    DEFAULT_MAX_FINETUNE_ITERATIONS,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_MIN_FAMILY_SIZE,
    DEFAULT_TREAT_NOISE_AS_FAMILY,
    PREDICTION_THRESHOLD,
    ENV_ALLOWED_SCAN_ROOT,
    SCAN_CACHE_PATH,
    SCAN_OUTPUT_DIR,
    HDBSCAN_SAVE_DIR,
    HELP_MAX_FILE_SIZE,
    HELP_FAST_DEV_RUN,
    HELP_SAVE_FEATURES,
    HELP_FINETUNE_ON_FALSE_POSITIVES,
    HELP_INCREMENTAL_TRAINING,
    HELP_INCREMENTAL_DATA_DIR,
    HELP_INCREMENTAL_RAW_DATA_DIR,
    HELP_FILE_EXTENSIONS,
    HELP_LABEL_INFERENCE,
    HELP_NUM_BOOST_ROUND,
    HELP_INCREMENTAL_ROUNDS,
    HELP_INCREMENTAL_EARLY_STOPPING,
    HELP_MAX_FINETUNE_ITERATIONS,
    HELP_USE_EXISTING_FEATURES,
    HELP_DATA_DIR,
    HELP_FEATURES_PATH,
    HELP_SAVE_DIR,
    HELP_MIN_CLUSTER_SIZE,
    HELP_MIN_SAMPLES,
    HELP_MIN_FAMILY_SIZE,
    HELP_PLOT_PCA,
    HELP_EXPLAIN_DISCREPANCY,
    HELP_TREAT_NOISE_AS_FAMILY,
    HELP_LIGHTGBM_MODEL_PATH,
    HELP_FAMILY_CLASSIFIER_PATH,
    HELP_CACHE_FILE,
    HELP_FILE_PATH,
    HELP_DIR_PATH,
    HELP_RECURSIVE,
    HELP_OUTPUT_PATH,
    HELP_AUTOML_METHOD,
    HELP_AUTOML_TRIALS,
    HELP_AUTOML_CV,
    HELP_AUTOML_METRIC,
    HELP_AUTOML_FAST_DEV_RUN,
    HELP_SKIP_TUNING,
    AUTOML_METHOD_DEFAULT,
    AUTOML_TRIALS_DEFAULT,
    AUTOML_CV_FOLDS_DEFAULT,
    AUTOML_METRIC_DEFAULT,
    DETECTED_MALICIOUS_PATHS_REPORT_PATH,
)
from utils.logging_utils import (
    configure_logging,
    get_logger,
    redirect_console_to_logger_allow_progress,
    redirect_print_to_logger,
    set_log_level,
)


def _serve_ipc_only() -> str:
    logger = get_logger("kolo")
    logger.debug("进入IPC服务启动流程")
    import scanner_service

    asyncio.run(scanner_service.run_ipc_forever())
    logger.debug("IPC服务流程结束")
    return "ipc"


_KVD_SCAN_DLL = None
_KVD_SCAN_DLL_READY = False


class _KvdConfig(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("model_normal_path", ctypes.c_char_p),
        ("model_packed_path", ctypes.c_char_p),
        ("family_classifier_json_path", ctypes.c_char_p),
        ("allowed_scan_root", ctypes.c_char_p),
        ("max_file_size", ctypes.c_uint),
        ("prediction_threshold", ctypes.c_float),
    ]


def _scan_dll_candidates():
    env_path = os.getenv("KVD_SCAN_DLL")
    base_dir = (
        PROJECT_ROOT
        if PROJECT_ROOT
        else os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    return [
        env_path,
        os.path.join(base_dir, "build-vcpkg", "bin", "Release", "axon_engine.dll"),
        os.path.join(base_dir, "build-vcpkg", "bin", "Debug", "axon_engine.dll"),
        os.path.join(
            base_dir, "build-vcpkg", "bin", "RelWithDebInfo", "axon_engine.dll"
        ),
        os.path.join(base_dir, "build-vcpkg", "bin", "MinSizeRel", "axon_engine.dll"),
        os.path.join(base_dir, "build", "Release", "axon_engine.dll"),
        os.path.join(base_dir, "build", "Debug", "axon_engine.dll"),
        os.path.join(
            base_dir, "src", "cpp", "build", "bin", "Release", "axon_engine.dll"
        ),
        os.path.join(
            base_dir, "src", "cpp", "build", "bin", "Debug", "axon_engine.dll"
        ),
        os.path.join(
            base_dir, "src", "cpp", "build", "src", "Release", "axon_engine.dll"
        ),
        os.path.join(
            base_dir, "src", "cpp", "build", "src", "Debug", "axon_engine.dll"
        ),
        os.path.join(
            base_dir,
            "src",
            "cpp",
            "kvd_core",
            "build",
            "bin",
            "Release",
            "axon_engine.dll",
        ),
        os.path.join(
            base_dir,
            "src",
            "cpp",
            "kvd_core",
            "build",
            "bin",
            "Debug",
            "axon_engine.dll",
        ),
        os.path.join(
            base_dir,
            "src",
            "cpp",
            "kvd_core",
            "build",
            "src",
            "Release",
            "axon_engine.dll",
        ),
        os.path.join(
            base_dir,
            "src",
            "cpp",
            "kvd_core",
            "build",
            "src",
            "Debug",
            "axon_engine.dll",
        ),
        os.path.join(os.path.dirname(__file__), "axon_engine.dll"),
    ]


def _load_kvd_scan_dll():
    global _KVD_SCAN_DLL, _KVD_SCAN_DLL_READY
    if _KVD_SCAN_DLL_READY:
        return _KVD_SCAN_DLL
    for candidate in _scan_dll_candidates():
        if not candidate:
            continue
        if not os.path.isfile(candidate):
            continue
        try:
            dll = ctypes.CDLL(candidate)
            dll.kvd_create.argtypes = [ctypes.POINTER(_KvdConfig)]
            dll.kvd_create.restype = ctypes.c_void_p
            dll.kvd_destroy.argtypes = [ctypes.c_void_p]
            dll.kvd_destroy.restype = None
            dll.kvd_free.argtypes = [ctypes.c_char_p]
            dll.kvd_free.restype = None
            dll.kvd_scan_path.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            dll.kvd_scan_path.restype = ctypes.c_int
            if hasattr(dll, "kvd_scan_paths"):
                dll.kvd_scan_paths.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_char_p),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_char_p),
                    ctypes.POINTER(ctypes.c_size_t),
                ]
                dll.kvd_scan_paths.restype = ctypes.c_int
            _KVD_SCAN_DLL = dll
            _KVD_SCAN_DLL_READY = True
            return _KVD_SCAN_DLL
        except Exception:
            continue
    _KVD_SCAN_DLL_READY = True
    return None


def _build_kvd_config(model_path, family_classifier_path, max_file_size):
    allowed_root = os.getenv(ENV_ALLOWED_SCAN_ROOT)
    return _KvdConfig(
        model_path=model_path.encode("utf-8") if model_path else None,
        model_normal_path=None,
        model_packed_path=None,
        family_classifier_json_path=(
            family_classifier_path.encode("utf-8") if family_classifier_path else None
        ),
        allowed_scan_root=allowed_root.encode("utf-8") if allowed_root else None,
        max_file_size=int(max_file_size) if max_file_size else 0,
        prediction_threshold=(
            float(PREDICTION_THRESHOLD) if PREDICTION_THRESHOLD else 0.0
        ),
    )


def _kvd_scan_path(dll, handle, path):
    out_json = ctypes.c_char_p()
    out_len = ctypes.c_size_t()
    rc = dll.kvd_scan_path(
        handle, path.encode("utf-8"), ctypes.byref(out_json), ctypes.byref(out_len)
    )
    if rc != 0:
        raise RuntimeError(f"kvd_scan_path failed: {rc}")
    raw = ctypes.string_at(out_json, out_len.value) if out_json.value else b""
    if out_json.value:
        dll.kvd_free(out_json)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _kvd_scan_paths(dll, handle, paths):
    encoded = [p.encode("utf-8") if p else None for p in paths]
    array_type = ctypes.c_char_p * len(encoded)
    out_json = ctypes.c_char_p()
    out_len = ctypes.c_size_t()
    rc = dll.kvd_scan_paths(
        handle,
        array_type(*encoded),
        len(encoded),
        ctypes.byref(out_json),
        ctypes.byref(out_len),
    )
    if rc != 0:
        raise RuntimeError(f"kvd_scan_paths failed: {rc}")
    raw = ctypes.string_at(out_json, out_len.value) if out_json.value else b""
    if out_json.value:
        dll.kvd_free(out_json)
    if not raw:
        return []
    data = json.loads(raw.decode("utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def _collect_scan_paths(directory_path, recursive):
    results = []
    if recursive:
        for root, _, files in os.walk(directory_path):
            for name in files:
                results.append(os.path.join(root, name))
    else:
        with os.scandir(directory_path) as it:
            for entry in it:
                if entry.is_file():
                    results.append(entry.path)
    return results


def _enrich_scan_results(results, paths):
    enriched = []
    for idx, path in enumerate(paths):
        base = (
            results[idx]
            if idx < len(results) and isinstance(results[idx], dict)
            else {}
        )
        item = dict(base)
        if "file_path" not in item:
            item["file_path"] = os.path.abspath(path)
        if "file_name" not in item:
            item["file_name"] = os.path.basename(path)
        if "file_size" not in item:
            try:
                item["file_size"] = int(os.path.getsize(path))
            except Exception:
                item["file_size"] = 0
        enriched.append(item)
    return enriched


def _save_scan_results(results, output_path, logger):
    json_path = output_path + ".json"
    json_dir = os.path.dirname(json_path)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"扫描结果已保存: {json_path}")

    csv_path = output_path + ".csv"
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if results:
            fieldnames = [
                "file_path",
                "file_name",
                "file_size",
                "is_malware",
                "confidence",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "file_path": result.get("file_path", ""),
                        "file_name": result.get("file_name", ""),
                        "file_size": result.get("file_size", 0),
                        "is_malware": result.get("is_malware", False),
                        "confidence": result.get("confidence", 0.0),
                    }
                )
    logger.info(f"扫描结果已保存: {csv_path}")


def _build_feature_name_map(model, feature_columns):
    import re

    model_feature_names = list(model.feature_name() or [])
    if len(model_feature_names) != len(feature_columns):
        model_feature_names = list(feature_columns[: len(model_feature_names)])
    if len(model_feature_names) < len(feature_columns):
        model_feature_names.extend(feature_columns[len(model_feature_names) :])
    feature_name_map = {}
    for i, feature_name in enumerate(model_feature_names):
        if i < len(model_feature_names):
            feature_name = str(model_feature_names[i])
        elif i < len(feature_columns):
            feature_name = str(feature_columns[i])
        else:
            feature_name = f"feature_{i}"
        m = re.search(r"(\d+)$", feature_name)
        if m:
            feature_id = int(m.group(1))
        else:
            feature_id = int(i)
        feature_name_map[f"Column_{feature_id}"] = feature_name
    return feature_name_map


def _build_sample_shap_map(shap_row):
    import numpy as np

    values = np.asarray(shap_row, dtype=np.float32)
    feature_dim = values.shape[0]
    if feature_dim > 1:
        values = values[:-1]
    abs_values = np.abs(values)
    return {f"Column_{i}": float(abs_values[i]) for i in range(abs_values.shape[0])}


def _build_sample_payload(
    sample_indices,
    sample_ids,
    y_true,
    y_pred,
    y_pred_proba,
    shap_values,
    feature_name_map,
):
    payload = []
    for pos, idx in enumerate(sample_indices):
        feature_importance = _build_sample_shap_map(shap_values[pos])
        payload.append(
            {
                "sample_id": str(sample_ids[idx]),
                "true_label": int(y_true[idx]),
                "predicted_label": int(y_pred[idx]),
                "prediction_probability": float(y_pred_proba[idx]),
                "feature_importance": feature_importance,
                "feature_name_map": feature_name_map,
            }
        )
    return payload


def _write_sample_report(report_path, samples, sample_name, logger):
    import json

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        if not samples:
            logger.warning(f"{sample_name}为空，已生成空文件: {report_path}")
    except Exception as e:
        logger.warning(f"{sample_name}写入失败，改为空文件: {e}")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def _export_train_all_sample_reports(logger):
    logger.debug("进入样本报告导出流程")
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import pickle
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    from config.config import (
        PREDICTION_THRESHOLD,
        DEFAULT_TEST_SIZE,
        DEFAULT_VAL_SIZE,
        DEFAULT_RANDOM_STATE,
        FEATURE_SCALER_PATH,
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = HDBSCAN_SAVE_DIR
    os.makedirs(output_dir, exist_ok=True)
    hard_path = os.path.join(output_dir, f"hard_samples_{timestamp}.json")
    fp_path = os.path.join(output_dir, f"false_positives_{timestamp}.json")
    fn_path = os.path.join(output_dir, f"false_negatives_{timestamp}.json")

    hard_samples = []
    false_positive_samples = []
    false_negative_samples = []
    full_hard_count = 0
    full_fp_count = 0
    full_fn_count = 0
    test_hard_count = 0
    test_fp_count = 0
    test_fn_count = 0
    try:
        confidence_threshold = 0.70
        df = pd.read_pickle(FEATURES_PKL_PATH)
        required_columns = {"filename", "label"}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError(f"特征文件缺少必要列: {required_columns}")
        feature_columns = [col for col in df.columns if col.startswith("feature_")]
        if not feature_columns:
            raise ValueError("特征文件中未找到有效特征列")
        sample_ids = df["filename"].tolist()
        y_true = df["label"].astype(int).to_numpy()
        X = df[feature_columns].to_numpy()
        if os.path.exists(FEATURE_SCALER_PATH):
            with open(FEATURE_SCALER_PATH, "rb") as f:
                export_scaler = pickle.load(f)
            X_scaled = export_scaler.transform(X)
            logger.info(f"样本导出使用标准化器: {FEATURE_SCALER_PATH}")
        else:
            X_scaled = X
            logger.warning(
                f"样本导出未找到标准化器，使用原始特征: {FEATURE_SCALER_PATH}"
            )

        train_model_path = MODEL_PATH
        if str(train_model_path).lower().endswith(".onnx"):
            candidate = os.path.splitext(train_model_path)[0] + ".train.txt"
            if os.path.exists(candidate):
                train_model_path = candidate
            else:
                fallback = os.path.splitext(train_model_path)[0] + ".txt"
                train_model_path = (
                    fallback if os.path.exists(fallback) else train_model_path
                )
        model = lgb.Booster(model_file=train_model_path)
        best_iteration = getattr(model, "best_iteration", None)

        total_count = len(y_true)
        all_indices = np.arange(total_count)
        if total_count > 10:
            stratify_label = y_true if len(np.unique(y_true)) > 1 else None
            idx_temp, idx_test, y_temp, y_test, sample_ids_temp, sample_ids_test = (
                train_test_split(
                    all_indices,
                    y_true,
                    sample_ids,
                    test_size=DEFAULT_TEST_SIZE,
                    random_state=DEFAULT_RANDOM_STATE,
                    stratify=stratify_label,
                )
            )
            if len(idx_temp) > 5:
                stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None
                idx_train, idx_val, y_train, y_val, sample_ids_train, sample_ids_val = (
                    train_test_split(
                        idx_temp,
                        y_temp,
                        sample_ids_temp,
                        test_size=DEFAULT_VAL_SIZE,
                        random_state=DEFAULT_RANDOM_STATE,
                        stratify=stratify_temp,
                    )
                )
            else:
                idx_train = idx_val = idx_temp
                y_train = y_val = y_temp
                sample_ids_train = sample_ids_val = sample_ids_temp
        else:
            idx_test = all_indices
            idx_train = idx_val = all_indices
            y_test = y_train = y_val = y_true
            sample_ids_test = sample_ids_train = sample_ids_val = sample_ids

        test_mask = np.zeros(total_count, dtype=bool)
        test_mask[idx_test] = True

        full_y_pred_proba = (
            model.predict(X_scaled, num_iteration=best_iteration)
            if isinstance(best_iteration, int) and best_iteration > 0
            else model.predict(X_scaled)
        )
        y_test_pred_proba = full_y_pred_proba[idx_test]
        y_test_pred = (y_test_pred_proba > PREDICTION_THRESHOLD).astype(int)
        confidence_test = np.maximum(y_test_pred_proba, 1 - y_test_pred_proba)
        hard_mask_test = confidence_test < confidence_threshold
        false_positive_mask_test = (y_test == 0) & (y_test_pred == 1)
        false_negative_mask_test = (y_test == 1) & (y_test_pred == 0)
        test_hard_count = int(np.sum(hard_mask_test))
        test_fp_count = int(np.sum(false_positive_mask_test))
        test_fn_count = int(np.sum(false_negative_mask_test))

        full_y_pred = (full_y_pred_proba > PREDICTION_THRESHOLD).astype(int)
        full_confidence = np.maximum(full_y_pred_proba, 1 - full_y_pred_proba)
        full_hard_mask = full_confidence < confidence_threshold
        full_false_positive_mask = (y_true == 0) & (full_y_pred == 1)
        full_false_negative_mask = (y_true == 1) & (full_y_pred == 0)
        full_hard_count = int(np.sum(full_hard_mask))
        full_fp_count = int(np.sum(full_false_positive_mask))
        full_fn_count = int(np.sum(full_false_negative_mask))

        feature_name_map = _build_feature_name_map(model, feature_columns)
        hard_indices = np.where(full_hard_mask)[0]
        fp_indices = np.where(full_false_positive_mask)[0]
        fn_indices = np.where(full_false_negative_mask)[0]
        shap_batch_size = 256

        def _predict_shap_for_indices(indices):
            if indices.size == 0:
                return np.zeros((0, len(feature_columns) + 1), dtype=np.float32)
            chunks = []
            for start in range(0, indices.size, shap_batch_size):
                batch_idx = indices[start : start + shap_batch_size]
                batch_X = X_scaled[batch_idx]
                if isinstance(best_iteration, int) and best_iteration > 0:
                    batch_shap = model.predict(
                        batch_X, num_iteration=best_iteration, pred_contrib=True
                    )
                else:
                    batch_shap = model.predict(batch_X, pred_contrib=True)
                chunks.append(np.asarray(batch_shap, dtype=np.float32))
            return (
                np.vstack(chunks)
                if chunks
                else np.zeros((0, len(feature_columns) + 1), dtype=np.float32)
            )

        hard_shap = _predict_shap_for_indices(hard_indices)
        fp_shap = _predict_shap_for_indices(fp_indices)
        fn_shap = _predict_shap_for_indices(fn_indices)
        hard_samples = _build_sample_payload(
            hard_indices,
            sample_ids,
            y_true,
            full_y_pred,
            full_y_pred_proba,
            hard_shap,
            feature_name_map,
        )
        false_positive_samples = _build_sample_payload(
            fp_indices,
            sample_ids,
            y_true,
            full_y_pred,
            full_y_pred_proba,
            fp_shap,
            feature_name_map,
        )
        false_negative_samples = _build_sample_payload(
            fn_indices,
            sample_ids,
            y_true,
            full_y_pred,
            full_y_pred_proba,
            fn_shap,
            feature_name_map,
        )
    except Exception as e:
        logger.warning(f"样本分析阶段失败，已输出空结果文件: {e}")

    _write_sample_report(hard_path, hard_samples, "困难样本", logger)
    _write_sample_report(fp_path, false_positive_samples, "假阳性样本", logger)
    _write_sample_report(fn_path, false_negative_samples, "假阴性样本", logger)

    logger.info(f"困难样本文件: {hard_path}")
    logger.info(f"假阳性样本文件: {fp_path}")
    logger.info(f"假阴性样本文件: {fn_path}")
    logger.info(f"全量-困难样本数量: {full_hard_count}")
    logger.info(f"全量-假阳性样本数量: {full_fp_count}")
    logger.info(f"全量-假阴性样本数量: {full_fn_count}")
    logger.info(f"测试集-困难样本数量: {test_hard_count}")
    logger.info(f"测试集-假阳性样本数量: {test_fp_count}")
    logger.info(f"测试集-假阴性样本数量: {test_fn_count}")
    return {
        "hard_path": hard_path,
        "false_positive_path": fp_path,
        "false_negative_path": fn_path,
        "full_hard_count": full_hard_count,
        "full_false_positive_count": full_fp_count,
        "full_false_negative_count": full_fn_count,
        "test_hard_count": test_hard_count,
        "test_false_positive_count": test_fp_count,
        "test_false_negative_count": test_fn_count,
    }


def _export_train_all_deep_engine_eval_report(logger, hardcase_metrics):
    import json
    from datetime import datetime

    eval_dir = os.path.join(RESOURCES_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    report_path = os.path.join(eval_dir, "train_all_deep_engine_eval.json")
    metrics = hardcase_metrics if isinstance(hardcase_metrics, dict) else {}
    validation = (
        metrics.get("validation", {})
        if isinstance(metrics.get("validation", {}), dict)
        else {}
    )
    report = {
        "engine": "hardcase_dl",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "accuracy": float(validation.get("accuracy", 0.0)),
            "macro_f1": float(validation.get("macro_f1", 0.0)),
        },
        "dataset": metrics.get("dataset", {}),
        "validation": validation,
        "validation_stability": metrics.get("validation_stability", {}),
        "cascade": metrics.get("cascade", {}),
        "artifacts": {
            "model_path": metrics.get("model_path"),
            "cxx_manifest_path": metrics.get("cxx_manifest_path"),
            "plots": metrics.get("plots", {}),
            "metrics_source": os.path.join(eval_dir, "hardcase_dl_metrics.json"),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"深度学习引擎评测报告已生成: {report_path}")
    logger.info(
        f"深度学习引擎评测摘要: accuracy={report['summary']['accuracy']:.4f}, macro_f1={report['summary']['macro_f1']:.4f}"
    )
    return report_path


def _merge_train_all_evaluation_summary(
    logger, deep_engine_report_path=None, sample_report_payload=None
):
    import json
    from datetime import datetime

    eval_dir = os.path.join(RESOURCES_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    summary_candidates = [
        os.path.join(eval_dir, "train_all_evaluation_summary.json"),
        os.path.join(eval_dir, "evaluation_summary.json"),
        os.path.join(eval_dir, "model_evaluation_summary.json"),
    ]
    summary_path = summary_candidates[0]
    for p in summary_candidates:
        if os.path.exists(p):
            summary_path = p
            break
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                summary = loaded
        except Exception:
            summary = {}
    deep_path = deep_engine_report_path or os.path.join(
        eval_dir, "train_all_deep_engine_eval.json"
    )
    deep_report = {}
    if os.path.exists(deep_path):
        try:
            with open(deep_path, "r", encoding="utf-8") as f:
                loaded_deep = json.load(f)
            if isinstance(loaded_deep, dict):
                deep_report = loaded_deep
        except Exception:
            deep_report = {}
    hardcase_metrics_path = os.path.join(eval_dir, "hardcase_dl_metrics.json")
    hardcase_metrics = {}
    if os.path.exists(hardcase_metrics_path):
        try:
            with open(hardcase_metrics_path, "r", encoding="utf-8") as f:
                loaded_metrics = json.load(f)
            if isinstance(loaded_metrics, dict):
                hardcase_metrics = loaded_metrics
        except Exception:
            hardcase_metrics = {}
    threshold_report = {}
    if os.path.exists(THRESHOLD_REPORT_PATH):
        try:
            with open(THRESHOLD_REPORT_PATH, "r", encoding="utf-8") as f:
                loaded_threshold = json.load(f)
            if isinstance(loaded_threshold, dict):
                threshold_report = loaded_threshold
        except Exception:
            threshold_report = {}
    summary["generated_at"] = datetime.now().isoformat()
    summary["deep_engine"] = deep_report
    summary["deep_engine_metrics"] = hardcase_metrics
    if isinstance(sample_report_payload, dict):
        summary["sample_reports"] = sample_report_payload
    summary["threshold_report"] = threshold_report
    lightgbm_eval_result = _evaluate_lightgbm_model(logger)
    if lightgbm_eval_result:
        summary["lightgbm_model"] = lightgbm_eval_result
    combined_eval_result = _generate_combined_evaluation(logger, summary)
    if combined_eval_result:
        summary["combined_evaluation"] = combined_eval_result
    summary["report_paths"] = {
        "deep_engine": deep_path if os.path.exists(deep_path) else None,
        "deep_engine_metrics": (
            hardcase_metrics_path if os.path.exists(hardcase_metrics_path) else None
        ),
        "sample_reports_dir": (
            HDBSCAN_SAVE_DIR if os.path.exists(HDBSCAN_SAVE_DIR) else None
        ),
        "threshold_report": (
            THRESHOLD_REPORT_PATH if os.path.exists(THRESHOLD_REPORT_PATH) else None
        ),
        "routing_report": (
            ROUTING_EVAL_REPORT_PATH
            if os.path.exists(ROUTING_EVAL_REPORT_PATH)
            else None
        ),
        "lightgbm_eval": (
            lightgbm_eval_result.get("report_path") if lightgbm_eval_result else None
        ),
        "combined_eval": (
            combined_eval_result.get("report_path") if combined_eval_result else None
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"总评测汇总已更新: {summary_path}")
    return summary_path


def _evaluate_lightgbm_model(logger):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM 未安装，跳过 LightGBM 模型评测")
        return None
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_curve,
        roc_auc_score,
    )

    eval_dir = os.path.join(RESOURCES_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    report_path = os.path.join(eval_dir, "lightgbm_model_eval_report.json")
    fig_dir = os.path.join(eval_dir, "lightgbm_model_eval")
    os.makedirs(fig_dir, exist_ok=True)
    if not Path(FEATURES_PKL_PATH).exists():
        logger.warning(f"特征文件不存在: {FEATURES_PKL_PATH}，跳过 LightGBM 模型评测")
        return None
    try:
        df = pd.read_pickle(FEATURES_PKL_PATH)
        files = df["filename"].tolist()
        y = df["label"].values

        def _safe_int_sort_key(c):
            try:
                return int(c.split("_")[1])
            except (ValueError, IndexError):
                return -1

        feature_columns = [c for c in df.columns if c.startswith("feature_")]
        feature_columns = sorted(feature_columns, key=_safe_int_sort_key)
        X = df[feature_columns].values
    except Exception as e:
        logger.warning(f"无法加载特征文件进行 LightGBM 评测: {e}")
        return None
    if len(X) <= 10:
        logger.warning("样本数量不足，跳过 LightGBM 模型评测")
        return None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    if len(X_test) == 0:
        logger.warning("测试集为空，跳过 LightGBM 模型评测")
        return None
    model_file = os.path.join(SAVED_MODEL_DIR, "lightgbm_model.train.txt")
    if not os.path.exists(model_file):
        txt_candidates = list(Path(SAVED_MODEL_DIR).glob("*.train.txt"))
        if txt_candidates:
            model_file = str(txt_candidates[0])
        else:
            logger.warning(
                f"找不到 LightGBM 模型文件: {model_file}，跳过 LightGBM 模型评测"
            )
            return None
    try:
        model = lgb.Booster(model_file=model_file)
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
    except Exception as e:
        logger.warning(f"无法加载 LightGBM 模型: {e}，跳过 LightGBM 模型评测")
        return None
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr_val = fp / max(1, fp + tn)
    auc_value = 0.0
    fpr_arr, tpr_arr = None, None
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_pred_proba)
        auc_value = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        pass
    report = {
        "model_path": model_file,
        "test_samples": int(len(y_test)),
        "accuracy": float(acc),
        "precision": float(pre),
        "recall": float(rec),
        "f1_score": float(f1),
        "fpr": float(fpr_val),
        "auc": float(auc_value),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "generated_at": datetime.now().isoformat(),
    }
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
    )
    axes[0].set_title("LightGBM 二分类混淆矩阵")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=axes[1],
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
    )
    axes[1].set_title("LightGBM 混淆矩阵热力图（按行归一化）")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    cm_fig_path = os.path.join(fig_dir, "lightgbm_confusion_matrix.png")
    plt.savefig(cm_fig_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    if auc_value > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(fpr_arr, tpr_arr, label=f"AUC={auc_value:.4f}", color="darkorange")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("LightGBM ROC 曲线")
        ax2.legend(loc="lower right")
        plt.tight_layout()
        roc_fig_path = os.path.join(fig_dir, "lightgbm_roc_auc.png")
        plt.savefig(roc_fig_path, dpi=150, bbox_inches="tight")
        plt.close("all")
    else:
        roc_fig_path = None
    report["confusion_matrix_fig_path"] = cm_fig_path
    report["roc_auc_fig_path"] = roc_fig_path
    report["report_path"] = report_path
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"LightGBM 模型评测图表已保存: {cm_fig_path}")
    return report


def _generate_combined_evaluation(logger, summary):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime

    eval_dir = os.path.join(RESOURCES_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    hardcase_val = None
    if (
        "deep_engine_metrics" in summary
        and "validation" in summary["deep_engine_metrics"]
    ):
        hardcase_val = summary["deep_engine_metrics"]["validation"]
    lightgbm_report = summary.get("lightgbm_model")
    if not hardcase_val and not lightgbm_report:
        logger.warning("缺少 hardcase_dl 和 LightGBM 评测数据，跳过联合评测")
        return None
    report_path = os.path.join(eval_dir, "combined_evaluation_report.json")
    fig_dir = os.path.join(eval_dir, "combined_evaluation")
    os.makedirs(fig_dir, exist_ok=True)
    combined = {
        "generated_at": datetime.now().isoformat(),
        "report_path": report_path,
    }
    if hardcase_val:
        combined["hardcase_dl"] = {
            "accuracy": hardcase_val.get("accuracy"),
            "macro_f1": hardcase_val.get("macro_f1"),
            "confusion_matrix": hardcase_val.get("confusion_matrix"),
            "report": hardcase_val.get("report"),
        }
    if lightgbm_report:
        combined["lightgbm_model"] = {
            "accuracy": lightgbm_report.get("accuracy"),
            "f1_score": lightgbm_report.get("f1_score"),
            "precision": lightgbm_report.get("precision"),
            "recall": lightgbm_report.get("recall"),
            "auc": lightgbm_report.get("auc"),
            "fpr": lightgbm_report.get("fpr"),
            "confusion_matrix": lightgbm_report.get("confusion_matrix"),
        }
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    def _plot_cm_with_normalized(cm, ax, xlabels, ylabels, title, cmap="Blues"):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            ax=ax,
            xticklabels=xlabels,
            yticklabels=ylabels,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        return cm_norm

    if hardcase_val and lightgbm_report:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        hardcase_cm = np.array(
            hardcase_val.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        )
        _plot_cm_with_normalized(
            hardcase_cm,
            axes[0],
            ["Hard", "FP", "FN"],
            ["Hard", "FP", "FN"],
            "HardCase DL 三分类混淆矩阵",
        )
        lightgbm_cm_dict = lightgbm_report.get("confusion_matrix", {})
        lightgbm_cm = np.array(
            [
                [lightgbm_cm_dict.get("tn", 0), lightgbm_cm_dict.get("fp", 0)],
                [lightgbm_cm_dict.get("fn", 0), lightgbm_cm_dict.get("tp", 0)],
            ]
        )
        sns.heatmap(
            lightgbm_cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            ax=axes[1],
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"],
        )
        axes[1].set_title("LightGBM 二分类混淆矩阵")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        plt.tight_layout()
        combined_cm_path = os.path.join(fig_dir, "combined_confusion_matrices.png")
        plt.savefig(combined_cm_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        combined["combined_confusion_matrices_path"] = combined_cm_path
        metrics_data = {
            "Model": ["HardCase DL", "LightGBM"],
            "Accuracy": [
                hardcase_val.get("accuracy", 0),
                lightgbm_report.get("accuracy", 0),
            ],
            "F1-Score": [
                hardcase_val.get("macro_f1", 0),
                lightgbm_report.get("f1_score", 0),
            ],
            "Precision": [
                (hardcase_val.get("report", {}).get("macro avg", {}) or {}).get(
                    "precision", 0
                ),
                lightgbm_report.get("precision", 0),
            ],
            "Recall": [
                (hardcase_val.get("report", {}).get("macro avg", {}) or {}).get(
                    "recall", 0
                ),
                lightgbm_report.get("recall", 0),
            ],
        }
        df_metrics = pd.DataFrame(metrics_data)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_metrics["Model"]))
        width = 0.2
        ax3.bar(
            x - 1.5 * width,
            df_metrics["Accuracy"],
            width,
            label="Accuracy",
            color="steelblue",
        )
        ax3.bar(
            x - 0.5 * width,
            df_metrics["F1-Score"],
            width,
            label="F1-Score",
            color="darkorange",
        )
        ax3.bar(
            x + 0.5 * width,
            df_metrics["Precision"],
            width,
            label="Precision",
            color="green",
        )
        ax3.bar(
            x + 1.5 * width, df_metrics["Recall"], width, label="Recall", color="red"
        )
        ax3.set_xlabel("Model")
        ax3.set_ylabel("Score")
        ax3.set_title("HardCase DL vs LightGBM 性能对比")
        ax3.set_xticks(x)
        ax3.set_xticklabels(df_metrics["Model"])
        ax3.legend()
        ax3.set_ylim(0, 1.0)
        for i, model in enumerate(df_metrics["Model"]):
            ax3.text(
                i - 1.5 * width,
                df_metrics["Accuracy"].iloc[i] + 0.02,
                f"{df_metrics['Accuracy'].iloc[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax3.text(
                i - 0.5 * width,
                df_metrics["F1-Score"].iloc[i] + 0.02,
                f"{df_metrics['F1-Score'].iloc[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax3.text(
                i + 0.5 * width,
                df_metrics["Precision"].iloc[i] + 0.02,
                f"{df_metrics['Precision'].iloc[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax3.text(
                i + 1.5 * width,
                df_metrics["Recall"].iloc[i] + 0.02,
                f"{df_metrics['Recall'].iloc[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        comparison_path = os.path.join(fig_dir, "model_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        combined["model_comparison_path"] = comparison_path
    elif hardcase_val:
        hardcase_cm = np.array(
            hardcase_val.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cm_norm = _plot_cm_with_normalized(
            hardcase_cm,
            axes[0],
            ["Hard", "FP", "FN"],
            ["Hard", "FP", "FN"],
            "HardCase DL 三分类混淆矩阵",
        )
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            ax=axes[1],
            xticklabels=["Hard", "FP", "FN"],
            yticklabels=["Hard", "FP", "FN"],
        )
        axes[1].set_title("HardCase DL 混淆矩阵热力图（按行归一化）")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        plt.tight_layout()
        combined_cm_path = os.path.join(fig_dir, "hardcase_confusion_matrices.png")
        plt.savefig(combined_cm_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        combined["combined_confusion_matrices_path"] = combined_cm_path
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    logger.info(f"联合评测报告已生成: {report_path}")
    return combined


def _convert_all_weights_to_onnx(weights_dir, logger):
    import json
    import pickle
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    from collections import OrderedDict
    import lightgbm as lgb
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType
    from onnxmltools import convert_lightgbm, convert_xgboost
    from onnxmltools.convert.common.data_types import (
        FloatTensorType as OnnxFloatTensorType,
    )

    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)
    summary = {"converted": [], "skipped": [], "failed": []}

    def _append(kind, src, dst=None, reason=""):
        item = {"source": str(src)}
        if dst is not None:
            item["target"] = str(dst)
        if reason:
            item["reason"] = reason
        summary[kind].append(item)

    txt_models = [
        p
        for p in sorted(weights_path.glob("*.txt"))
        if not p.name.endswith(".train.txt")
    ]
    for src in txt_models:
        try:
            booster = lgb.Booster(model_file=str(src))
            input_dim = int(booster.num_feature())
            model = convert_lightgbm(
                booster,
                initial_types=[("input", OnnxFloatTensorType([None, input_dim]))],
                target_opset=15,
            )
            dst = src.with_suffix(".onnx")
            with open(dst, "wb") as f:
                f.write(model.SerializeToString())
            _append("converted", src, dst)
        except Exception as e:
            _append("failed", src, reason=str(e))

    scaler_path = weights_path / "feature_scaler.pkl"
    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            if hasattr(scaler, "n_features_in_"):
                input_dim = int(scaler.n_features_in_)
            elif hasattr(scaler, "mean_"):
                input_dim = int(len(scaler.mean_))
            else:
                raise RuntimeError("无法识别feature_scaler输入维度")
            scaler_onnx = to_onnx(
                scaler,
                initial_types=[("input", SkFloatTensorType([None, input_dim]))],
                target_opset=15,
            )
            scaler_out = scaler_path.with_suffix(".onnx")
            with open(scaler_out, "wb") as f:
                f.write(scaler_onnx.SerializeToString())
            _append("converted", scaler_path, scaler_out)
        except Exception as e:
            _append("failed", scaler_path, reason=str(e))

    hardcase_payload_path = weights_path / "hardcase_dl_model.pkl"
    if hardcase_payload_path.exists():
        try:
            with open(hardcase_payload_path, "rb") as f:
                payload = pickle.load(f)
            selected = payload.get("selected_feature_indices")
            input_dim = int(len(selected)) if selected is not None else 384

            lgb_ovr = payload.get("lgb_ovr")
            if lgb_ovr is not None and hasattr(lgb_ovr, "estimators_"):
                for i, est in enumerate(lgb_ovr.estimators_):
                    try:
                        lgb_model = convert_lightgbm(
                            est,
                            initial_types=[
                                ("input", OnnxFloatTensorType([None, input_dim]))
                            ],
                            target_opset=15,
                        )
                        dst = weights_path / f"hardcase_lgb_ovr_class_{i}.onnx"
                        with open(dst, "wb") as f:
                            f.write(lgb_model.SerializeToString())
                        _append("converted", f"{hardcase_payload_path}#lgb_{i}", dst)
                    except Exception as sub_e:
                        _append(
                            "failed",
                            f"{hardcase_payload_path}#lgb_{i}",
                            reason=str(sub_e),
                        )

            xgb_ovr = payload.get("xgb_ovr")
            if xgb_ovr is not None and hasattr(xgb_ovr, "estimators_"):
                for i, est in enumerate(xgb_ovr.estimators_):
                    try:
                        xgb_model = convert_xgboost(
                            est,
                            initial_types=[
                                ("input", OnnxFloatTensorType([None, input_dim]))
                            ],
                            target_opset=15,
                        )
                        dst = weights_path / f"hardcase_xgb_ovr_class_{i}.onnx"
                        with open(dst, "wb") as f:
                            f.write(xgb_model.SerializeToString())
                        _append("converted", f"{hardcase_payload_path}#xgb_{i}", dst)
                    except Exception as sub_e:
                        _append(
                            "failed",
                            f"{hardcase_payload_path}#xgb_{i}",
                            reason=str(sub_e),
                        )

            onnx_manifest = {
                "input_dim": input_dim,
                "selected_feature_indices": (
                    selected.tolist() if hasattr(selected, "tolist") else selected
                ),
                "scaler_onnx": (
                    "feature_scaler.onnx"
                    if (weights_path / "feature_scaler.onnx").exists()
                    else None
                ),
                "main_lightgbm_onnx": (
                    "lightgbm_model.onnx"
                    if (weights_path / "lightgbm_model.onnx").exists()
                    else None
                ),
                "normal_lightgbm_onnx": (
                    "lightgbm_model_normal.onnx"
                    if (weights_path / "lightgbm_model_normal.onnx").exists()
                    else None
                ),
                "packed_lightgbm_onnx": (
                    "lightgbm_model_packed.onnx"
                    if (weights_path / "lightgbm_model_packed.onnx").exists()
                    else None
                ),
                "hardcase_lgb_ovr_onnx": [
                    f"hardcase_lgb_ovr_class_{i}.onnx" for i in range(3)
                ],
                "hardcase_xgb_ovr_onnx": [
                    f"hardcase_xgb_ovr_class_{i}.onnx" for i in range(3)
                ],
            }
            manifest_path = weights_path / "weights_onnx_manifest.json"
            manifest_path.write_text(
                json.dumps(onnx_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _append("converted", hardcase_payload_path, manifest_path)
        except Exception as e:
            _append("failed", hardcase_payload_path, reason=str(e))

    class _LegacyAttentionExpert(nn.Module):
        def __init__(self, input_dim, hidden_dim, heads, latent_tokens):
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.heads = int(heads)
            self.latent_tokens = int(latent_tokens)
            self.to_tokens = nn.Linear(
                self.input_dim, self.hidden_dim * self.latent_tokens
            )
            self.attn = nn.MultiheadAttention(
                self.hidden_dim, self.heads, batch_first=True
            )
            self.norm1 = nn.LayerNorm(self.hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
            self.norm2 = nn.LayerNorm(self.hidden_dim)
            self.to_gate = nn.Linear(
                self.hidden_dim * self.latent_tokens, self.input_dim
            )

        def forward(self, x):
            tokens = self.to_tokens(x).view(-1, self.latent_tokens, self.hidden_dim)
            attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            tokens = self.norm1(tokens + attn_out)
            ffn_out = self.ffn(tokens)
            tokens = self.norm2(tokens + ffn_out)
            flattened = tokens.reshape(tokens.shape[0], -1)
            return self.to_gate(flattened)

    class _LegacyNNExpert(nn.Module):
        def __init__(self, input_dim, hidden_dim, dropout, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                OrderedDict(
                    [
                        ("0", nn.Linear(int(input_dim), int(hidden_dim))),
                        ("1", nn.ReLU()),
                        ("2", nn.BatchNorm1d(int(hidden_dim))),
                        ("3", nn.Dropout(float(dropout))),
                        ("4", nn.Linear(int(hidden_dim), int(hidden_dim // 2))),
                        ("5", nn.ReLU()),
                        ("6", nn.BatchNorm1d(int(hidden_dim // 2))),
                        ("7", nn.Dropout(float(dropout))),
                        ("8", nn.Linear(int(hidden_dim // 2), int(output_dim))),
                    ]
                )
            )

        def forward(self, x):
            out = self.net(x)
            if out.shape[-1] == 1:
                return torch.sigmoid(out)
            return out

    class _HardcaseBaseMLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                OrderedDict(
                    [
                        ("0", nn.Linear(int(input_dim), 512)),
                        ("1", nn.BatchNorm1d(512)),
                        ("2", nn.GELU()),
                        ("3", nn.Dropout(0.2)),
                        ("4", nn.Linear(512, 256)),
                        ("5", nn.BatchNorm1d(256)),
                        ("6", nn.GELU()),
                        ("7", nn.Dropout(0.2)),
                        ("8", nn.Linear(256, 128)),
                        ("9", nn.GELU()),
                        ("10", nn.Dropout(0.1)),
                        ("11", nn.Linear(128, int(output_dim))),
                    ]
                )
            )

        def forward(self, x):
            return self.net(x)

    class _HardcaseResidualBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.block = nn.Sequential(
                OrderedDict(
                    [
                        ("0", nn.Linear(int(dim), int(dim))),
                        ("1", nn.BatchNorm1d(int(dim))),
                        ("2", nn.GELU()),
                        ("3", nn.Dropout(0.15)),
                        ("4", nn.Linear(int(dim), int(dim))),
                        ("5", nn.BatchNorm1d(int(dim))),
                    ]
                )
            )

        def forward(self, x):
            return torch.relu(self.block(x) + x)

    class _HardcaseFnModel(nn.Module):
        def __init__(self, input_dim, stem_dim, proj_dim, head_dim, output_dim):
            super().__init__()
            self.stem = nn.Sequential(
                OrderedDict(
                    [
                        ("0", nn.Linear(int(input_dim), int(stem_dim))),
                        ("1", nn.BatchNorm1d(int(stem_dim))),
                        ("2", nn.GELU()),
                    ]
                )
            )
            self.res1 = _HardcaseResidualBlock(int(stem_dim))
            self.down1 = nn.Sequential(
                OrderedDict(
                    [
                        ("0", nn.Linear(int(stem_dim), int(proj_dim))),
                        ("1", nn.BatchNorm1d(int(proj_dim))),
                    ]
                )
            )
            self.res2 = _HardcaseResidualBlock(int(proj_dim))
            self.head = nn.Sequential(
                OrderedDict(
                    [
                        ("0", nn.Linear(int(proj_dim), int(head_dim))),
                        ("1", nn.GELU()),
                        ("2", nn.Dropout(0.1)),
                        ("3", nn.Linear(int(head_dim), int(output_dim))),
                    ]
                )
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.res1(x)
            x = torch.relu(self.down1(x))
            x = self.res2(x)
            return self.head(x)

    def _export_torch_onnx(model, input_dim, dst):
        model.eval()
        dummy = torch.randn(2, int(input_dim), dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            str(dst),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )

    def _secure_load_torch_model(pt_file):
        try:
            return torch.load(pt_file, map_location="cpu", weights_only=True), None
        except Exception:
            pass
        try:
            payload = torch.load(pt_file, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                return None, "Model payload is not a dictionary"
            if "state_dict" not in payload and "model_state_dict" not in payload:
                return None, "Model payload does not contain state_dict"
            return payload, None
        except Exception as e:
            return None, f"Unsafe model load rejected: {e}"

    for pt_file in sorted(weights_path.glob("*.pt")):
        try:
            payload, error = _secure_load_torch_model(pt_file)
            if error:
                _append("failed", pt_file, reason=error)
                continue
            state = None
            model = None
            input_dim = None
            output_dim = None

            if (
                isinstance(payload, dict)
                and "state_dict" in payload
                and "to_tokens.weight" in payload["state_dict"]
            ):
                state = payload["state_dict"]
                input_dim = int(
                    payload.get("input_dim", state["to_tokens.weight"].shape[1])
                )
                hidden_dim = int(
                    payload.get("hidden_dim", state["attn.out_proj.weight"].shape[0])
                )
                heads = int(payload.get("heads", 8))
                latent_tokens = int(
                    payload.get(
                        "latent_tokens",
                        state["to_tokens.weight"].shape[0] // hidden_dim,
                    )
                )
                model = _LegacyAttentionExpert(
                    input_dim, hidden_dim, heads, latent_tokens
                )
            elif (
                isinstance(payload, dict)
                and "state_dict" in payload
                and "net.0.weight" in payload["state_dict"]
            ):
                state = payload["state_dict"]
                input_dim = int(
                    payload.get("input_dim", state["net.0.weight"].shape[1])
                )
                hidden_dim = int(
                    payload.get("hidden_dim", state["net.0.weight"].shape[0])
                )
                output_dim = int(state["net.8.weight"].shape[0])
                dropout = float(payload.get("dropout", 0.2))
                model = _LegacyNNExpert(input_dim, hidden_dim, dropout, output_dim)
            elif (
                isinstance(payload, dict)
                and "model_state_dict" in payload
                and "stem.0.weight" in payload["model_state_dict"]
            ):
                state = payload["model_state_dict"]
                input_dim = int(state["stem.0.weight"].shape[1])
                stem_dim = int(state["stem.0.weight"].shape[0])
                proj_dim = int(state["down1.0.weight"].shape[0])
                head_dim = int(state["head.0.weight"].shape[0])
                output_dim = int(state["head.3.weight"].shape[0])
                model = _HardcaseFnModel(
                    input_dim, stem_dim, proj_dim, head_dim, output_dim
                )
            elif (
                isinstance(payload, dict)
                and "model_state_dict" in payload
                and "net.0.weight" in payload["model_state_dict"]
            ):
                state = payload["model_state_dict"]
                input_dim = int(
                    payload.get("input_dim", state["net.0.weight"].shape[1])
                )
                output_dim = int(
                    payload.get("num_classes", state["net.11.weight"].shape[0])
                )
                model = _HardcaseBaseMLP(input_dim, output_dim)
            else:
                raise RuntimeError(
                    f"未识别的PT权重结构，无法构建网络: keys={list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__}"
                )

            model.load_state_dict(state, strict=True)
            dst = pt_file.with_suffix(".onnx")
            _export_torch_onnx(model, input_dim, dst)
            _append("converted", pt_file, dst)
        except Exception as e:
            _append("failed", pt_file, reason=str(e))

    for pkl_file in sorted(weights_path.glob("*.pkl")):
        if pkl_file.name not in {"feature_scaler.pkl", "hardcase_dl_model.pkl"}:
            _append(
                "skipped",
                pkl_file,
                reason="当前仅支持feature_scaler.pkl与hardcase_dl_model.pkl自动导出",
            )

    summary_path = weights_path / "weights_onnx_conversion_report.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"ONNX转换完成，报告: {summary_path}")
    logger.info(
        f"ONNX转换统计 converted={len(summary['converted'])} skipped={len(summary['skipped'])} failed={len(summary['failed'])}"
    )
    return summary_path


def _evaluate_onnx_before_conversion(
    logger,
    features_pkl_path: str,
    scaler_path: str,
    model_path: str,
    resources_dir: str,
):
    import json
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_curve,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import seaborn as sns

    eval_dir = os.path.join(resources_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    report_path = os.path.join(eval_dir, "onnx_deployment_eval_report.json")

    try:
        df = pd.read_pickle(features_pkl_path)
        files = df["filename"].tolist()
        y = df["label"].values
        feature_columns = [c for c in df.columns if c.startswith("feature_")]
        feature_columns = sorted(feature_columns, key=lambda c: int(c.split("_")[1]))
        X = df[feature_columns].values
    except Exception as e:
        logger.warning(f"无法加载特征文件进行ONNX评测: {e}")
        return None

    if len(X) <= 10:
        logger.warning("样本数量不足，跳过ONNX部署前评测")
        return None

    X_temp, X_test, y_temp, y_test, files_temp, files_test = train_test_split(
        X,
        y,
        files,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    if len(X_test) == 0:
        logger.warning("测试集为空，跳过ONNX部署前评测")
        return None

    if not os.path.exists(scaler_path):
        logger.warning(f"找不到scaler文件: {scaler_path}，跳过ONNX部署前评测")
        return None

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        logger.warning(f"无法加载scaler: {e}，跳过ONNX部署前评测")
        return None

    onnx_path = model_path
    if not os.path.exists(onnx_path):
        txt_path = os.path.splitext(onnx_path)[0] + ".txt"
        if os.path.exists(txt_path):
            onnx_path = txt_path

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        y_pred_proba = sess.run(None, {input_name: X_test_scaled.astype(np.float32)})[
            0
        ].flatten()
    except Exception as e:
        logger.warning(f"无法加载ONNX模型: {e}，跳过ONNX部署前评测")
        return None

    y_pred = (y_pred_proba > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr_val = fp / max(1, fp + tn)

    auc_value = 0.0
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_pred_proba)
        auc_value = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        pass

    report = {
        "model_path": str(onnx_path),
        "test_samples": int(len(y_test)),
        "accuracy": float(acc),
        "precision": float(pre),
        "recall": float(rec),
        "f1_score": float(f1),
        "fpr": float(fpr_val),
        "auc": float(auc_value),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "generated_at": np.datetime64("now").astype(str),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"ONNX部署前评测报告已生成: {report_path}")
    logger.info(
        f"ONNX部署前评测摘要: accuracy={acc:.4f}, precision={pre:.4f}, recall={rec:.4f}, f1={f1:.4f}, auc={auc_value:.4f}, fpr={fpr_val:.4f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    if auc_value > 0:
        axes[1].plot(fpr_arr, tpr_arr, label=f"AUC={auc_value:.4f}", color="darkorange")
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend(loc="lower right")
    else:
        axes[1].text(0.5, 0.5, "ROC not available", ha="center", va="center")

    plt.tight_layout()
    fig_path = os.path.join(eval_dir, "onnx_deployment_eval.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"ONNX部署前评测图表已保存: {fig_path}")

    return report_path


def _ensure_onnx_artifacts_after_training(logger, keep_train_txt=False):
    import json

    logger.info("开始ONNX转换前的部署前评测...")
    onnx_eval_report = _evaluate_onnx_before_conversion(
        logger, FEATURES_PKL_PATH, FEATURE_SCALER_PATH, MODEL_PATH, RESOURCES_DIR
    )
    summary_path = _convert_all_weights_to_onnx(SAVED_MODEL_DIR, logger)
    summary = json.loads(open(summary_path, "r", encoding="utf-8").read())
    failed = summary.get("failed", [])
    if failed:
        raise RuntimeError(f"训练后ONNX导出存在失败项: {len(failed)}")
    if not keep_train_txt:
        train_txt = os.path.splitext(MODEL_PATH)[0] + ".train.txt"
        if os.path.exists(train_txt):
            os.remove(train_txt)
            logger.info(f"已清理非增量训练侧车文件: {train_txt}")


def main():
    import matplotlib

    matplotlib.use("Agg")
    log_level = os.environ.get("KVD_LOG_LEVEL", "INFO")
    configure_logging(log_file_name="app.log", level=log_level)
    set_log_level(log_level)
    redirect_print_to_logger("kolo.print")
    if os.environ.get("KVD_REDIRECT_CONSOLE", "0") == "1":
        redirect_console_to_logger_allow_progress("kolo.console")
    logger = get_logger("kolo")
    logger.debug(f"进入主流程 argv={sys.argv}")
    parser = argparse.ArgumentParser(
        prog="KoloVirusDetector", description="KoloVirusDetector 项目入口"
    )
    subs = parser.add_subparsers(dest="command", required=True)

    sp_pretrain = subs.add_parser("pretrain", help="预训练 LightGBM 模型")
    sp_pretrain.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help="单个样本允许处理的最大字节数",
    )
    sp_pretrain.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="快速开发模式，仅使用小规模样本进行流程验证",
    )
    sp_pretrain.add_argument(
        "--save-features", action="store_true", help="保存本次提取的特征到特征文件"
    )
    sp_pretrain.add_argument(
        "--finetune-on-false-positives",
        action="store_true",
        help="检测到假阳性后继续进行强化训练",
    )
    sp_pretrain.add_argument(
        "--incremental-training",
        action="store_true",
        help="启用增量训练，基于已有模型继续训练",
    )
    sp_pretrain.add_argument(
        "--incremental-data-dir", type=str, help="增量训练数据目录（.npz）"
    )
    sp_pretrain.add_argument(
        "--incremental-raw-data-dir",
        type=str,
        help="增量训练原始文件目录（自动提取后再训练）",
    )
    sp_pretrain.add_argument(
        "--file-extensions",
        type=str,
        nargs="+",
        help="仅处理指定后缀文件，例如 .exe .dll",
    )
    sp_pretrain.add_argument(
        "--label-inference",
        type=str,
        default="filename",
        choices=["filename", "directory"],
        help="标签推断方式：filename 按文件名，directory 按目录名",
    )
    sp_pretrain.add_argument(
        "--num-boost-round",
        type=int,
        default=DEFAULT_NUM_BOOST_ROUND,
        help="预训练 boosting 轮数",
    )
    sp_pretrain.add_argument(
        "--incremental-rounds",
        type=int,
        default=DEFAULT_INCREMENTAL_ROUNDS,
        help="增量训练轮数",
    )
    sp_pretrain.add_argument(
        "--incremental-early-stopping",
        type=int,
        default=DEFAULT_INCREMENTAL_EARLY_STOPPING,
        help="增量训练早停轮数",
    )
    sp_pretrain.add_argument(
        "--max-finetune-iterations",
        type=int,
        default=DEFAULT_MAX_FINETUNE_ITERATIONS,
        help="强化训练最大迭代次数",
    )
    sp_pretrain.add_argument(
        "--use-existing-features",
        action="store_true",
        help="复用已有特征文件并跳过重新提取",
    )

    sp_finetune = subs.add_parser("finetune", help="执行 HDBSCAN 家族发现与分类器训练")
    sp_finetune.add_argument(
        "--data-dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help="处理后数据目录（.npz + metadata）",
    )
    sp_finetune.add_argument(
        "--features-path",
        type=str,
        default=FEATURES_PKL_PATH,
        help="已提取特征文件路径",
    )
    sp_finetune.add_argument(
        "--save-dir", type=str, default=HDBSCAN_SAVE_DIR, help="聚类与模型输出目录"
    )
    sp_finetune.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help="单个样本允许处理的最大字节数",
    )
    sp_finetune.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help="HDBSCAN 最小簇大小",
    )
    sp_finetune.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="HDBSCAN 核心点最小样本数",
    )
    sp_finetune.add_argument(
        "--min-family-size",
        type=int,
        default=DEFAULT_MIN_FAMILY_SIZE,
        help="家族最小样本数",
    )
    sp_finetune.add_argument(
        "--plot-pca", action="store_true", help="输出 PCA 可视化图"
    )
    sp_finetune.add_argument(
        "--explain-discrepancy", action="store_true", help="输出聚类差异解释信息"
    )
    sp_finetune.add_argument(
        "--treat-noise-as-family",
        action="store_true",
        default=DEFAULT_TREAT_NOISE_AS_FAMILY,
        help="将噪声点作为独立家族处理",
    )
    sp_finetune.add_argument(
        "--skip-cluster-quality-eval", action="store_true", help="跳过聚类质量评估"
    )

    sp_scan = subs.add_parser("scan", help="执行单文件或目录扫描")
    sp_scan.add_argument(
        "--lightgbm-model-path", type=str, default=MODEL_PATH, help="LightGBM 模型路径"
    )
    sp_scan.add_argument(
        "--family-classifier-path",
        type=str,
        default=FAMILY_CLASSIFIER_PATH,
        help="家族分类器模型路径",
    )
    sp_scan.add_argument(
        "--cache-file", type=str, default=SCAN_CACHE_PATH, help="扫描缓存文件路径"
    )
    sp_scan.add_argument("--file-path", type=str, help="待扫描的单文件路径")
    sp_scan.add_argument("--dir-path", type=str, help="待扫描的目录路径")
    sp_scan.add_argument("--recursive", action="store_true", help="递归扫描目录")
    sp_scan.add_argument(
        "--output-path", type=str, default=SCAN_OUTPUT_DIR, help="扫描结果输出目录"
    )
    sp_scan.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help="单个样本允许处理的最大字节数",
    )

    sp_extract = subs.add_parser("extract", help="从样本目录提取特征并生成处理数据")
    sp_extract.add_argument(
        "--output-dir", type=str, default=PROCESSED_DATA_DIR, help="提取结果输出目录"
    )
    sp_extract.add_argument(
        "--file-extensions",
        type=str,
        nargs="+",
        help="仅处理指定后缀文件，例如 .exe .dll",
    )
    sp_extract.add_argument(
        "--label-inference",
        type=str,
        default="directory",
        choices=["filename", "directory"],
        help="标签推断方式：filename 按文件名，directory 按目录名",
    )
    sp_extract.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help="单个样本允许处理的最大字节数",
    )
    sp_extract.add_argument(
        "--max-workers", type=int, default=16, help="特征提取并发线程上限"
    )

    subs.add_parser("serve", help="启动IPC扫描服务")

    sp_train_routing = subs.add_parser(
        "train-routing", help="训练路由门控与专家模型系统"
    )
    sp_train_routing.add_argument(
        "--use-existing-features",
        action="store_true",
        help="复用已有特征文件并跳过重新提取",
    )
    sp_train_routing.add_argument(
        "--save-features", action="store_true", help="保存本次提取的特征到特征文件"
    )
    sp_train_routing.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="快速开发模式，仅使用小规模样本进行流程验证",
    )
    sp_train_routing.add_argument(
        "--finetune-on-false-positives",
        action="store_true",
        help="检测到假阳性后继续进行强化训练",
    )
    sp_train_routing.add_argument(
        "--incremental-training",
        action="store_true",
        help="启用增量训练，基于已有模型继续训练",
    )
    sp_train_routing.add_argument(
        "--incremental-data-dir", type=str, help="增量训练数据目录（.npz）"
    )
    sp_train_routing.add_argument(
        "--incremental-raw-data-dir",
        type=str,
        help="增量训练原始文件目录（自动提取后再训练）",
    )
    sp_train_routing.add_argument(
        "--file-extensions",
        type=str,
        nargs="+",
        help="仅处理指定后缀文件，例如 .exe .dll",
    )
    sp_train_routing.add_argument(
        "--label-inference",
        type=str,
        default="filename",
        choices=["filename", "directory"],
        help="标签推断方式：filename 按文件名，directory 按目录名",
    )
    sp_train_routing.add_argument(
        "--num-boost-round",
        type=int,
        default=DEFAULT_NUM_BOOST_ROUND,
        help="预训练 boosting 轮数",
    )
    sp_train_routing.add_argument(
        "--incremental-rounds",
        type=int,
        default=DEFAULT_INCREMENTAL_ROUNDS,
        help="增量训练轮数",
    )
    sp_train_routing.add_argument(
        "--incremental-early-stopping",
        type=int,
        default=DEFAULT_INCREMENTAL_EARLY_STOPPING,
        help="增量训练早停轮数",
    )
    sp_train_routing.add_argument(
        "--max-finetune-iterations",
        type=int,
        default=DEFAULT_MAX_FINETUNE_ITERATIONS,
        help="强化训练最大迭代次数",
    )
    sp_train_routing.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help="单个样本允许处理的最大字节数",
    )

    sp_train_all = subs.add_parser(
        "train-all", help="一键执行特征提取、模型训练、评估与聚类"
    )
    sp_train_all.add_argument(
        "--finetune-on-false-positives",
        action="store_true",
        help="检测到假阳性后继续进行强化训练",
    )
    sp_train_all.add_argument(
        "--skip-tuning", action="store_true", help="跳过 AutoML 超参调优阶段"
    )
    sp_train_all.add_argument(
        "--skip-cluster-quality-eval", action="store_true", help="跳过聚类质量评估"
    )
    sp_train_all.add_argument(
        "--hardcase-gbdt-estimators", type=int, default=260, help="hardcase GBDT树数量"
    )
    sp_train_all.add_argument(
        "--hardcase-gbdt-learning-rate",
        type=float,
        default=0.05,
        help="hardcase GBDT学习率",
    )
    sp_train_all.add_argument(
        "--hardcase-gbdt-num-leaves",
        type=int,
        default=63,
        help="hardcase LightGBM叶子节点数",
    )
    sp_train_all.add_argument(
        "--hardcase-gbdt-max-depth",
        type=int,
        default=6,
        help="hardcase XGBoost最大深度",
    )
    sp_train_all.add_argument(
        "--hardcase-gbdt-subsample", type=float, default=0.9, help="hardcase 子采样比例"
    )
    sp_train_all.add_argument(
        "--hardcase-gbdt-colsample-bytree",
        type=float,
        default=0.9,
        help="hardcase 列采样比例",
    )
    sp_train_all.add_argument(
        "--hardcase-cascade-fn-threshold",
        type=float,
        default=0.35,
        help="hardcase 级联FN覆盖阈值",
    )
    sp_train_all.add_argument(
        "--hardcase-cascade-fn-margin",
        type=float,
        default=-0.02,
        help="hardcase 级联FN覆盖领先边际",
    )
    sp_train_all.add_argument(
        "--hardcase-bootstrap-rounds",
        type=int,
        default=300,
        help="hardcase 稳定性评估bootstrap轮数",
    )
    sp_train_all.add_argument(
        "--hardcase-val-size", type=float, default=0.2, help="hardcase 验证集比例"
    )
    sp_train_all.add_argument(
        "--hardcase-seed", type=int, default=42, help="hardcase 随机种子"
    )
    sp_train_all.add_argument(
        "--hardcase-max-input-dim",
        type=int,
        default=384,
        help="hardcase 筛选后的最大输入特征维数",
    )
    sp_train_all.add_argument(
        "--hardcase-max-samples-per-class",
        type=int,
        default=0,
        help="hardcase 每类样本上限，0 表示不限制",
    )

    sp_export_family_json = subs.add_parser(
        "export-family-json",
        help="从 family_classifier.pkl 快速导出 family_classifier.json",
    )
    sp_export_family_json.add_argument(
        "--input",
        type=str,
        default=os.path.join(HDBSCAN_SAVE_DIR, "family_classifier.pkl"),
    )
    sp_export_family_json.add_argument(
        "--output",
        type=str,
        default=os.path.join(HDBSCAN_SAVE_DIR, "family_classifier.json"),
    )

    sp_autotune = subs.add_parser(
        "auto-tune", help="执行 AutoML 超参调优并输出交叉验证对比"
    )
    sp_autotune.add_argument(
        "--method",
        type=str,
        default=AUTOML_METHOD_DEFAULT,
        choices=["optuna", "hyperopt"],
        help="超参调优方法，可选 optuna 或 hyperopt",
    )
    sp_autotune.add_argument(
        "--trials", type=int, default=AUTOML_TRIALS_DEFAULT, help="超参搜索试验次数"
    )
    sp_autotune.add_argument(
        "--cv", type=int, default=AUTOML_CV_FOLDS_DEFAULT, help="交叉验证折数"
    )
    sp_autotune.add_argument(
        "--metric",
        type=str,
        default=AUTOML_METRIC_DEFAULT,
        choices=["roc_auc", "accuracy", "f1", "precision", "recall"],
        help="优化目标指标",
    )
    sp_autotune.add_argument(
        "--use-existing-features",
        action="store_true",
        help="复用已有特征文件并跳过重新提取",
    )
    sp_autotune.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="快速开发模式，仅使用小规模样本进行流程验证",
    )
    sp_autotune.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help="单个样本允许处理的最大字节数",
    )

    sp_train_hardcase_dl = subs.add_parser(
        "train-hardcase-dl", help="训练困难样本/假阳性/假阴性专用GBDT级联模型"
    )
    sp_train_hardcase_dl.add_argument(
        "--gbdt-estimators", type=int, default=260, help="GBDT树数量"
    )
    sp_train_hardcase_dl.add_argument(
        "--gbdt-learning-rate", type=float, default=0.05, help="GBDT学习率"
    )
    sp_train_hardcase_dl.add_argument(
        "--gbdt-num-leaves", type=int, default=63, help="LightGBM叶子节点数"
    )
    sp_train_hardcase_dl.add_argument(
        "--gbdt-max-depth", type=int, default=6, help="XGBoost最大深度"
    )
    sp_train_hardcase_dl.add_argument(
        "--gbdt-subsample", type=float, default=0.9, help="子采样比例"
    )
    sp_train_hardcase_dl.add_argument(
        "--gbdt-colsample-bytree", type=float, default=0.9, help="列采样比例"
    )
    sp_train_hardcase_dl.add_argument(
        "--cascade-fn-threshold", type=float, default=0.35, help="级联时FN覆盖阈值"
    )
    sp_train_hardcase_dl.add_argument(
        "--cascade-fn-margin", type=float, default=-0.02, help="级联时FN覆盖领先边际"
    )
    sp_train_hardcase_dl.add_argument(
        "--bootstrap-rounds", type=int, default=300, help="稳定性评估bootstrap轮数"
    )
    sp_train_hardcase_dl.add_argument(
        "--val-size", type=float, default=0.2, help="验证集比例"
    )
    sp_train_hardcase_dl.add_argument("--seed", type=int, default=42, help="随机种子")
    sp_train_hardcase_dl.add_argument(
        "--max-input-dim", type=int, default=384, help="筛选后的最大输入特征维数"
    )
    sp_train_hardcase_dl.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="每类样本上限，0 表示不限制",
    )
    subs.add_parser("hardcase-model-trials", help="执行 hardcase A/B/C 三方案对比试验")
    sp_convert_weights_onnx = subs.add_parser(
        "convert-weights-onnx", help="将权重目录中的可转换模型统一导出为ONNX"
    )
    sp_convert_weights_onnx.add_argument(
        "--weights-dir", type=str, default=SAVED_MODEL_DIR, help="权重目录路径"
    )

    args = parser.parse_args()

    if args.command == "pretrain":
        import pretrain

        try:
            pretrain.main(args)
            _ensure_onnx_artifacts_after_training(
                logger,
                keep_train_txt=bool(getattr(args, "incremental_training", False)),
            )
        except Exception as e:
            logger.error(f"预训练失败: {e}")
            raise
    elif args.command == "finetune":
        import finetune

        try:
            finetune.main(args)
        except Exception as e:
            logger.error(f"微调失败: {e}")
            raise
    elif args.command == "scan":
        try:
            if args.file_path:
                if not os.path.exists(args.file_path):
                    logger.error(f"文件不存在: {args.file_path}")
                    return
                paths = [args.file_path]
            elif args.dir_path:
                if not os.path.exists(args.dir_path):
                    logger.error(f"目录不存在: {args.dir_path}")
                    return
                paths = _collect_scan_paths(args.dir_path, args.recursive)
            else:
                logger.error("请指定 --file-path 或 --dir-path")
                return
            dll = _load_kvd_scan_dll()
            if dll is None:
                raise RuntimeError("未找到可用的扫描DLL")
            cfg = _build_kvd_config(
                args.lightgbm_model_path,
                args.family_classifier_path,
                args.max_file_size,
            )
            handle = dll.kvd_create(ctypes.byref(cfg))
            if not handle:
                raise RuntimeError("kvd_create failed")
            try:
                if not paths:
                    results = []
                elif hasattr(dll, "kvd_scan_paths") and len(paths) > 1:
                    results = _kvd_scan_paths(dll, handle, paths)
                else:
                    results = [_kvd_scan_path(dll, handle, p) for p in paths]
            finally:
                dll.kvd_destroy(handle)
            results = _enrich_scan_results(results, paths)
            _save_scan_results(results, args.output_path, logger)
            malicious_paths = [
                r.get("file_path") for r in results if r.get("is_malware")
            ]
            report_dir = os.path.dirname(DETECTED_MALICIOUS_PATHS_REPORT_PATH)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
            with open(DETECTED_MALICIOUS_PATHS_REPORT_PATH, "w", encoding="utf-8") as f:
                for p in malicious_paths:
                    if p:
                        f.write(p + "\n")
            logger.info(
                f"恶意样本路径已保存: {DETECTED_MALICIOUS_PATHS_REPORT_PATH}，数量: {len(malicious_paths)}"
            )
        except Exception as e:
            logger.error(f"扫描失败: {e}")
            raise
    elif args.command == "extract":
        from training.data_loader import extract_features_from_raw_files

        try:
            sources = [BENIGN_SAMPLES_DIR, MALICIOUS_SAMPLES_DIR]
            all_files = []
            all_labels = []
            for src in sources:
                file_names, labels = extract_features_from_raw_files(
                    src,
                    args.output_dir,
                    args.max_file_size,
                    args.file_extensions,
                    args.label_inference,
                    args.max_workers,
                )
                all_files.extend(file_names)
                all_labels.extend(labels)
            if all_files:
                import json

                os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
                mapping = {
                    fn: ("benign" if lab == 0 else "malicious")
                    for fn, lab in zip(all_files, all_labels)
                }
                with open(METADATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
                logger.info(f"已生成元数据: {METADATA_FILE}，样本数: {len(all_files)}")
        except Exception as e:
            logger.error(f"提取失败: {e}")
            raise
    elif args.command == "serve":
        try:
            _serve_ipc_only()
        except Exception as e:
            logger.error(f"服务启动失败: {e}")
            raise
    elif args.command == "train-routing":
        from training import train_routing

        try:
            train_routing.main(args)
            _ensure_onnx_artifacts_after_training(
                logger,
                keep_train_txt=bool(getattr(args, "incremental_training", False)),
            )
        except Exception as e:
            logger.error(f"路由系统训练失败: {e}")
            raise
    elif args.command == "train-hardcase-dl":
        from training import hardcase_dl

        try:
            hardcase_dl.main(args)
            _ensure_onnx_artifacts_after_training(
                logger,
                keep_train_txt=bool(getattr(args, "incremental_training", False)),
            )
        except Exception as e:
            logger.error(f"困难样本深度学习训练失败: {e}")
            raise
    elif args.command == "hardcase-model-trials":
        from training import hardcase_model_trials

        try:
            hardcase_model_trials.main(args)
        except Exception as e:
            logger.error(f"hardcase 模型对比试验失败: {e}")
            raise
    elif args.command == "train-all":
        required_deps = ["numpy", "pandas", "sklearn", "lightgbm"]
        missing_deps = [
            name for name in required_deps if importlib.util.find_spec(name) is None
        ]
        if missing_deps:
            logger.error(
                f'缺少依赖: {", ".join(missing_deps)}，请先安装后再运行 train-all'
            )
            return
        import pretrain

        torch_available = importlib.util.find_spec("torch") is not None
        if torch_available:
            from training import train_routing
            from training import hardcase_dl
            import finetune
        else:
            train_routing = None
            hardcase_dl = None
            finetune = None
        try:
            if not os.path.exists(METADATA_FILE):
                logger.info(
                    f"[*] 元数据文件不存在: {METADATA_FILE}，将自动开始特征提取..."
                )
                from training.data_loader import extract_features_from_raw_files
                import json

                sources = [BENIGN_SAMPLES_DIR, MALICIOUS_SAMPLES_DIR]
                all_files = []
                all_labels = []
                for src in sources:
                    if os.path.exists(src):
                        file_names, labels = extract_features_from_raw_files(
                            src,
                            PROCESSED_DATA_DIR,
                            DEFAULT_MAX_FILE_SIZE,
                            None,
                            "filename",
                        )
                        all_files.extend(file_names)
                        all_labels.extend(labels)
                if all_files:
                    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
                    mapping = {
                        fn: ("benign" if lab == 0 else "malicious")
                        for fn, lab in zip(all_files, all_labels)
                    }
                    with open(METADATA_FILE, "w", encoding="utf-8") as f:
                        json.dump(mapping, f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"已自动生成元数据: {METADATA_FILE}，样本数: {len(all_files)}"
                    )
                else:
                    logger.warning("未找到原始样本，跳过自动提取。")

            pre_args = argparse.Namespace(
                max_file_size=DEFAULT_MAX_FILE_SIZE,
                fast_dev_run=False,
                save_features=True,
                finetune_on_false_positives=args.finetune_on_false_positives,
                incremental_training=False,
                incremental_data_dir=None,
                incremental_raw_data_dir=None,
                file_extensions=None,
                label_inference="filename",
                num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                incremental_rounds=DEFAULT_INCREMENTAL_ROUNDS,
                incremental_early_stopping=DEFAULT_INCREMENTAL_EARLY_STOPPING,
                max_finetune_iterations=DEFAULT_MAX_FINETUNE_ITERATIONS,
                use_existing_features=True,
            )
            pretrain.main(pre_args)
            from training import automl

            override_params = None
            if not args.skip_tuning and os.path.exists(FEATURES_PKL_PATH):
                auto_args = argparse.Namespace(
                    method=AUTOML_METHOD_DEFAULT,
                    trials=AUTOML_TRIALS_DEFAULT,
                    cv=AUTOML_CV_FOLDS_DEFAULT,
                    metric=AUTOML_METRIC_DEFAULT,
                    use_existing_features=True,
                    fast_dev_run=True,
                    max_file_size=DEFAULT_MAX_FILE_SIZE,
                )
                auto_result = automl.main(auto_args)
                override_params = auto_result.get("best_params", None)
                if override_params:
                    pre_args2 = argparse.Namespace(
                        max_file_size=DEFAULT_MAX_FILE_SIZE,
                        fast_dev_run=False,
                        save_features=True,
                        finetune_on_false_positives=args.finetune_on_false_positives,
                        incremental_training=False,
                        incremental_data_dir=None,
                        incremental_raw_data_dir=None,
                        file_extensions=None,
                        label_inference="filename",
                        num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                        incremental_rounds=DEFAULT_INCREMENTAL_ROUNDS,
                        incremental_early_stopping=DEFAULT_INCREMENTAL_EARLY_STOPPING,
                        max_finetune_iterations=DEFAULT_MAX_FINETUNE_ITERATIONS,
                        use_existing_features=True,
                        override_params=override_params,
                    )
                    pretrain.main(pre_args2)
            sample_report_payload = _export_train_all_sample_reports(logger)
            hardcase_args = argparse.Namespace(
                gbdt_estimators=args.hardcase_gbdt_estimators,
                gbdt_learning_rate=args.hardcase_gbdt_learning_rate,
                gbdt_num_leaves=args.hardcase_gbdt_num_leaves,
                gbdt_max_depth=args.hardcase_gbdt_max_depth,
                gbdt_subsample=args.hardcase_gbdt_subsample,
                gbdt_colsample_bytree=args.hardcase_gbdt_colsample_bytree,
                cascade_fn_threshold=args.hardcase_cascade_fn_threshold,
                cascade_fn_margin=args.hardcase_cascade_fn_margin,
                bootstrap_rounds=args.hardcase_bootstrap_rounds,
                val_size=args.hardcase_val_size,
                seed=args.hardcase_seed,
                max_input_dim=args.hardcase_max_input_dim,
                max_samples_per_class=args.hardcase_max_samples_per_class,
            )
            deep_engine_report_path = None
            if torch_available:
                hardcase_metrics = hardcase_dl.main(hardcase_args)
                deep_engine_report_path = _export_train_all_deep_engine_eval_report(
                    logger, hardcase_metrics
                )
                routing_args = argparse.Namespace(
                    use_existing_features=True,
                    save_features=False,
                    fast_dev_run=False,
                    incremental_training=False,
                    incremental_data_dir=None,
                    incremental_raw_data_dir=None,
                    file_extensions=None,
                    label_inference="filename",
                    num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                    incremental_rounds=DEFAULT_INCREMENTAL_ROUNDS,
                    incremental_early_stopping=DEFAULT_INCREMENTAL_EARLY_STOPPING,
                    max_finetune_iterations=DEFAULT_MAX_FINETUNE_ITERATIONS,
                    max_file_size=DEFAULT_MAX_FILE_SIZE,
                )
                if override_params:
                    routing_args.override_params = override_params
                train_routing.main(routing_args)
                fine_args = argparse.Namespace(
                    data_dir=PROCESSED_DATA_DIR,
                    features_path=FEATURES_PKL_PATH,
                    save_dir=HDBSCAN_SAVE_DIR,
                    max_file_size=DEFAULT_MAX_FILE_SIZE,
                    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
                    min_samples=DEFAULT_MIN_SAMPLES,
                    min_family_size=DEFAULT_MIN_FAMILY_SIZE,
                    plot_pca=False,
                    explain_discrepancy=False,
                    treat_noise_as_family=DEFAULT_TREAT_NOISE_AS_FAMILY,
                    skip_cluster_quality_eval=args.skip_cluster_quality_eval,
                )
                finetune.main(fine_args)
            else:
                logger.warning("未检测到 torch，已跳过路由训练、深度学习与聚类步骤")
            _merge_train_all_evaluation_summary(
                logger,
                deep_engine_report_path=deep_engine_report_path,
                sample_report_payload=sample_report_payload,
            )
            _ensure_onnx_artifacts_after_training(logger, keep_train_txt=True)
            if torch_available:
                logger.info("训练与聚类流程已完成")
            else:
                logger.info("训练流程已完成（部分步骤跳过）")
        except Exception as e:
            logger.error(f"一键训练失败: {e}")
            raise
    elif args.command == "export-family-json":
        from training.export_family_classifier_json import export_family_classifier
        from pathlib import Path

        try:
            export_family_classifier(Path(args.input), Path(args.output))
            logger.info(f"已导出 family_classifier.json: {args.output}")
        except Exception as e:
            logger.error(f"导出 family_classifier.json 失败: {e}")
            raise
    elif args.command == "auto-tune":
        from training import automl

        try:
            result = automl.main(args)
            logger.info(f"AutoML完成: {result}")
        except Exception as e:
            logger.error(f"AutoML失败: {e}")
            raise
    elif args.command == "convert-weights-onnx":
        try:
            _convert_all_weights_to_onnx(args.weights_dir, logger)
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            raise


if __name__ == "__main__":
    entry_logger = get_logger("kolo.entry")
    try:
        main()
    except Exception as e:
        entry_logger.error(f"主程序执行失败: {e}", exc_info=True)
        raise
