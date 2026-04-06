import json

files = [
    "/workspace/src/python/kvd_detector/cli.py",
    "/workspace/src/python/kvd_detector/collect_benign_pe.py",
    "/workspace/src/python/kvd_detector/config/__init__.py",
    "/workspace/src/python/kvd_detector/config/config.py",
    "/workspace/src/python/kvd_detector/dataset_v2.py",
    "/workspace/src/python/kvd_detector/ensemble.py",
    "/workspace/src/python/kvd_detector/feature_enhancer.py",
    "/workspace/src/python/kvd_detector/feature_extractor_enhanced.py",
    "/workspace/src/python/kvd_detector/features/__init__.py",
    "/workspace/src/python/kvd_detector/features/extractor_in_memory.py",
    "/workspace/src/python/kvd_detector/features/extractor_save.py",
    "/workspace/src/python/kvd_detector/features/statistics.py",
    "/workspace/src/python/kvd_detector/finetune.py",
    "/workspace/src/python/kvd_detector/gating_v2.py",
    "/workspace/src/python/kvd_detector/hard_negative.py",
    "/workspace/src/python/kvd_detector/main.py",
    "/workspace/src/python/kvd_detector/models/__init__.py",
    "/workspace/src/python/kvd_detector/models/family_classifier.py",
    "/workspace/src/python/kvd_detector/models/gating.py",
    "/workspace/src/python/kvd_detector/models/routing_model.py",
    "/workspace/src/python/kvd_detector/onnx_backend.py",
    "/workspace/src/python/kvd_detector/pipeline.py",
    "/workspace/src/python/kvd_detector/pretrain.py",
    "/workspace/src/python/kvd_detector/scanner.py",
    "/workspace/src/python/kvd_detector/scanner_service.py",
    "/workspace/src/python/kvd_detector/scanner_v2.py",
    "/workspace/src/python/kvd_detector/settings.py",
    "/workspace/src/python/kvd_detector/threshold.py",
    "/workspace/src/python/kvd_detector/training/__init__.py",
    "/workspace/src/python/kvd_detector/training/automl.py",
    "/workspace/src/python/kvd_detector/training/data_loader.py",
    "/workspace/src/python/kvd_detector/training/evaluate.py",
    "/workspace/src/python/kvd_detector/training/export_family_classifier_json.py",
    "/workspace/src/python/kvd_detector/training/feature_io.py",
    "/workspace/src/python/kvd_detector/training/hardcase_dl.py",
    "/workspace/src/python/kvd_detector/training/hardcase_model_trials.py",
    "/workspace/src/python/kvd_detector/training/incremental.py",
    "/workspace/src/python/kvd_detector/training/model_io.py",
    "/workspace/src/python/kvd_detector/training/train_lightgbm.py",
    "/workspace/src/python/kvd_detector/training/train_routing.py",
    "/workspace/src/python/kvd_detector/utils/__init__.py",
    "/workspace/src/python/kvd_detector/utils/logging_utils.py",
    "/workspace/src/python/kvd_detector/utils/path_utils.py",
    "/workspace/src/python/kvd_detector/validation/__init__.py",
    "/workspace/src/python/kvd_detector/validation/feature_gating_experiment.py",
    "/workspace/src/python/kvd_detector/validation/gating_validator.py"
]

def get_desc(path):
    if path.endswith("main.py"):
        return "- Removed over 9000 lines of embedded code strings<br>- Refactored to import from the newly created modular structure"
    elif path.endswith("__init__.py"):
        return "- Added initialization file to establish the module package"
    else:
        name = path.split("/")[-1]
        name_no_ext = name.replace(".py", "")
        formatted = name_no_ext.replace("_", " ").title()
        return f"- Extracted {formatted} module from the monolithic main.py into a standalone file"

for f in files:
    print(f"| {f} | {get_desc(f)} |")
