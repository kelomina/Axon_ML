import os
import json
import numpy as np
from features.extractor_in_memory import extract_features_in_memory
from feature_enhancer import build_feature_vector, get_base_dim
from pipeline import load_system
from settings import MAX_FILE_SIZE

class MalwareScannerV2:
    def __init__(self):
        self.gating_model, self.normal_artifacts, self.packed_artifacts = load_system()
        self.base_dim = get_base_dim()

    def _predict(self, features):
        gating_model, normal_artifacts, packed_artifacts = self.gating_model, self.normal_artifacts, self.packed_artifacts
        normal_model, normal_cal, normal_th = normal_artifacts
        packed_model, packed_cal, packed_th = packed_artifacts
        x = features.reshape(1, -1)
        x_base = x[:, :self.base_dim]
        from gating_v2 import predict_gating
        g = predict_gating(gating_model, x_base)[0]
        if g >= 0.5:
            prob = packed_model.predict_proba(x)[0]
            prob = packed_cal.predict(np.array([prob]))[0]
            is_malware = prob >= packed_th
        else:
            prob = normal_model.predict_proba(x)[0]
            prob = normal_cal.predict(np.array([prob]))[0]
            is_malware = prob >= normal_th
        confidence = float(prob if is_malware else (1 - prob))
        return bool(is_malware), confidence

    def scan_file(self, file_path):
        byte_sequence, pe_features, orig_length = extract_features_in_memory(file_path, MAX_FILE_SIZE)
        if byte_sequence is None or pe_features is None:
            return None
        features = build_feature_vector(byte_sequence, pe_features, orig_length)
        is_malware, confidence = self._predict(features)
        return {
            "file_path": os.path.abspath(file_path),
            "is_malware": is_malware,
            "confidence": float(confidence)
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    scanner = MalwareScannerV2()
    result = scanner.scan_file(args.file)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
