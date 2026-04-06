import os
import json
import numpy as np

def _try_import_onnxruntime():
    try:
        import onnxruntime as ort
        return ort
    except Exception:
        return None

class OnnxPredictor:
    def __init__(self, model_path, feature_names, providers=None, expected_hash=None):
        self.model_path = model_path
        self.feature_names = feature_names
        self.providers = providers
        self.session = None
        self.input_name = None
        self.index_map = None
        ort = _try_import_onnxruntime()
        if ort is None:
            return
        if not os.path.exists(model_path):
            return
        if expected_hash is not None:
            if not self._verify_file_hash(model_path, expected_hash):
                print(f"[Warning] ONNX model hash mismatch for {model_path}")
                return
        self.session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        feat_path = os.path.join(os.path.dirname(model_path), "features.json")
        if os.path.exists(feat_path):
            with open(feat_path, "r", encoding="utf-8") as f:
                model_features = json.load(f)
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            self.index_map = [name_to_idx.get(n) for n in model_features]

    @staticmethod
    def _verify_file_hash(path, expected_hash):
        import hashlib
        try:
            with open(path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash == expected_hash
        except Exception:
            return False

    def available(self):
        return self.session is not None and self.input_name is not None

    def _reorder(self, X):
        if self.index_map is None:
            return X
        out = np.zeros((X.shape[0], len(self.index_map)), dtype=np.float32)
        for i, idx in enumerate(self.index_map):
            if idx is None:
                continue
            out[:, i] = X[:, idx]
        return out

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = self._reorder(X)
        outputs = self.session.run(None, {self.input_name: X})
        if isinstance(outputs, list) and len(outputs) > 1:
            result = outputs[1]
        else:
            result = outputs[0]
        if isinstance(result, list) and len(result) > 0 and hasattr(result[0], "get"):
            return np.array([float(r.get(1, r.get("1", 0.0))) for r in result], dtype=np.float32)
        if isinstance(result, np.ndarray):
            if result.ndim == 2 and result.shape[1] > 1:
                return result[:, 1].astype(np.float32)
            return result.reshape(-1).astype(np.float32)
        return np.zeros(X.shape[0], dtype=np.float32)
