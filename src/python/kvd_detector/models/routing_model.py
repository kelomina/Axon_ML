import os
import numpy as np
import lightgbm as lgb
from features.extractor_in_memory import PE_FEATURE_ORDER
from config.config import (
    GATING_MODE, EXPERT_NORMAL_MODEL_PATH, EXPERT_PACKED_MODEL_PATH, PE_FEATURE_VECTOR_DIM,
    PACKED_SECTIONS_RATIO_THRESHOLD, PACKER_KEYWORD_HITS_THRESHOLD
)

class RoutingModel:
    def __init__(self):
        self.expert_normal = None
        self.expert_packed = None
        self._idx_packed_sections = self._feature_index('packed_sections_ratio')
        self._idx_packer_hits = self._feature_index('packer_keyword_hits_count')
        self.load_models()

    def load_models(self):
        if GATING_MODE != 'rule':
            print(f"[!] GATING_MODE={GATING_MODE} 非 'rule'，将使用规则门控以避免依赖 torch")
        try:
            if os.path.exists(EXPERT_NORMAL_MODEL_PATH):
                print(f"[*] Loading Normal Expert from {EXPERT_NORMAL_MODEL_PATH}")
                self.expert_normal = lgb.Booster(model_file=EXPERT_NORMAL_MODEL_PATH)
            else:
                print(f"[!] Normal expert model not found at {EXPERT_NORMAL_MODEL_PATH}")
        except Exception as e:
            print(f"[!] Failed to load Normal Expert: {e}")
            self.expert_normal = None
        try:
            if os.path.exists(EXPERT_PACKED_MODEL_PATH):
                print(f"[*] Loading Packed Expert from {EXPERT_PACKED_MODEL_PATH}")
                self.expert_packed = lgb.Booster(model_file=EXPERT_PACKED_MODEL_PATH)
            else:
                print(f"[!] Packed expert model not found at {EXPERT_PACKED_MODEL_PATH}")
        except Exception as e:
            print(f"[!] Failed to load Packed Expert: {e}")
            self.expert_packed = None

    def predict(self, features):
        x = np.asarray(features)
        routing_decisions = self._rule_gating(x)
        predictions = np.zeros(len(x))
        normal_indices = np.where(routing_decisions == 0)[0]
        packed_indices = np.where(routing_decisions == 1)[0]
        if len(normal_indices) > 0:
            if self.expert_normal:
                X_normal = x[normal_indices]
                pred_normal = self.expert_normal.predict(X_normal)
                predictions[normal_indices] = pred_normal
            else:
                print("[!] Expert Normal not loaded, skipping predictions for normal samples.")
        if len(packed_indices) > 0:
            if self.expert_packed:
                X_packed = x[packed_indices]
                pred_packed = self.expert_packed.predict(X_packed)
                predictions[packed_indices] = pred_packed
            else:
                print("[!] Expert Packed not loaded, skipping predictions for packed samples.")
        return predictions, routing_decisions

    def _feature_index(self, key):
        try:
            idx = 256 + PE_FEATURE_ORDER.index(key)
            return idx
        except ValueError:
            print(f"[Warning] Feature key '{key}' not found in PE_FEATURE_ORDER, routing may not work correctly")
            return None

    def _rule_gating(self, x):
        start = x.shape[1] - PE_FEATURE_VECTOR_DIM
        p = self._idx_packed_sections
        k = self._idx_packer_hits
        if p is None or k is None:
            return np.zeros(len(x), dtype=int)
        ps = x[:, start + p]
        kh = x[:, start + k]
        return np.logical_or(
            ps > PACKED_SECTIONS_RATIO_THRESHOLD,
            kh > PACKER_KEYWORD_HITS_THRESHOLD
        ).astype(int)

    def get_routing_stats(self, routing_decisions):
        total = len(routing_decisions)
        packed_count = np.sum(routing_decisions)
        normal_count = total - packed_count
        return {
            'total': total,
            'normal': normal_count,
            'packed': packed_count,
            'packed_ratio': packed_count / total if total > 0 else 0
        }
