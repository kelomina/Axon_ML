import os
import json
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from settings import STACKING_FOLDS, ENSEMBLE_SEEDS, LIGHTGBM_NUM_ROUNDS, LIGHTGBM_EARLY_STOPPING, COST_POS_WEIGHT, COST_NEG_WEIGHT, POS_WEIGHT_SCALE, HARD_NEGATIVE_WEIGHT, MODEL_DIR

def _slice_array(X, slc):
    if slc is None:
        return X
    return X[:, slc]

def _train_lgb(X_train, y_train, X_val, y_val, seed, sample_weights=None, params_override=None, num_boost_round=None):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 40,
        "min_gain_to_split": 0.0,
        "seed": seed,
        "verbose": -1
    }
    if params_override:
        params.update(params_override)
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=LIGHTGBM_NUM_ROUNDS if num_boost_round is None else num_boost_round,
        callbacks=[lgb.early_stopping(LIGHTGBM_EARLY_STOPPING, verbose=False)]
    )
    return model

def _train_lr(X_train, y_train):
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    base = (n_neg / max(1, n_pos)) * POS_WEIGHT_SCALE
    pos_weight = max(0.01, base)
    class_weight = {0: COST_NEG_WEIGHT, 1: COST_POS_WEIGHT * pos_weight}
    model = LogisticRegression(max_iter=200, solver="liblinear", class_weight=class_weight)
    model.fit(X_train, y_train)
    return model

def _predict_model(model, X, model_type):
    if model_type == "lgb":
        return model.predict(X)
    return model.predict_proba(X)[:, 1]

class EnsembleModel:
    def __init__(self, base_models, meta_model, base_dim, total_dim):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_dim = base_dim
        self.total_dim = total_dim

    def _base_predictions(self, X):
        preds = []
        for item in self.base_models:
            slc = item["slice"]
            X_sub = _slice_array(X, slc)
            preds.append(_predict_model(item["model"], X_sub, item["type"]))
        return np.vstack(preds).T

    def predict_proba(self, X):
        meta_X = self._base_predictions(X)
        return self.meta_model.predict_proba(meta_X)[:, 1]

    def save(self, model_dir, prefix):
        os.makedirs(model_dir, exist_ok=True)
        meta_path = os.path.join(model_dir, f"{prefix}_meta.joblib")
        joblib.dump(self.meta_model, meta_path)
        meta = {
            "base_dim": int(self.base_dim),
            "total_dim": int(self.total_dim),
            "models": []
        }
        for idx, item in enumerate(self.base_models):
            name = item["name"]
            mtype = item["type"]
            slc = item["slice"]
            if mtype == "lgb":
                path = os.path.join(model_dir, f"{prefix}_{name}_{idx}.txt")
                item["model"].save_model(path)
            else:
                path = os.path.join(model_dir, f"{prefix}_{name}_{idx}.joblib")
                joblib.dump(item["model"], path)
            meta["models"].append({
                "name": name,
                "type": mtype,
                "path": path,
                "slice": [slc.start if slc else None, slc.stop if slc else None]
            })
        with open(os.path.join(model_dir, f"{prefix}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(model_dir, prefix):
        meta_path = os.path.join(model_dir, f"{prefix}_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta_model = EnsembleModel._secure_joblib_load(os.path.join(model_dir, f"{prefix}_meta.joblib"))
        base_models = []
        for item in meta["models"]:
            slc = None
            if item["slice"][0] is not None:
                slc = slice(item["slice"][0], item["slice"][1])
            if item["type"] == "lgb":
                model = lgb.Booster(model_file=item["path"])
            else:
                model = EnsembleModel._secure_joblib_load(item["path"])
            base_models.append({
                "name": item["name"],
                "type": item["type"],
                "slice": slc,
                "model": model
            })
        return EnsembleModel(base_models, meta_model, meta["base_dim"], meta["total_dim"])

    @staticmethod
    def _secure_joblib_load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        abs_path = os.path.abspath(path)
        cwd = os.getcwd()
        if not (abs_path.startswith(cwd + os.sep) or abs_path == cwd):
            raise RuntimeError(f"Model path outside allowed directory: {path}")
        try:
            import joblib
            data = joblib.load(abs_path)
            if data is None:
                raise ValueError("Loaded model is None")
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to securely load model from {path}: {e}")

def _compute_sample_weights(y, files=None, hard_negative_set=None):
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    base = (n_neg / max(1, n_pos)) * POS_WEIGHT_SCALE
    pos_weight = max(0.01, base)
    weights = np.where(y == 1, COST_POS_WEIGHT * pos_weight, COST_NEG_WEIGHT).astype(np.float32)
    if files is not None and hard_negative_set:
        for i, f in enumerate(files):
            if y[i] == 0 and f in hard_negative_set:
                weights[i] = weights[i] * HARD_NEGATIVE_WEIGHT
    return weights

def train_ensemble(X_train, y_train, X_val, y_val, base_dim, total_dim, prefix, params_override=None, files_train=None, hard_negative_set=None, num_boost_round=None):
    base_slice = slice(0, base_dim)
    ngram_slice = slice(base_dim, total_dim) if total_dim > base_dim else None
    model_specs = []
    for seed in ENSEMBLE_SEEDS:
        model_specs.append(("lgb_full_" + str(seed), "lgb", None, seed))
        model_specs.append(("lgb_base_" + str(seed), "lgb", base_slice, seed))
    if ngram_slice is not None:
        model_specs.append(("lr_ngram", "lr", ngram_slice, None))
    kfold = StratifiedKFold(n_splits=STACKING_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X_train), len(model_specs)), dtype=np.float32)
    for m_idx, (name, mtype, slc, seed) in enumerate(model_specs):
        for train_idx, val_idx in kfold.split(X_train, y_train):
            X_tr = _slice_array(X_train[train_idx], slc)
            y_tr = y_train[train_idx]
            X_va = _slice_array(X_train[val_idx], slc)
            y_va = y_train[val_idx]
            if mtype == "lgb":
                files_tr = [files_train[i] for i in train_idx] if files_train is not None else None
                weights = _compute_sample_weights(y_tr, files=files_tr, hard_negative_set=hard_negative_set)
                model = _train_lgb(X_tr, y_tr, X_va, y_va, seed, sample_weights=weights, params_override=params_override, num_boost_round=num_boost_round)
                pred = _predict_model(model, X_va, "lgb")
            else:
                model = _train_lr(X_tr, y_tr)
                pred = _predict_model(model, X_va, "lr")
            oof[val_idx, m_idx] = pred
    meta_model = LogisticRegression(max_iter=200, solver="liblinear")
    meta_model.fit(oof, y_train)
    base_models = []
    for name, mtype, slc, seed in model_specs:
        X_tr = _slice_array(X_train, slc)
        if mtype == "lgb":
            weights = _compute_sample_weights(y_train, files=files_train, hard_negative_set=hard_negative_set)
            model = _train_lgb(X_tr, y_train, _slice_array(X_val, slc), y_val, seed, sample_weights=weights, params_override=params_override, num_boost_round=num_boost_round)
        else:
            model = _train_lr(X_tr, y_train)
        base_models.append({
            "name": name,
            "type": mtype,
            "slice": slc,
            "model": model
        })
    ensemble = EnsembleModel(base_models, meta_model, base_dim, total_dim)
    ensemble.save(MODEL_DIR, prefix)
    return ensemble
