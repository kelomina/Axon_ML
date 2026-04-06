import os
import json
import random
import re
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import lightgbm as lgb
import xgboost as xgb
from config.config import HDBSCAN_SAVE_DIR, RESOURCES_DIR, FEATURES_PKL_PATH

LABEL_TO_ID = {'hard_samples': 0, 'false_positives': 1, 'false_negatives': 2}
ID_TO_LABEL = {0: 'hard_samples', 1: 'false_positives', 2: 'false_negatives'}
ID_TO_LABEL_ZH = {0: '原区分困难', 1: '原假阳性', 2: '原假阴性'}

class HardCaseNet(nn.Module):
    def __init__(self, input_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)

class HardCaseResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class HardCaseResNet(nn.Module):
    def __init__(self, input_dim, dropout=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.res1 = HardCaseResBlock(512, dropout=dropout)
        self.down1 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout))
        self.res2 = HardCaseResBlock(256, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout * 0.8), nn.Linear(128, 3))

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        return self.head(x)

def _build_hardcase_model(args, input_dim):
    arch = str(getattr(args, 'model_arch', 'resmlp')).lower()
    if arch == 'mlp':
        return HardCaseNet(input_dim=input_dim, dropout=args.dropout), 'HardCaseNet'
    return HardCaseResNet(input_dim=input_dim, dropout=args.dropout), 'HardCaseResNet'
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _latest_report(pattern):
    base_dir = Path(HDBSCAN_SAVE_DIR)
    files = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def _read_json(path):
    if path is None or not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []

def _feature_idx(name):
    m = re.search(r'(\d+)$', str(name))
    if not m:
        return None
    return int(m.group(1))

def _collect_records():
    sources = [
        ('hard_samples', 'hard_samples_*.json'),
        ('false_positives', 'false_positives_*.json'),
        ('false_negatives', 'false_negatives_*.json'),
    ]
    grouped = {}
    for name, pattern in sources:
        grouped[name] = _read_json(_latest_report(pattern))
    return grouped

def _build_dataset(records_by_type, max_samples_per_class=0):
    max_idx = -1
    for records in records_by_type.values():
        for item in records:
            fmap = item.get('feature_importance', {}) or {}
            for k in fmap.keys():
                idx = _feature_idx(k)
                if idx is not None and idx > max_idx:
                    max_idx = idx
    if max_idx < 0:
        return None, None, {}
    feature_dim = int(max_idx + 2)
    X = []
    y = []
    class_counts = {}
    for label_name, records in records_by_type.items():
        if not records:
            continue
        if max_samples_per_class and max_samples_per_class > 0:
            records = records[:max_samples_per_class]
        class_counts[label_name] = len(records)
        for item in records:
            vec = np.zeros(feature_dim, dtype=np.float32)
            fmap = item.get('feature_importance', {}) or {}
            for k, v in fmap.items():
                idx = _feature_idx(k)
                if idx is None or idx < 0 or idx >= feature_dim - 1:
                    continue
                vec[idx] = float(abs(float(v)))
            vec[feature_dim - 1] = float(item.get('prediction_probability', 0.0))
            X.append(vec)
            y.append(int(LABEL_TO_ID[label_name]))
    if not X:
        return None, None, class_counts
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), class_counts

def _normalize_sample_name(v):
    return str(v).replace('\\', '/').lower()

def _build_dataset_from_features_pkl(records_by_type, max_samples_per_class=0):
    pkl_path = Path(FEATURES_PKL_PATH)
    if not pkl_path.exists():
        return None, None, {}
    try:
        df = pd.read_pickle(pkl_path)
    except Exception:
        return None, None, {}
    if 'filename' not in df.columns:
        return None, None, {}
    feature_cols = [c for c in df.columns if str(c).startswith('feature_')]
    if not feature_cols:
        return None, None, {}
    filename_norm = df['filename'].astype(str).map(_normalize_sample_name)
    full_map = {}
    base_map = {}
    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
    for i, name in enumerate(filename_norm.tolist()):
        full_map[name] = i
        base = name.split('/')[-1]
        if base not in base_map:
            base_map[base] = []
        base_map[base].append(i)
    X = []
    y = []
    class_counts = {}
    for label_name, records in records_by_type.items():
        if not records:
            continue
        if max_samples_per_class and max_samples_per_class > 0:
            records = records[:max_samples_per_class]
        used = 0
        for item in records:
            sid = _normalize_sample_name(item.get('sample_id', ''))
            idx = full_map.get(sid)
            if idx is None:
                base = sid.split('/')[-1]
                cand = base_map.get(base, [])
                if len(cand) == 1:
                    idx = cand[0]
            if idx is None:
                continue
            X.append(feature_matrix[idx])
            y.append(int(LABEL_TO_ID[label_name]))
            used += 1
        class_counts[label_name] = int(used)
    if len(X) < 30:
        return None, None, class_counts
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), class_counts

def _select_dense_features(X_train, X_val, max_input_dim, extra_stat_dim=6):
    if X_train.ndim != 2 or X_val.ndim != 2:
        return X_train, X_val, np.arange(X_train.shape[1], dtype=np.int64)
    n_feat = int(X_train.shape[1])
    if n_feat <= 0:
        return X_train, X_val, np.zeros((0,), dtype=np.int64)
    mean_abs = np.mean(np.abs(X_train), axis=0)
    var = np.var(X_train, axis=0)
    nz_ratio = np.mean(np.abs(X_train) > 1e-12, axis=0)
    score = mean_abs * 0.6 + var * 0.3 + nz_ratio * 0.1
    k = int(max(8, min(max_input_dim, n_feat)))
    selected_idx = np.argsort(score)[::-1][:k].astype(np.int64)
    X_train_sel = X_train[:, selected_idx]
    X_val_sel = X_val[:, selected_idx]
    if extra_stat_dim > 0:
        def _extra(x):
            eps = 1e-12
            abs_x = np.abs(x)
            l1 = np.sum(abs_x, axis=1, keepdims=True)
            l2 = np.sqrt(np.sum(x * x, axis=1, keepdims=True) + eps)
            max_v = np.max(abs_x, axis=1, keepdims=True)
            mean_v = np.mean(abs_x, axis=1, keepdims=True)
            nz = np.mean(abs_x > eps, axis=1, keepdims=True)
            p90 = np.percentile(abs_x, 90, axis=1).reshape(-1, 1)
            return np.concatenate([l1, l2, max_v, mean_v, nz, p90], axis=1).astype(np.float32)
        X_train_sel = np.hstack([X_train_sel, _extra(X_train_sel)]).astype(np.float32)
        X_val_sel = np.hstack([X_val_sel, _extra(X_val_sel)]).astype(np.float32)
    return X_train_sel, X_val_sel, selected_idx

def _compute_class_weights(y_train):
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    counts[counts <= 0] = 1.0
    inv = 1.0 / counts
    w = inv / np.mean(inv)
    return torch.tensor(w, dtype=torch.float32)

class FocalCrossEntropy(nn.Module):
    def __init__(self, class_weights, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing), reduction='none')

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        focal = ((1.0 - pt).clamp(min=1e-8) ** self.gamma) * ce
        return focal.mean()

def _build_sample_weights(y_train, class_weights, hard_class_boost=1.0):
    y = np.asarray(y_train, dtype=np.int64)
    cw = np.asarray(class_weights, dtype=np.float32)
    sample_w = cw[y].astype(np.float32)
    if sample_w.size > 0 and hard_class_boost > 1.0:
        sample_w[y == 0] *= float(hard_class_boost)
    sample_w = np.clip(sample_w, 1e-6, None)
    return sample_w

def _decide_predictions(score_array, args=None):
    preds = np.argmax(score_array, axis=1).astype(np.int64)
    if args is None or not bool(getattr(args, 'fn_priority_mode', False)):
        return preds
    fn_idx = int(LABEL_TO_ID['false_negatives'])
    fn_threshold = float(getattr(args, 'fn_threshold', 0.45))
    fn_margin = float(getattr(args, 'fn_margin', 0.0))
    other = np.delete(score_array, fn_idx, axis=1)
    other_max = np.max(other, axis=1)
    fn_score = score_array[:, fn_idx]
    force_mask = (fn_score >= fn_threshold) & ((fn_score - other_max) >= fn_margin)
    preds[force_mask] = fn_idx
    return preds

def _evaluate(model, loader, device, args=None):
    model.eval()
    all_preds = []
    all_targets = []
    all_scores = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_targets.extend(yb.cpu().numpy().tolist())
            all_scores.extend(probs.cpu().numpy().tolist())
    if not all_targets or not all_scores:
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'report': {}, 'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 'targets': [], 'preds': [], 'scores': []}
    score_array = np.asarray(all_scores, dtype=np.float32)
    pred_array = _decide_predictions(score_array, args=args)
    all_preds = pred_array.tolist()
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average='macro', zero_division=0))
    report = classification_report(all_targets, all_preds, target_names=[ID_TO_LABEL[i] for i in range(3)], output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2]).tolist()
    return {'accuracy': acc, 'macro_f1': macro_f1, 'report': report, 'confusion_matrix': cm, 'targets': all_targets, 'preds': all_preds, 'scores': score_array.tolist()}

def _evaluate_from_scores(targets, scores, args=None):
    target_array = np.asarray(targets, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float32)
    if target_array.size == 0 or score_array.size == 0:
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'report': {}, 'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 'targets': [], 'preds': [], 'scores': []}
    pred_array = _decide_predictions(score_array, args=args)
    acc = float(accuracy_score(target_array, pred_array))
    macro_f1 = float(f1_score(target_array, pred_array, average='macro', zero_division=0))
    report = classification_report(target_array, pred_array, target_names=[ID_TO_LABEL[i] for i in range(3)], output_dict=True, zero_division=0)
    cm = confusion_matrix(target_array, pred_array, labels=[0, 1, 2]).tolist()
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'report': report,
        'confusion_matrix': cm,
        'targets': target_array.tolist(),
        'preds': pred_array.tolist(),
        'scores': score_array.tolist(),
    }

def _make_decision_args(fn_priority_mode=False, fn_threshold=0.45, fn_margin=0.0):
    return type('DecisionArgs', (), {
        'fn_priority_mode': bool(fn_priority_mode),
        'fn_threshold': float(fn_threshold),
        'fn_margin': float(fn_margin),
    })()

def _train_one_branch(train_ds, val_loader, X_train, y_train, device, args, model_arch, fn_boost, fn_priority_mode, fn_threshold, fn_margin):
    class_weights = _compute_class_weights(y_train).cpu().numpy()
    if float(fn_boost) > 1.0:
        class_weights[LABEL_TO_ID['false_negatives']] *= float(fn_boost)
        class_weights = class_weights / np.mean(class_weights)
    if args.use_weighted_sampler:
        sample_weights = _build_sample_weights(y_train, class_weights, hard_class_boost=args.hard_class_boost)
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model_args = type('ModelArgs', (), {'model_arch': model_arch, 'dropout': args.dropout})()
    model, model_name = _build_hardcase_model(model_args, X_train.shape[1])
    model = model.to(device)
    criterion = FocalCrossEntropy(
        class_weights=torch.from_numpy(class_weights).to(device),
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {'macro_f1': -1.0, 'state': None, 'epoch': 0}
    wait = 0
    history = []
    eval_args = _make_decision_args(fn_priority_mode=fn_priority_mode, fn_threshold=fn_threshold, fn_margin=fn_margin)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        val_metric = _evaluate(model, val_loader, device, args=eval_args)
        train_loss = float(epoch_loss / max(1, len(train_loader)))
        history.append({'epoch': int(epoch), 'train_loss': train_loss, 'val_accuracy': float(val_metric['accuracy']), 'val_macro_f1': float(val_metric['macro_f1'])})
        if val_metric['macro_f1'] > best['macro_f1'] + 1e-9:
            best['macro_f1'] = float(val_metric['macro_f1'])
            best['epoch'] = int(epoch)
            best['state'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    final_eval = _evaluate(model, val_loader, device, args=eval_args)
    return {
        'model': model,
        'model_name': model_name,
        'best_epoch': int(best['epoch']),
        'history': history,
        'eval': final_eval,
    }

def _bootstrap_stability(final_eval, rounds=200, seed=42):
    targets = np.asarray(final_eval.get('targets', []), dtype=np.int64)
    preds = np.asarray(final_eval.get('preds', []), dtype=np.int64)
    n = int(targets.shape[0])
    if n == 0 or preds.shape[0] != n:
        return {'rounds': int(rounds), 'macro_f1_mean': 0.0, 'macro_f1_p05': 0.0, 'macro_f1_p95': 0.0, 'accuracy_mean': 0.0, 'accuracy_p05': 0.0, 'accuracy_p95': 0.0}
    rng = np.random.default_rng(int(seed))
    f1_list = []
    acc_list = []
    for _ in range(int(max(10, rounds))):
        idx = rng.integers(0, n, size=n)
        yt = targets[idx]
        yp = preds[idx]
        f1_list.append(float(f1_score(yt, yp, average='macro', zero_division=0)))
        acc_list.append(float(accuracy_score(yt, yp)))
    f1_arr = np.asarray(f1_list, dtype=np.float32)
    acc_arr = np.asarray(acc_list, dtype=np.float32)
    return {
        'rounds': int(max(10, rounds)),
        'macro_f1_mean': float(np.mean(f1_arr)),
        'macro_f1_p05': float(np.percentile(f1_arr, 5)),
        'macro_f1_p95': float(np.percentile(f1_arr, 95)),
        'accuracy_mean': float(np.mean(acc_arr)),
        'accuracy_p05': float(np.percentile(acc_arr, 5)),
        'accuracy_p95': float(np.percentile(acc_arr, 95)),
    }

def _save_eval_figures(final_eval, eval_dir):
    eval_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    cm_path = eval_dir / 'hardcase_dl_confusion_matrix.png'
    roc_path = eval_dir / 'hardcase_dl_roc_auc.png'
    cm = np.asarray(final_eval.get('confusion_matrix', [[0, 0, 0], [0, 0, 0], [0, 0, 0]]), dtype=np.int64)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[ID_TO_LABEL_ZH[i] for i in range(3)], yticklabels=[ID_TO_LABEL_ZH[i] for i in range(3)])
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('HardCase 三分类混淆矩阵')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=220)
    plt.close()
    targets = np.asarray(final_eval.get('targets', []), dtype=np.int64)
    preds = np.asarray(final_eval.get('preds', []), dtype=np.int64)
    scores = np.asarray(final_eval.get('scores', []), dtype=np.float32)
    if preds.size == 0 and scores.size > 0 and scores.ndim == 2:
        preds = np.argmax(scores, axis=1).astype(np.int64)
    one_vs_rest_paths = {}
    if targets.size > 0 and preds.size == targets.size:
        for i in range(3):
            cls_zh = ID_TO_LABEL_ZH[i]
            y_true_bin = (targets == i).astype(np.int64)
            y_pred_bin = (preds == i).astype(np.int64)
            cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            path_bin = eval_dir / f'hardcase_dl_confusion_matrix_{ID_TO_LABEL[i]}.png'
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm_bin,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['预测非该类', f'预测{cls_zh}'],
                yticklabels=['真实非该类', f'真实{cls_zh}']
            )
            plt.xlabel('预测')
            plt.ylabel('真实')
            plt.title(f'{cls_zh} 二分类混淆矩阵')
            plt.tight_layout()
            plt.savefig(path_bin, dpi=220)
            plt.close()
            one_vs_rest_paths[ID_TO_LABEL[i]] = str(path_bin)
    if targets.size > 0 and scores.size > 0 and scores.ndim == 2 and scores.shape[1] == 3:
        y_bin = label_binarize(targets, classes=[0, 1, 2])
        plt.figure(figsize=(8, 6))
        auc_macro_values = []
        for i in range(3):
            if np.unique(y_bin[:, i]).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
            class_auc = float(auc(fpr, tpr))
            auc_macro_values.append(class_auc)
            plt.plot(fpr, tpr, label=f"{ID_TO_LABEL_ZH[i]} AUC={class_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        macro_auc = float(np.mean(auc_macro_values)) if auc_macro_values else 0.0
        plt.title(f'HardCase ROC 曲线（宏平均AUC={macro_auc:.4f}）')
        plt.xlabel('假阳性率')
        plt.ylabel('真正率')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(roc_path, dpi=220)
        plt.close()
    else:
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('HardCase ROC 曲线（数据不足）')
        plt.xlabel('假阳性率')
        plt.ylabel('真正率')
        plt.tight_layout()
        plt.savefig(roc_path, dpi=220)
        plt.close()
    return {
        'multiclass_confusion_matrix_path': str(cm_path),
        'roc_auc_path': str(roc_path),
        'binary_confusion_matrices': one_vs_rest_paths,
    }

def run_training(args):
    _set_seed(args.seed)
    records = _collect_records()
    X, y, class_counts = _build_dataset_from_features_pkl(records, max_samples_per_class=args.max_samples_per_class)
    data_source = 'features_pkl'
    if X is None or y is None or len(y) < 30:
        X, y, class_counts = _build_dataset(records, max_samples_per_class=args.max_samples_per_class)
        data_source = 'hardcase_reports'
    if X is None or y is None or len(y) < 30:
        raise RuntimeError('困难样本/假阳性/假阴性样本不足，无法训练深度学习模型')
    unique_cls = np.unique(y)
    if len(unique_cls) < 3:
        raise RuntimeError('三类样本不完整，至少需要 hard_samples/false_positives/false_negatives 三类')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=args.seed, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_train, X_val, selected_idx = _select_dense_features(X_train, X_val, max_input_dim=args.max_input_dim, extra_stat_dim=6)
    lgb_ovr = OneVsRestClassifier(
        lgb.LGBMClassifier(
            objective='binary',
            n_estimators=int(args.gbdt_estimators),
            learning_rate=float(args.gbdt_learning_rate),
            num_leaves=int(args.gbdt_num_leaves),
            subsample=float(args.gbdt_subsample),
            colsample_bytree=float(args.gbdt_colsample_bytree),
            random_state=args.seed,
            verbose=-1,
        )
    )
    xgb_ovr = OneVsRestClassifier(
        xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=int(args.gbdt_estimators),
            learning_rate=float(args.gbdt_learning_rate),
            max_depth=int(args.gbdt_max_depth),
            subsample=float(args.gbdt_subsample),
            colsample_bytree=float(args.gbdt_colsample_bytree),
            reg_lambda=1.0,
            random_state=args.seed,
            eval_metric='logloss',
            n_jobs=4,
        )
    )
    lgb_ovr.fit(X_train, y_train)
    xgb_ovr.fit(X_train, y_train)
    lgb_scores = np.asarray(lgb_ovr.predict_proba(X_val), dtype=np.float32)
    xgb_scores = np.asarray(xgb_ovr.predict_proba(X_val), dtype=np.float32)
    score_array = (lgb_scores + xgb_scores) / 2.0
    fn_idx = int(LABEL_TO_ID['false_negatives'])
    other = np.delete(score_array, fn_idx, axis=1)
    other_max = np.max(other, axis=1)
    fn_prob = score_array[:, fn_idx]
    cascade_mask = (fn_prob >= float(args.cascade_fn_threshold)) & ((fn_prob - other_max) >= float(args.cascade_fn_margin))
    preds = np.argmax(score_array, axis=1).astype(np.int64)
    preds[cascade_mask] = fn_idx
    y_val_arr = np.asarray(y_val, dtype=np.int64)
    final_eval = {
        'accuracy': float(accuracy_score(y_val_arr, preds)),
        'macro_f1': float(f1_score(y_val_arr, preds, average='macro', zero_division=0)),
        'report': classification_report(y_val_arr, preds, target_names=[ID_TO_LABEL[i] for i in range(3)], output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_val_arr, preds, labels=[0, 1, 2]).tolist(),
        'targets': y_val_arr.tolist(),
        'preds': preds.tolist(),
        'scores': score_array.tolist(),
    }
    model_name = 'PlanA(OneVsRest LightGBM+XGBoost)'
    best_epoch = int(args.gbdt_estimators)
    history = []
    cascade_summary = {'cascade_triggered': int(np.sum(cascade_mask))}
    print(f"[hardcase-dl] eval best_epoch={best_epoch} val_acc={final_eval['accuracy']:.4f} val_f1={final_eval['macro_f1']:.4f}")
    print(f"[hardcase-dl] cascade_triggered={cascade_summary.get('cascade_triggered', 0)}")
    print(f"[hardcase-dl] data_source={data_source}")
    cm = final_eval.get('confusion_matrix', [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print(f"[hardcase-dl] confusion_matrix={cm}")
    report = final_eval.get('report', {})
    for cls_name in ['hard_samples', 'false_positives', 'false_negatives']:
        cls = report.get(cls_name, {})
        p = float(cls.get('precision', 0.0))
        r = float(cls.get('recall', 0.0))
        f1 = float(cls.get('f1-score', 0.0))
        s = int(cls.get('support', 0))
        print(f"[hardcase-dl] class={cls_name} precision={p:.4f} recall={r:.4f} f1={f1:.4f} support={s}")
    weights_dir = Path(RESOURCES_DIR) / 'weights'
    eval_dir = Path(RESOURCES_DIR) / 'eval'
    weights_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = _save_eval_figures(final_eval, eval_dir)
    print(f"[hardcase-dl] confusion matrix plot saved: {plot_paths.get('multiclass_confusion_matrix_path')}")
    for k, p in (plot_paths.get('binary_confusion_matrices') or {}).items():
        print(f"[hardcase-dl] binary confusion matrix saved ({k}): {p}")
    print(f"[hardcase-dl] roc auc plot saved: {plot_paths.get('roc_auc_path')}")
    stability = _bootstrap_stability(final_eval, rounds=args.bootstrap_rounds, seed=args.seed)
    print(f"[hardcase-dl] stability macro_f1={stability['macro_f1_mean']:.4f} p05={stability['macro_f1_p05']:.4f} p95={stability['macro_f1_p95']:.4f}")
    model_path = weights_dir / 'hardcase_dl_model.pkl'
    payload = {
        'model_family': 'plan_a_gbdt_ovr',
        'lgb_ovr': lgb_ovr,
        'xgb_ovr': xgb_ovr,
        'input_dim': int(X_train.shape[1]),
        'num_classes': 3,
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL,
        'scaler_mean': scaler.mean_.astype(np.float32),
        'scaler_scale': scaler.scale_.astype(np.float32),
        'selected_feature_indices': selected_idx.astype(np.int64),
        'best_epoch': int(best_epoch),
        'cascade_fn_threshold': float(args.cascade_fn_threshold),
        'cascade_fn_margin': float(args.cascade_fn_margin),
        'cascade_triggered': int(cascade_summary.get('cascade_triggered', 0)),
    }
    with open(model_path, 'wb') as f:
        pickle.dump(payload, f)
    cxx_model_paths = []
    for i, est in enumerate(lgb_ovr.estimators_):
        booster = getattr(est, 'booster_', None)
        if booster is None:
            raise RuntimeError(f'第{i}个LightGBM子模型缺少booster_，无法导出C++部署模型')
        cxx_model_path = weights_dir / f'hardcase_lgb_ovr_class_{i}.txt'
        booster.save_model(str(cxx_model_path))
        cxx_model_paths.append(cxx_model_path.name)
    cxx_manifest = {
        'model_family': 'plan_a_lgb_ovr_cxx',
        'class_model_paths': cxx_model_paths,
        'input_dim': int(X_train.shape[1]),
        'num_classes': 3,
        'label_to_id': LABEL_TO_ID,
        'id_to_label': {str(k): v for k, v in ID_TO_LABEL.items()},
        'scaler_mean': scaler.mean_.astype(np.float32).tolist(),
        'scaler_scale': scaler.scale_.astype(np.float32).tolist(),
        'selected_feature_indices': selected_idx.astype(np.int64).tolist(),
        'cascade_fn_threshold': float(args.cascade_fn_threshold),
        'cascade_fn_margin': float(args.cascade_fn_margin),
    }
    cxx_manifest_path = weights_dir / 'hardcase_cxx_manifest.json'
    cxx_manifest_path.write_text(json.dumps(cxx_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    metrics = {
        'dataset': {
            'data_source': data_source,
            'total_samples': int(len(y)),
            'train_samples': int(len(y_train)),
            'val_samples': int(len(y_val)),
            'class_counts': class_counts,
        },
        'architecture': {
            'type': model_name,
            'input_dim': int(X_train.shape[1]),
            'hidden_dims': [],
            'dropout': 0.0,
            'selected_input_dim': int(X_train.shape[1]),
            'selected_raw_feature_count': int(selected_idx.shape[0]),
            'num_classes': 3,
        },
        'training': {
            'epochs': int(args.gbdt_estimators),
            'best_epoch': int(best_epoch),
            'lr': float(args.gbdt_learning_rate),
            'batch_size': 0,
            'weight_decay': 0.0,
            'label_smoothing': 0.0,
            'focal_gamma': 0.0,
            'hard_class_boost': 1.0,
            'use_weighted_sampler': False,
            'fn_priority_mode': False,
            'fn_class_weight_boost': 1.0,
            'fn_threshold': 0.0,
            'fn_margin': 0.0,
            'enable_cascade': True,
            'cascade_base_arch': 'lightgbm',
            'cascade_fn_arch': 'xgboost',
            'cascade_fn_threshold': float(args.cascade_fn_threshold),
            'cascade_fn_margin': float(args.cascade_fn_margin),
            'patience': 0,
            'device': 'cpu',
            'seed': int(args.seed),
            'gbdt_estimators': int(args.gbdt_estimators),
            'gbdt_num_leaves': int(args.gbdt_num_leaves),
            'gbdt_max_depth': int(args.gbdt_max_depth),
            'gbdt_subsample': float(args.gbdt_subsample),
            'gbdt_colsample_bytree': float(args.gbdt_colsample_bytree),
        },
        'validation': final_eval,
        'validation_stability': stability,
        'history': history,
        'cascade': cascade_summary,
        'model_path': str(model_path),
        'cxx_manifest_path': str(cxx_manifest_path),
        'plots': plot_paths,
    }
    metrics_path = eval_dir / 'hardcase_dl_metrics.json'
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[hardcase-dl] model saved: {model_path}")
    print(f"[hardcase-dl] cxx manifest saved: {cxx_manifest_path}")
    print(f"[hardcase-dl] metrics saved: {metrics_path}")
    return metrics

def main(args):
    return run_training(args)
