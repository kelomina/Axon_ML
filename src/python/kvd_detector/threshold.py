import numpy as np

def choose_threshold(y_true, y_prob, fp_weight=1.0, fn_weight=1.0, max_fpr=None):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float32)
    thresholds = np.unique(y_prob)
    if thresholds.size == 0:
        return 0.5, {}
    best_t = 0.5
    best_cost = float("inf")
    best_stats = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fpr = fp / max(1, fp + tn)
        if max_fpr is not None and fpr > max_fpr:
            continue
        cost = fp_weight * fp + fn_weight * fn
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
            best_stats = {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "fpr": float(fpr)
            }
    return best_t, best_stats
