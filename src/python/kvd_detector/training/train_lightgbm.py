import numpy as np
import lightgbm as lgb
import multiprocessing
from config.config import WARMUP_ROUNDS, WARMUP_START_LR, LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ, LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF, LIGHTGBM_NUM_THREADS_MAX, FP_WEIGHT_BASE, FP_WEIGHT_GROWTH_PER_ITER, FP_WEIGHT_MAX, DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_LIGHTGBM_LEARNING_RATE, DEFAULT_LIGHTGBM_NUM_LEAVES

def warmup_scheduler(warmup_rounds=WARMUP_ROUNDS, start_lr=WARMUP_START_LR, target_lr=0.05):
    def callback(env):
        if env.iteration < warmup_rounds:
            lr = start_lr + (target_lr - start_lr) * (env.iteration / warmup_rounds)
            env.model.params['learning_rate'] = lr
            if env.iteration % 20 == 0:
                 print(f"[*] Warmup: Iteration {env.iteration}, LR: {lr:.6f}")
    return callback

def train_lightgbm_model(X_train, y_train, X_val, y_val, false_positive_files=None, files_train=None, iteration=1, num_boost_round=5000, init_model=None, params_override=None, sample_weights=None):
    print(f"[*] Training LightGBM model (Round {iteration})...")
    combined_weights = np.ones(len(X_train), dtype=np.float32)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if sample_weights.shape[0] == len(combined_weights):
            combined_weights = np.maximum(combined_weights, sample_weights)
    if false_positive_files is not None and files_train is not None:
        print(f"[*] Detected {len(false_positive_files)} false positive samples, increasing their training weights")
        false_positive_set = set(false_positive_files)
        false_positive_count = 0
        weight_factor = min(FP_WEIGHT_BASE + iteration * FP_WEIGHT_GROWTH_PER_ITER, FP_WEIGHT_MAX)
        print(f"[*] Current false positive weight factor: {weight_factor}")
        for i, file in enumerate(files_train):
            if file in false_positive_set:
                combined_weights[i] = max(combined_weights[i], weight_factor)
                false_positive_count += 1
        print(f"[+] Identified {false_positive_count} false positive samples, adjusted weights")
    train_data = lgb.Dataset(X_train, label=y_train, weight=combined_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    learning_rate = DEFAULT_LIGHTGBM_LEARNING_RATE
    num_leaves = DEFAULT_LIGHTGBM_NUM_LEAVES
    feature_fraction = LIGHTGBM_FEATURE_FRACTION
    bagging_fraction = LIGHTGBM_BAGGING_FRACTION
    bagging_freq = LIGHTGBM_BAGGING_FREQ
    min_gain_to_split = LIGHTGBM_MIN_GAIN_TO_SPLIT
    min_data_in_leaf = LIGHTGBM_MIN_DATA_IN_LEAF
    if params_override:
        learning_rate = params_override.get('learning_rate', learning_rate)
        num_leaves = params_override.get('num_leaves', num_leaves)
        feature_fraction = params_override.get('feature_fraction', feature_fraction)
        bagging_fraction = params_override.get('bagging_fraction', bagging_fraction)
        bagging_freq = params_override.get('bagging_freq', bagging_freq)
        min_gain_to_split = params_override.get('min_gain_to_split', min_gain_to_split)
        min_data_in_leaf = params_override.get('min_data_in_leaf', min_data_in_leaf)
    positive_count = float(np.sum(y_train == 1))
    negative_count = float(np.sum(y_train == 0))
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
    if params_override:
        scale_pos_weight = float(params_override.get('scale_pos_weight', scale_pos_weight))
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'min_gain_to_split': min_gain_to_split,
        'min_data_in_leaf': min_data_in_leaf,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), LIGHTGBM_NUM_THREADS_MAX)
    }
    print(f"[*] Current training parameters - Learning rate: {learning_rate:.4f}, Number of leaves: {num_leaves}")
    callbacks = [lgb.early_stopping(DEFAULT_EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50)]
    if iteration == 1 and init_model is None:
        print("[*] Applying Warm Start scheduler...")
        callbacks.append(warmup_scheduler(warmup_rounds=WARMUP_ROUNDS, start_lr=WARMUP_START_LR, target_lr=learning_rate))
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['validation'],
        num_boost_round=num_boost_round,
        init_model=init_model,
        callbacks=callbacks
    )
    return model
