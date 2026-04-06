import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.dataset import MalwareDataset
from feature_enhancer import build_feature_vector, get_base_dim
from settings import DATA_DIR, METADATA_FILE, MAX_FILE_SIZE, CACHE_DIR

def _build_label_map(metadata):
    label_map = {}
    for file, label in metadata.items():
        fname_lower = file.lower()
        if ('待加入白名单' in file) or ('whitelist' in fname_lower) or ('benign' in fname_lower) or ('good' in fname_lower) or ('clean' in fname_lower):
            label_map[file] = 0
        elif ('malicious' in fname_lower) or ('virus' in fname_lower) or ('trojan' in fname_lower):
            label_map[file] = 1
        elif label == 'benign' or label == 0:
            label_map[file] = 0
        elif label == 'malicious' or label == 1:
            label_map[file] = 1
        elif label == '待加入白名单':
            label_map[file] = 0
        else:
            label_map[file] = 1
    return label_map

def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)

def load_dataset_v2(use_cache=True, fast_dev_run=False):
    cache_path = _cache_path("features_v2.pkl")
    if use_cache and os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        y = df["label"].values.astype(np.int64)
        files = df["filename"].tolist()
        X = df.drop(["filename", "label"], axis=1).values.astype(np.float32)
        base_dim = get_base_dim()
        return X, y, files, base_dim
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    label_map = _build_label_map(metadata)
    all_files = list(metadata.keys())
    all_labels = [label_map[fname] for fname in all_files]
    if fast_dev_run:
        benign_files = [f for f, label in zip(all_files, all_labels) if label == 0]
        malicious_files = [f for f, label in zip(all_files, all_labels) if label == 1]
        n_samples_per_class = 2000
        selected_benign_files = benign_files[:min(n_samples_per_class, len(benign_files))]
        selected_malicious_files = malicious_files[:min(n_samples_per_class, len(malicious_files))]
        all_files = selected_benign_files + selected_malicious_files
        all_labels = [0] * len(selected_benign_files) + [1] * len(selected_malicious_files)
    dataset = MalwareDataset(DATA_DIR, all_files, all_labels, MAX_FILE_SIZE)
    features_list = []
    labels_list = []
    valid_files = []
    for i in tqdm(range(len(dataset)), desc="Extracting enhanced features"):
        byte_sequence, pe_features, label, orig_length = dataset[i]
        feat = build_feature_vector(byte_sequence, pe_features, orig_length)
        features_list.append(feat)
        labels_list.append(label)
        valid_files.append(all_files[i])
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    df = pd.DataFrame(X)
    df["filename"] = valid_files
    df["label"] = y
    if use_cache:
        df.to_pickle(cache_path)
    base_dim = get_base_dim()
    return X, y, valid_files, base_dim
