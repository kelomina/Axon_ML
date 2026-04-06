import numpy as np
from features.statistics import extract_statistical_features
from features.extractor_in_memory import PE_FEATURE_ORDER
from config.config import PE_FEATURE_VECTOR_DIM
from settings import NGRAM_SPECS, NGRAM_SCALE

STAT_FEATURE_DIM = 49
STAT_NAMES = [
    "byte_mean","byte_std","byte_min","byte_max","byte_median","byte_q25","byte_q75",
    "count_0","count_255","count_0x90","count_printable","entropy",
    "seg0_mean","seg0_std","seg0_entropy",
    "seg1_mean","seg1_std","seg1_entropy",
    "seg2_mean","seg2_std","seg2_entropy",
    "chunk_mean_0","chunk_mean_1","chunk_mean_2","chunk_mean_3","chunk_mean_4",
    "chunk_mean_5","chunk_mean_6","chunk_mean_7","chunk_mean_8","chunk_mean_9",
    "chunk_std_0","chunk_std_1","chunk_std_2","chunk_std_3","chunk_std_4",
    "chunk_std_5","chunk_std_6","chunk_std_7","chunk_std_8","chunk_std_9",
    "chunk_mean_diff_mean_abs","chunk_mean_diff_std","chunk_mean_diff_max","chunk_mean_diff_min",
    "chunk_std_diff_mean_abs","chunk_std_diff_std","chunk_std_diff_max","chunk_std_diff_min"
]

def _hashed_ngram(byte_sequence, n, dim):
    if byte_sequence is None or len(byte_sequence) < n:
        return np.zeros(dim, dtype=np.float32)
    seq = np.asarray(byte_sequence, dtype=np.uint8)
    if n == 2:
        idx = (seq[:-1].astype(np.int64) * 257 + seq[1:].astype(np.int64)) % dim
    else:
        idx = (seq[:-2].astype(np.int64) * 65537 + seq[1:-1].astype(np.int64) * 257 + seq[2:].astype(np.int64)) % dim
    counts = np.bincount(idx, minlength=dim).astype(np.float32)
    denom = max(1, idx.shape[0])
    return counts / float(denom)

def build_ngram_features(byte_sequence, ngram_specs=None):
    specs = ngram_specs or NGRAM_SPECS
    features = []
    for n, dim in specs:
        f = _hashed_ngram(byte_sequence, n, dim)
        features.append(f)
    if not features:
        return np.zeros(0, dtype=np.float32)
    out = np.concatenate(features, axis=0)
    if NGRAM_SCALE != 1.0:
        out = out * float(NGRAM_SCALE)
    return out.astype(np.float32)

def build_feature_vector(byte_sequence, pe_features, orig_length=None, ngram_specs=None):
    base = extract_statistical_features(byte_sequence, pe_features, orig_length)
    ngram = build_ngram_features(byte_sequence, ngram_specs=ngram_specs)
    if ngram.size == 0:
        return base
    return np.concatenate([base, ngram]).astype(np.float32)

def get_pe_feature_index(feature_key):
    try:
        return PE_FEATURE_ORDER.index(feature_key)
    except ValueError:
        return None

def get_packed_feature_indices():
    packed_idx = get_pe_feature_index("packed_sections_ratio")
    packer_idx = get_pe_feature_index("packer_keyword_hits_count")
    if packed_idx is None or packer_idx is None:
        return None, None
    return STAT_FEATURE_DIM + packed_idx, STAT_FEATURE_DIM + packer_idx

def split_feature_slices(total_dim, base_dim):
    base_slice = slice(0, base_dim)
    ngram_slice = slice(base_dim, total_dim)
    return base_slice, ngram_slice

def get_base_dim():
    return STAT_FEATURE_DIM + PE_FEATURE_VECTOR_DIM

def get_feature_names(ngram_specs=None):
    names = list(STAT_NAMES)
    lw = [f"lw_{i}" for i in range(256)]
    names.extend(lw)
    names.extend(list(PE_FEATURE_ORDER))
    if len(names) < STAT_FEATURE_DIM + PE_FEATURE_VECTOR_DIM:
        for i in range(len(names), STAT_FEATURE_DIM + PE_FEATURE_VECTOR_DIM):
            names.append(f"pe_extra_{i - (STAT_FEATURE_DIM + 256)}")
    specs = ngram_specs or NGRAM_SPECS
    for n, dim in specs:
        for i in range(dim):
            names.append(f"ngram_{n}_{i}")
    return names
