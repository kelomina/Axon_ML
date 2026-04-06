import numpy as np
from config.config import BYTE_HISTOGRAM_BINS, STAT_CHUNK_COUNT

def extract_statistical_features(byte_sequence, pe_features, orig_length=None):
    if orig_length is not None and orig_length >= 0:
        byte_array = np.asarray(byte_sequence[:orig_length], dtype=np.uint8)
    else:
        byte_array = np.asarray(byte_sequence, dtype=np.uint8)
    length = len(byte_array)
    features = []
    counts = np.bincount(byte_array, minlength=256) if length > 0 else np.zeros(256, dtype=np.int64)
    if length > 0:
        value_axis = np.arange(256, dtype=np.float64)
        weighted_sum = float(np.dot(counts, value_axis))
        mean_val = weighted_sum / float(length)
        weighted_sq_sum = float(np.dot(counts, value_axis * value_axis))
        variance = max(0.0, weighted_sq_sum / float(length) - mean_val * mean_val)
        std_val = float(np.sqrt(variance))
        nonzero_indices = np.flatnonzero(counts)
        min_val = float(nonzero_indices[0]) if nonzero_indices.size > 0 else 0.0
        max_val = float(nonzero_indices[-1]) if nonzero_indices.size > 0 else 0.0
        cdf = np.cumsum(counts)
        median_val = float(np.searchsorted(cdf, int(np.ceil(0.50 * length))))
        q25 = float(np.searchsorted(cdf, int(np.ceil(0.25 * length))))
        q75 = float(np.searchsorted(cdf, int(np.ceil(0.75 * length))))
    else:
        mean_val = 0.0
        std_val = 0.0
        min_val = 0.0
        max_val = 0.0
        median_val = 0.0
        q25 = 0.0
        q75 = 0.0
    features.extend([mean_val, std_val, min_val, max_val, median_val, q25, q75])
    features.extend([
        int(counts[0]),
        int(counts[255]),
        int(counts[0x90]),
        int(np.sum(counts[32:127])),
    ])
    p = counts.astype(np.float64) / float(length) if length > 0 else np.zeros_like(counts, dtype=np.float64)
    p = p[p > 0]
    entropy = float((-np.sum(p * np.log2(p)) / 8.0) if p.size > 0 else 0.0)
    features.append(entropy)
    if length >= 3:
        one_third = length // 3
        segments = [
            byte_array[:one_third].copy(),
            byte_array[one_third:2 * one_third].copy(),
            byte_array[2 * one_third:].copy(),
        ]
    else:
        segments = [byte_array.copy(), byte_array.copy(), byte_array.copy()]
    for seg in segments:
        if len(seg) == 0:
            seg_mean = 0.0
            seg_std = 0.0
            seg_entropy = 0.0
        else:
            seg_mean = float(np.mean(seg))
            seg_std = float(np.std(seg))
            seg_counts = np.bincount(seg, minlength=256)
            seg_p = seg_counts.astype(np.float64) / float(len(seg))
            seg_p = seg_p[seg_p > 0]
            seg_entropy = float((-np.sum(seg_p * np.log2(seg_p)) / 8.0) if seg_p.size > 0 else 0.0)
        features.extend([seg_mean, seg_std, seg_entropy])
    chunk_size = max(1, length // STAT_CHUNK_COUNT)
    chunk_means = []
    chunk_stds = []
    for i in range(STAT_CHUNK_COUNT):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < STAT_CHUNK_COUNT - 1 else length
        chunk = byte_array[start_idx:end_idx]
        if len(chunk) > 0:
            chunk_means.append(float(np.mean(chunk)))
            chunk_stds.append(float(np.std(chunk)))
        else:
            chunk_means.append(0.0)
            chunk_stds.append(0.0)
    features.extend(chunk_means)
    features.extend(chunk_stds)
    chunk_means = np.array(chunk_means, dtype=np.float32)
    chunk_stds = np.array(chunk_stds, dtype=np.float32)
    if len(chunk_means) > 1:
        mean_diffs = np.diff(chunk_means)
        std_diffs = np.diff(chunk_stds)
        features.extend([
            float(np.mean(np.abs(mean_diffs))),
            float(np.std(mean_diffs)),
            float(np.max(mean_diffs)),
            float(np.min(mean_diffs)),
        ])
        features.extend([
            float(np.mean(np.abs(std_diffs))),
            float(np.std(std_diffs)),
            float(np.max(std_diffs)),
            float(np.min(std_diffs)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
        features.extend([0.0, 0.0, 0.0, 0.0])
    features.extend(pe_features.tolist())
    return np.array(features, dtype=np.float32)
