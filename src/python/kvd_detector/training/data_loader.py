import os
import json
import concurrent.futures
import numpy as np
from tqdm import tqdm

from data.dataset import MalwareDataset
from features.statistics import extract_statistical_features
from config.config import DEFAULT_MAX_FILE_SIZE, PE_FEATURE_VECTOR_DIM

RAW_EXTRACT_WRITE_WORKERS = 16
RAW_EXTRACT_COMPRESS_NPZ = False
RAW_EXTRACT_CACHE_BACKFILL = True
RAW_EXTRACT_CACHE_KEY = 'stat_features'

def _save_npz_with_stat_cache(file_path, byte_sequence, pe_features, orig_length, stat_features, compress=False):
    tmp_file = f"{file_path}.tmp.npz"
    if compress:
        np.savez_compressed(
            tmp_file,
            byte_sequence=byte_sequence,
            pe_features=pe_features,
            orig_length=int(orig_length),
            stat_features=stat_features.astype(np.float32, copy=False),
        )
    else:
        np.savez(
            tmp_file,
            byte_sequence=byte_sequence,
            pe_features=pe_features,
            orig_length=int(orig_length),
            stat_features=stat_features.astype(np.float32, copy=False),
        )
    os.replace(tmp_file, file_path)

def _infer_label_from_filename(filename):
    fname_lower = filename.lower()
    benign_patterns = (
        '待加入白名单', 'whitelist', 'benign', 'good', 'clean',
        'white_list', 'known_good', 'trusted', 'safe', 'negative'
    )
    malicious_patterns = (
        'malicious', 'virus', 'trojan', 'malware', 'infected',
        'harmful', 'dangerous', 'positive', 'threat', 'ransomware',
        'backdoor', 'rootkit', 'worm', 'spyware', 'adware', 'dropper'
    )
    for pattern in benign_patterns:
        if pattern in fname_lower or pattern in filename:
            return 0
    for pattern in malicious_patterns:
        if pattern in fname_lower:
            return 1
    return None

def _infer_label_from_metadata_label(label):
    if label is None:
        return None
    if isinstance(label, str):
        label_lower = label.lower()
        if label_lower in ('benign', 'good', 'clean', 'white', 'safe', 'negative', '0', '待加入白名单'):
            return 0
        if label_lower in ('malicious', 'malware', 'bad', 'dangerous', 'positive', 'threat', '1', '恶意'):
            return 1
    if isinstance(label, (int, float)):
        if label == 0 or label == 0.0:
            return 0
        if label == 1 or label == 1.0:
            return 1
    return None

def load_dataset(data_dir, metadata_file, max_file_size=DEFAULT_MAX_FILE_SIZE, fast_dev_run=False, max_workers=16):
    print("[*] Loading dataset...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    label_map = {}
    for file, label in metadata.items():
        inferred_from_filename = _infer_label_from_filename(file)
        inferred_from_meta = _infer_label_from_metadata_label(label)
        if inferred_from_filename is not None:
            label_map[file] = inferred_from_filename
        elif inferred_from_meta is not None:
            label_map[file] = inferred_from_meta
        else:
            label_map[file] = 1
            print(f"[Warning] Unknown label for {file}, defaulting to malicious (1)")
    benign_count = sum(1 for v in label_map.values() if v == 0)
    malicious_count = sum(1 for v in label_map.values() if v == 1)
    print(f"[+] Label distribution: Benign={benign_count}, Malicious={malicious_count}")
    all_files = list(metadata.keys())
    all_labels = [label_map[fname] for fname in all_files]
    if fast_dev_run:
        print("[!] Fast development mode enabled, balancing benign and malicious samples.")
        benign_files = [f for f, label in zip(all_files, all_labels) if label == 0]
        malicious_files = [f for f, label in zip(all_files, all_labels) if label == 1]
        n_samples_per_class = 5000
        selected_benign_files = benign_files[:min(n_samples_per_class, len(benign_files))]
        selected_malicious_files = malicious_files[:min(n_samples_per_class, len(malicious_files))]
        all_files = selected_benign_files + selected_malicious_files
        all_labels = [0] * len(selected_benign_files) + [1] * len(selected_malicious_files)
        print(f"    Benign samples: {len(selected_benign_files)}")
        print(f"    Malicious samples: {len(selected_malicious_files)}")
    print(f"[+] Loaded {len(all_files)} files")
    features_list = []
    labels_list = []
    valid_files = []
    total_samples = len(all_files)
    progress_desc = "Extracting features"
    from config.config import PE_FEATURE_VECTOR_DIM
    count_ok = 0
    count_padded = 0
    count_truncated = 0
    count_cache_hit = 0
    count_cache_write = 0
    effective_workers = int(max(1, max_workers))
    def _process_sample(i):
        try:
            filename = all_files[i]
            label = all_labels[i]
            file_path = os.path.join(data_dir, filename) if filename.endswith('.npz') else os.path.join(data_dir, f"{filename}.npz")
            cache_features = None
            try:
                with np.load(file_path) as data:
                    byte_sequence = data['byte_sequence']
                    if 'pe_features' in data:
                        pe_features = data['pe_features']
                        if pe_features.ndim > 1:
                            pe_features = pe_features.flatten()
                    else:
                        pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                    orig_length = int(data['orig_length']) if 'orig_length' in data else max_file_size
                    if RAW_EXTRACT_CACHE_KEY in data:
                        cache_features = np.asarray(data[RAW_EXTRACT_CACHE_KEY], dtype=np.float32).flatten()
            except FileNotFoundError:
                byte_sequence = np.zeros(max_file_size, dtype=np.uint8)
                pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                orig_length = 0
                cache_features = None
            except Exception:
                byte_sequence = np.zeros(max_file_size, dtype=np.uint8)
                pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                orig_length = 0
                cache_features = None
            if len(byte_sequence) > max_file_size:
                byte_sequence = byte_sequence[:max_file_size]
            elif len(byte_sequence) < max_file_size:
                padded = np.zeros(max_file_size, dtype=np.uint8)
                padded[:len(byte_sequence)] = byte_sequence
                byte_sequence = padded
            orig_pe_len = len(pe_features)
            status = 'ok'
            if orig_pe_len != PE_FEATURE_VECTOR_DIM:
                fixed_pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                copy_len = min(orig_pe_len, PE_FEATURE_VECTOR_DIM)
                fixed_pe_features[:copy_len] = pe_features[:copy_len]
                pe_features = fixed_pe_features
                if orig_pe_len < PE_FEATURE_VECTOR_DIM:
                    status = 'padded'
                    print(f"[Warning] PE features padded: {file_path} (expected {PE_FEATURE_VECTOR_DIM}, got {orig_pe_len})")
                else:
                    status = 'truncated'
                    print(f"[Warning] PE features truncated: {file_path} (expected {PE_FEATURE_VECTOR_DIM}, got {orig_pe_len})")
            if cache_features is not None and cache_features.size > 0:
                return i, cache_features, label, status, None, True, False
            features = extract_statistical_features(byte_sequence, pe_features, orig_length)
            cache_written = False
            if RAW_EXTRACT_CACHE_BACKFILL and os.path.isfile(file_path):
                try:
                    _save_npz_with_stat_cache(
                        file_path,
                        byte_sequence=byte_sequence,
                        pe_features=pe_features,
                        orig_length=orig_length,
                        stat_features=features,
                        compress=RAW_EXTRACT_COMPRESS_NPZ,
                    )
                    cache_written = True
                except Exception:
                    cache_written = False
            return i, features, label, status, None, False, cache_written
        except Exception as e:
            return i, None, None, None, e, False, False
    if total_samples > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(effective_workers, total_samples)) as executor:
            for i, features, label, status, error, cache_hit, cache_written in tqdm(
                executor.map(_process_sample, range(total_samples)),
                total=total_samples,
                desc=progress_desc
            ):
                if error is not None:
                    print(f"[!] Error processing file {all_files[i]}: {error}")
                    continue
                if cache_hit:
                    count_cache_hit += 1
                if cache_written:
                    count_cache_write += 1
                if status == 'ok':
                    count_ok += 1
                elif status == 'padded':
                    count_padded += 1
                else:
                    count_truncated += 1
                features_list.append(features)
                labels_list.append(label)
                valid_files.append(all_files[i])
    try:
        X = np.array(features_list, dtype=np.float32)
    except ValueError as e:
        print(f"[!] Feature array shape inconsistency: {e}")
        print("[*] Attempting to manually align feature dimensions...")
        max_features = max(len(f) for f in features_list)
        aligned_features = []
        for f in features_list:
            if len(f) < max_features:
                padded_f = np.zeros(max_features, dtype=np.float32)
                padded_f[:len(f)] = f
                aligned_features.append(padded_f)
            else:
                aligned_features.append(f)
        X = np.array(aligned_features, dtype=np.float32)
    y = np.array(labels_list)
    print(f"[+] Feature extraction completed, feature dimension: {X.shape[1]}")
    print(f"[+] Valid samples: {X.shape[0]}")
    print(f"[+] 统计特征缓存：hit={count_cache_hit}，backfill={count_cache_write}")
    try:
        total = count_ok + count_padded + count_truncated
        if total > 0:
            print(f"[+] PE维度汇总：total={total}，ok={count_ok}，padded={count_padded}，truncated={count_truncated}")
            from config.config import SCAN_OUTPUT_DIR, PE_DIM_SUMMARY_DATASET
            os.makedirs(SCAN_OUTPUT_DIR, exist_ok=True)
            with open(PE_DIM_SUMMARY_DATASET, 'w', encoding='utf-8') as f:
                json.dump({
                    'total': int(total),
                    'ok': int(count_ok),
                    'padded': int(count_padded),
                    'truncated': int(count_truncated),
                    'feature_dim': int(X.shape[1]),
                    'pe_dim': int(PE_FEATURE_VECTOR_DIM)
                }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return X, y, valid_files

def extract_features_from_raw_files(data_dir, output_dir, max_file_size=DEFAULT_MAX_FILE_SIZE, file_extensions=None, label_inference='filename', max_workers=16):
    print(f"[*] Extracting features from raw files: {data_dir}")
    print(f"[*] Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_extensions:
                _, ext = os.path.splitext(file)
                if ext.lower() not in file_extensions:
                    continue
            all_files.append(file_path)
    if not all_files:
        print(f"[!] No files found in raw file directory: {data_dir}")
        return [], []
    print(f"[+] Found {len(all_files)} files in raw file directory")
    try:
        from features.extractor_in_memory import (
            extract_byte_sequence,
            extract_combined_pe_features_batch_native,
            NATIVE_BATCH_SIZE,
            NATIVE_BATCH_THREADS,
        )
        print("[+] Successfully imported feature extraction module")
    except ImportError as e:
        print(f"[!] Failed to import feature extraction module: {e}")
        return [], []
    labels = []
    output_files = []
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, data_dir)
        output_file = os.path.join(output_dir, rel_path + '.npz')
        output_files.append(output_file)
        output_subdir = os.path.dirname(output_file)
        os.makedirs(output_subdir, exist_ok=True)
        if label_inference == 'filename':
            file_name = os.path.basename(file_path)
            if 'benign' in file_name.lower() or 'good' in file_name.lower() or 'clean' in file_name.lower():
                labels.append(0)
            else:
                labels.append(1)
        elif label_inference == 'directory':
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if 'benign' in parent_dir.lower() or 'good' in parent_dir.lower() or 'clean' in parent_dir.lower():
                labels.append(0)
            else:
                labels.append(1)
        else:
            labels.append(1)
    print("[*] Starting feature extraction...")
    success_count = 0
    from tqdm import tqdm
    successful_output_files = []
    successful_labels = []
    effective_workers = int(max(1, max_workers))
    write_workers = min(RAW_EXTRACT_WRITE_WORKERS, effective_workers)
    compress_npz = RAW_EXTRACT_COMPRESS_NPZ
    for batch_start in tqdm(range(0, len(all_files), NATIVE_BATCH_SIZE), total=(len(all_files) + NATIVE_BATCH_SIZE - 1) // NATIVE_BATCH_SIZE, desc="Feature extraction"):
        batch_end = min(batch_start + NATIVE_BATCH_SIZE, len(all_files))
        batch_inputs = all_files[batch_start:batch_end]
        batch_outputs = output_files[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        batch_pe_features, batch_status = extract_combined_pe_features_batch_native(
            batch_inputs,
            thread_count=min(NATIVE_BATCH_THREADS, effective_workers),
        )
        def _process_one(idx):
            input_file = batch_inputs[idx]
            output_file = batch_outputs[idx]
            byte_sequence, orig_length = extract_byte_sequence(input_file, max_file_size)
            if byte_sequence is None:
                raise Exception(f"Failed to extract byte sequence from {input_file}")
            if idx >= len(batch_status) or batch_status[idx] != 0:
                raise Exception("skip_unparsable_pe")
            pe_features = batch_pe_features[idx]
            stat_features = extract_statistical_features(byte_sequence, pe_features, orig_length)
            if compress_npz:
                np.savez_compressed(
                    output_file,
                    byte_sequence=byte_sequence,
                    pe_features=pe_features,
                    orig_length=orig_length,
                    stat_features=stat_features,
                )
            else:
                np.savez(
                    output_file,
                    byte_sequence=byte_sequence,
                    pe_features=pe_features,
                    orig_length=orig_length,
                    stat_features=stat_features,
                )
            return output_file, batch_labels[idx]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(write_workers, len(batch_inputs))) as executor:
            futures = [executor.submit(_process_one, idx) for idx in range(len(batch_inputs))]
            for idx, future in enumerate(futures):
                input_file = batch_inputs[idx]
                try:
                    output_file, label = future.result()
                    successful_output_files.append(output_file)
                    successful_labels.append(label)
                    success_count += 1
                except Exception as e:
                    if str(e) == "skip_unparsable_pe":
                        continue
                    print(f"[!] Error processing file {input_file}: {e}")
                    continue
    output_files = successful_output_files
    labels = successful_labels
    print(f"[+] Feature extraction completed: {success_count}/{len(all_files)} files processed successfully")
    try:
        count_ok = 0
        count_padded = 0
        count_truncated = 0
        for of in output_files:
            try:
                with np.load(of) as data:
                    if 'pe_features' in data:
                        pe = data['pe_features']
                        orig_len = pe.shape[0] if hasattr(pe, 'shape') else len(pe)
                        if orig_len == PE_FEATURE_VECTOR_DIM:
                            count_ok += 1
                        elif orig_len < PE_FEATURE_VECTOR_DIM:
                            count_padded += 1
                        else:
                            count_truncated += 1
            except Exception:
                pass
        total = count_ok + count_padded + count_truncated
        if total > 0:
            print(f"[+] 原始批处理PE维度汇总：total={total}，ok={count_ok}，padded={count_padded}，truncated={count_truncated}")
            from config.config import SCAN_OUTPUT_DIR, PE_DIM_SUMMARY_RAW
            os.makedirs(SCAN_OUTPUT_DIR, exist_ok=True)
            with open(PE_DIM_SUMMARY_RAW, 'w', encoding='utf-8') as f:
                json.dump({
                    'total': int(total),
                    'ok': int(count_ok),
                    'padded': int(count_padded),
                    'truncated': int(count_truncated),
                    'pe_dim': int(PE_FEATURE_VECTOR_DIM)
                }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    file_names = [os.path.relpath(f, output_dir) for f in output_files]
    return file_names, labels

def load_incremental_dataset(data_dir, max_file_size=DEFAULT_MAX_FILE_SIZE):
    print(f"[*] Loading dataset from incremental directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"[!] Incremental training directory does not exist: {data_dir}")
        return None, None, None
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                all_files.append(os.path.join(root, file))
    if not all_files:
        print(f"[!] No .npz files found in incremental training directory: {data_dir}")
        return None, None, None
    print(f"[+] Found {len(all_files)} files in incremental directory")
    labels = []
    valid_files = []
    file_names = []
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        if 'benign' in file_name.lower() or 'good' in file_name.lower() or 'clean' in file_name.lower():
            labels.append(0)
        else:
            labels.append(1)
        valid_files.append(file_path)
        file_names.append(file_name)
    features_list = []
    valid_file_names = []
    from config.config import PE_FEATURE_VECTOR_DIM
    count_ok = 0
    count_padded = 0
    count_truncated = 0
    for i, file_path in enumerate(tqdm(valid_files, desc="Extracting incremental features")):
        try:
            with np.load(file_path) as data:
                byte_sequence = data['byte_sequence']
                if 'pe_features' in data:
                    pe_features = data['pe_features']
                    if pe_features.ndim > 1:
                        pe_features = pe_features.flatten()
                else:
                    pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                orig_length = data['orig_length'] if 'orig_length' in data else max_file_size
                cache_features = np.asarray(data[RAW_EXTRACT_CACHE_KEY], dtype=np.float32).flatten() if RAW_EXTRACT_CACHE_KEY in data else None
            if len(byte_sequence) > max_file_size:
                byte_sequence = byte_sequence[:max_file_size]
            else:
                byte_sequence = np.pad(byte_sequence, (0, max_file_size - len(byte_sequence)), 'constant')
            orig_pe_len = len(pe_features)
            status = 'ok'
            if orig_pe_len != PE_FEATURE_VECTOR_DIM:
                fixed_pe_features = np.zeros(PE_FEATURE_VECTOR_DIM, dtype=np.float32)
                copy_len = min(orig_pe_len, PE_FEATURE_VECTOR_DIM)
                fixed_pe_features[:copy_len] = pe_features[:copy_len]
                pe_features = fixed_pe_features
                status = 'padded' if orig_pe_len < PE_FEATURE_VECTOR_DIM else 'truncated'
            if status == 'ok':
                count_ok += 1
            elif status == 'padded':
                count_padded += 1
            else:
                count_truncated += 1
            if cache_features is not None and cache_features.size > 0:
                features = cache_features
            else:
                features = extract_statistical_features(byte_sequence, pe_features, int(orig_length))
                if RAW_EXTRACT_CACHE_BACKFILL:
                    try:
                        _save_npz_with_stat_cache(
                            file_path,
                            byte_sequence=byte_sequence,
                            pe_features=pe_features,
                            orig_length=orig_length,
                            stat_features=features,
                            compress=RAW_EXTRACT_COMPRESS_NPZ,
                        )
                    except Exception:
                        pass
            features_list.append(features)
            valid_file_names.append(file_names[i])
        except Exception as e:
            print(f"[!] Error processing file {file_path}: {e}")
            continue
    if not features_list:
        print("[!] Failed to extract any features from incremental data")
        return None, None, None
    try:
        X = np.array(features_list, dtype=np.float32)
    except ValueError as e:
        print(f"[!] Incremental feature array shape inconsistency: {e}")
        return None, None, None
    y = np.array(labels[:len(features_list)])
    print(f"[+] Incremental feature extraction completed, feature dimension: {X.shape[1]}")
    print(f"[+] Valid samples: {X.shape[0]}")
    try:
        total = count_ok + count_padded + count_truncated
        if total > 0:
            print(f"[+] PE维度汇总：total={total}，ok={count_ok}，padded={count_padded}，truncated={count_truncated}")
            from config.config import SCAN_OUTPUT_DIR, PE_DIM_SUMMARY_INCREMENTAL
            os.makedirs(SCAN_OUTPUT_DIR, exist_ok=True)
            with open(PE_DIM_SUMMARY_INCREMENTAL, 'w', encoding='utf-8') as f:
                json.dump({
                    'total': int(total),
                    'ok': int(count_ok),
                    'padded': int(count_padded),
                    'truncated': int(count_truncated),
                    'feature_dim': int(X.shape[1]),
                    'pe_dim': int(PE_FEATURE_VECTOR_DIM)
                }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return X, y, valid_file_names
