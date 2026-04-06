import os
import json
import random
from settings import HARD_NEGATIVE_POOL_PATH, HARD_NEGATIVE_MAX

def load_pool(path=HARD_NEGATIVE_POOL_PATH):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(dict.fromkeys(data))
    except json.JSONDecodeError as e:
        print(f"[Warning] Failed to parse hard negative pool JSON: {e}")
        return []
    except IOError as e:
        print(f"[Warning] Failed to read hard negative pool file: {e}")
        return []
    except Exception as e:
        print(f"[Warning] Unexpected error loading hard negative pool: {type(e).__name__}: {e}")
        return []

def save_pool(pool, path=HARD_NEGATIVE_POOL_PATH, limit=HARD_NEGATIVE_MAX * 10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    uniq = list(dict.fromkeys(pool))
    if len(uniq) > limit:
        uniq = uniq[-limit:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(uniq, f, ensure_ascii=False, indent=2)
    return uniq

def update_pool(false_positive_files, path=HARD_NEGATIVE_POOL_PATH):
    pool = load_pool(path)
    for f in false_positive_files:
        if f not in pool:
            pool.append(f)
    return save_pool(pool, path)

def sample_pool(pool, max_count=HARD_NEGATIVE_MAX, seed=42):
    if not pool:
        return []
    if len(pool) <= max_count:
        return list(pool)
    rng = random.Random(seed)
    return rng.sample(pool, max_count)
