import os
import json
import argparse
from pathlib import Path
from features.extractor_in_memory import extract_enhanced_pe_features
from config.config import PACKED_SECTIONS_RATIO_THRESHOLD, PACKER_KEYWORD_HITS_THRESHOLD

def collect_signals(file_path: str) -> dict:
    pe = extract_enhanced_pe_features(file_path)
    signals = {
        'packed_sections_ratio': float(pe.get('packed_sections_ratio', 0.0)),
        'packer_keyword_hits_count': float(pe.get('packer_keyword_hits_count', 0.0)),
    }
    return signals

def decide(signals: dict) -> str:
    packed_sections_ratio = float(signals.get('packed_sections_ratio', 0.0))
    packer_keyword_hits_count = float(signals.get('packer_keyword_hits_count', 0.0))
    is_packed = (packed_sections_ratio > float(PACKED_SECTIONS_RATIO_THRESHOLD)) or (
        packer_keyword_hits_count > float(PACKER_KEYWORD_HITS_THRESHOLD)
    )
    return 'packed' if is_packed else 'normal'

def evaluate_directory(directory_path: str, recursive: bool = False) -> dict:
    files = Path(directory_path).rglob('*') if recursive else Path(directory_path).glob('*')
    files = [f for f in files if f.is_file()]
    packed = 0
    normal = 0
    details = []
    for f in files:
        s = collect_signals(str(f))
        d = decide(s)
        if d == 'packed':
            packed += 1
        else:
            normal += 1
        if len(details) < 50:
            details.append({'file_path': str(f), 'decision': d, 'signals': s})
    return {'total': len(files), 'packed': packed, 'normal': normal, 'details_sample': details}

def main():
    parser = argparse.ArgumentParser(description='Gating validation')
    parser.add_argument('--dir-path', type=str)
    parser.add_argument('--file-path', type=str)
    parser.add_argument('--recursive', '-r', action='store_true')
    args = parser.parse_args()
    if args.file_path:
        s = collect_signals(args.file_path)
        d = decide(s)
        print(json.dumps({'file_path': args.file_path, 'decision': d, 'signals': s}, ensure_ascii=False, indent=2))
        return
    if args.dir_path:
        stats = evaluate_directory(args.dir_path, args.recursive)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    parser.print_help()

if __name__ == '__main__':
    main()
