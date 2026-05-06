import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import json
from collections import defaultdict
from profile_layers import categorize_layer

def main():
    with open('results/profile/vits_fp16_layers.json') as f:
        data = json.load(f)
    
    layers = data['layers']
    
    # 카테고리별로 레이어 분류
    by_cat = defaultdict(list)
    for name, stats in layers.items():
        cat = categorize_layer(name)
        by_cat[cat].append((name, stats['mean_ms']))
    
    # 카테고리별 출력 (latency 큰 순)
    for cat in sorted(by_cat.keys()):
        layers_in_cat = sorted(by_cat[cat], key=lambda x: -x[1])
        total = sum(t for _, t in layers_in_cat)
        print(f'\n{"="*70}')
        print(f'{cat}: {total:.4f} ms total, {len(layers_in_cat)} layers')
        print(f'{"="*70}')
        # 상위 15개만 출력 (전체는 너무 많음)
        for name, t in layers_in_cat[:15]:
            print(f'  {t:7.4f} ms  |  {name}')
        if len(layers_in_cat) > 15:
            print(f'  ... and {len(layers_in_cat) - 15} more')


if __name__ == '__main__':
    main()