import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import subprocess
import yaml

def main():
    with open('configs/models.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    for model_key in cfg['models'].keys():
        for prec in cfg['precisions']:
            print(f'\n>>> {model_key} / {prec}')
            try:
                subprocess.run([
                    'python', 'src/measure_latency.py',
                    '--model', model_key,
                    '--precision', prec,
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f'    FAILED: {e}')

if __name__ == '__main__':
    main()