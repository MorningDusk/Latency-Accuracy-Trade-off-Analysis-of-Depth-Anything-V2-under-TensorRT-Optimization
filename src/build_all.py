import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import os
import yaml
import torch
import torch.nn as nn
from depth_anything_v2.dpt import DepthAnythingV2
from export_onnx import DAv2Wrapper
from build_engine import build_engine

DEVICE = 'cuda'

def load_config():
    with open('configs/models.yaml', 'r') as f:
        return yaml.safe_load(f)
    
def export_one(model_key, cfg):
    m = cfg['models'][model_key]
    input_size = cfg['input_size']

    print(f'\n   Exporting {model_key}   ')
    model = DepthAnythingV2(
        encoder=m['encoder'],
        features=m['features'],
        out_channels=m['out_channels']
    )
    model.load_state_dict(torch.load(m['checkpoint'], map_location='cpu'))
    model = model.to(DEVICE).eval()

    wrapper = DAv2Wrapper(model).to(DEVICE).eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=DEVICE)

    onnx_path = f'onnx_models/dav2_{model_key}.onnx'
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=['input'], output_names=['depth'],
        opset_version=17, do_constant_folding=True
    )

    # Simplify
    import onnx
    from onnxsim import simplify
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check
    sim_path = f'onnx_models/dav2_{model_key}_sim.onnx'
    onnx.save(model_simp, sim_path)
    print(f'    ONNX: {sim_path}')

    # clear memory
    del model, wrapper
    torch.cuda.empty_cache()

    return sim_path

def build_all_engines(model_key, onnx_path, precisions):
    for prec in precisions:
        engine_path = f'engines/dav2_{model_key}_{prec}.engine'
        if os.path.exists(engine_path):
            print(f'    Skipping (exists): {engine_path}')
            continue
        print(f'    Building {prec}...')
        build_engine(onnx_path, engine_path, precision=prec)

def main():
    cfg = load_config()
    os.makedirs('engines', exist_ok=True)
    os.makedirs('onnx_models', exist_ok=True)

    for model_key in cfg['models'].keys():
        onnx_path = export_one(model_key, cfg)
        build_all_engines(model_key, onnx_path, cfg['precisions'])

    print("\n   All done    ")

if __name__ == '__main__':
    main()