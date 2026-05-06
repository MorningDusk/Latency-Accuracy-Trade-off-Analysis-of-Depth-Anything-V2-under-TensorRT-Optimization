import sys
import torch
import torch.nn as nn
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda'
INPUT_SIZE = 518   # Standard Input Size of DAv2

class DAv2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # patch_h, patch_w calculates inside of DAv2
        depth = self.model.forward(x)    # [B, H, W]
        return depth

def export_onnx():
    # load model
    model = DepthAnythingV2(
        encoder='vits', features=64,
        out_channels=[48, 96, 192, 384]
    )
    checkpoint_path = _project_root / 'checkpoints' / 'depth_anything_v2_vits.pth'
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'))
    model = model.to(DEVICE).eval()

    wrapper = DAv2Wrapper(model).to(DEVICE).eval()

    # input dummy
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)

    # onnx export
    onnx_dir = _project_root / 'onnx_models'
    onnx_dir.mkdir(exist_ok=True)
    onnx_path = onnx_dir / 'dav2_vits.onnx'

    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=['input'], output_names=['depth'], opset_version=17,
        dynamic_axes=None, do_constant_folding=True
    )

    # Verify ONNX
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print('ONNX model is valid')

    # Simplify
    from onnxsim import simplify
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, str(onnx_dir / 'dav2_vits_sim.onnx'))
    print('Simplified ONNX saved.')

if __name__ == '__main__':
    export_onnx()
