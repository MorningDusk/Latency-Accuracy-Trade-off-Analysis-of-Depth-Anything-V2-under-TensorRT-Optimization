import sys
from pathlib import Path
import onnxruntime as ort
import torch
import numpy as np
import cv2

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from load_pytorch import load_model

INPUT_SIZE = 518

def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.transpose(2, 0, 1).astype(np.float32)
    img = np.expand_dims(img, 0)    # [1, 3, H, W]
    return img

def main():
    img = preprocess('test_images/sample.jpg')

    # deduct ONNX
    sess = ort.InferenceSession(
        'onnx_models/dav2_vits_sim.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    onnx_out = sess.run(['depth'], {'input': img})[0]

    # deduct PyTorch
    pt_model = load_model()
    img_tensor = torch.from_numpy(img).cuda()
    with torch.no_grad():
        pt_out = pt_model(img_tensor).cpu().numpy()
    
    # compare
    diff = np.abs(onnx_out - pt_out)
    print(f'Max abs diff: {diff.max():.6f}')
    print(f'Mean abs diff: {diff.mean():.6f}')
    print(f'PyTorch range: [{pt_out.min():.3f}, pt_out.max():.3f]')
    print(f'ONNX range: [{onnx_out.min():.3f}, {onnx_out.max():.3f}]')

if __name__ == '__main__':
    main()