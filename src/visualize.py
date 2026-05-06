import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk
from load_pytorch import load_model
from infer_trt import TRTInference
from evaluate import preprocess

INPUT_SIZE = 518

def colormap(depth, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.percentile(depth, 5)
    if vmax is None:
        vmax = np.percentile(depth, 95)
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

def main():
    sample_indices = [0, 100, 300, 500]

    dataset = load_from_disk('data/nyu_v2/validation')

    pt_model = load_model()
    trt_fp32 = TRTInference('engines/dav2_vits_fp32.engine')
    trt_fp16 = TRTInference('engines/dav2_vits_fp16.engine')

    fig, axes = plt.subplots(len(sample_indices), 5, figsize=(20, 4 * len(sample_indices)))

    col_titles = ['RGB Input', 'GT Depth', 'PyTorch', 'TRT FP32', 'TRT FP16']

    for row, idx in enumerate(sample_indices):
        sample = dataset[idx]
        img_pil = sample['image']
        gt_depth = np.array(sample['depth_map']).astype(np.float32)

        vmin = np.percentile(gt_depth[gt_depth > 0], 5)
        vmax = np.percentile(gt_depth[gt_depth > 0], 95)

        x = preprocess(img_pil)

        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x).cuda()).cpu().numpy()[0]
        
        pt_depth = 1.0 / np.clip(pt_out, 1e-3, None)
        pt_depth = cv2.resize(pt_depth, (gt_depth.shape[1], gt_depth.shape[0]))

        fp32_out = trt_fp32.infer(x)[0]
        fp32_depth = 1.0 / np.clip(fp32_out, 1e-3, None)
        fp32_depth = cv2.resize(fp32_depth, (gt_depth.shape[1], gt_depth.shape[0]))

        fp16_out = trt_fp16.infer(x)[0]
        fp16_depth = 1.0 / np.clip(fp16_out, 1e-3, None)
        fp16_depth = cv2.resize(fp16_depth, (gt_depth.shape[1], gt_depth.shape[0]))

        axes[row, 0].imshow(np.array(img_pil))
        axes[row, 1].imshow(colormap(gt_depth, vmin, vmax))
        axes[row, 2].imshow(colormap(pt_depth))
        axes[row, 3].imshow(colormap(fp32_depth))
        axes[row, 4].imshow(colormap(fp16_depth))

        for col in range(5):
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=14)
        
    plt.tight_layout()
    out_path = 'results/tables/qualitive_comparison.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {out_path}')

    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    print('Saved PDF version too')

if __name__ == '__main__':
    main()