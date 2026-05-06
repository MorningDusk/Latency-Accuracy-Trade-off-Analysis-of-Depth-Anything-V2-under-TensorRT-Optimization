import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import cv2
from datasets import load_from_disk
from infer_trt import TRTInference
from evaluate import preprocess

def main():
    dataset = load_from_disk('data/nyu_v2/validation')
    sample = dataset[0]
    img = sample['image']
    gt_depth = np.array(sample['depth_map']).astype(np.float32)

    print('     GT Depth Analysis       ')
    print(f'Shape: {gt_depth.shape}')
    print(f'dtype: {gt_depth.dtype}')
    print(f'Min (non-zero): {gt_depth[gt_depth > 0].min():.4f}')
    print(f'Max: {gt_depth.max():.4f}')
    print(f'Mean: {gt_depth[gt_depth > 0].mean():.4f}')
    print(f'Median: {np.median(gt_depth[gt_depth > 0]):.4f}')
    print(f'Zero ratio: {(gt_depth == 0).mean():.2%}')

    print('\n   Prediction Analysis')
    trt_engine = TRTInference('engines/dav2_vits_fp16.engine')
    x = preprocess(img)
    pred = trt_engine.infer(x)[0]
    pred = cv2.resize(pred, (gt_depth.shape[1], gt_depth.shape[0]))

    print(f'Pred shape: {pred.shape}')
    print(f'Pred min: {pred.min():.4f}')
    print(f'Pred max: {pred.max():.4f}')
    print(f'Pred mean: {pred.mean():.4f}')

    valid = (gt_depth > 0.001) & (gt_depth < 10.0)
    if valid.sum() > 0:
        gt_v = gt_depth[valid]
        pred_v = pred[valid]

        corr_direct = np.corrcoef(pred_v, gt_v)[0, 1]
        corr_inverse = np.corrcoef(pred_v, 1.0 / (gt_v + 1e-6))[0, 1]

        print(f'\nCorrelation pred vs gt:   {corr_direct:+.4f}')
        print(f'Correlation pred vs 1/gt:   {corr_inverse:.4f}')

        near_mask = (gt_depth > 0.5) & (gt_depth < 1.5)
        far_mask = (gt_depth > 4.0) & (gt_depth < 6.0)
        if near_mask.sum() > 100 and far_mask.sum() > 100:
            print(f'\nNear mask pred mean: {pred[near_mask].mean():.4f}')
            print(f'\nFar mask pred mean: {pred[far_mask].mean():.4f}')

if __name__ == '__main__':
    main()