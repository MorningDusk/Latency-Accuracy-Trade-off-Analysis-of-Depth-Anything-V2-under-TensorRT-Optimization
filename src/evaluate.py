import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import os
import json
import argparse
import numpy as np
import cv2
from datasets import load_from_disk
from infer_trt import TRTInference
from metrics import compute_depth_metrics

INPUT_SIZE = 518

def preprocess(img_pil):
    img = np.array(img_pil, dtype=np.uint8)  # dataset returns int64 list; cv2.resize requires uint8
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.transpose(2, 0, 1).astype(np.float32)
    img = np.expand_dims(img, 0)
    return np.ascontiguousarray(img)

def postprocess(depth_pred, target_size):
    """Resize predicted depth back to GT size."""
    return cv2.resize(depth_pred, (target_size[1], target_size[0]))

def evaluate(engine_path, dataset, max_samples=None):
    trt_engine = TRTInference(engine_path)
    
    all_metrics = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    for i in range(n):
        sample = dataset[i]
        img = sample['image']  # PIL
        gt_depth = np.array(sample['depth_map']).astype(np.float32)
        
        # Inference
        x = preprocess(img)
        depth_pred = trt_engine.infer(x)[0]  # [H, W]
        
        # Resize to GT shape
        depth_pred = postprocess(depth_pred, gt_depth.shape)
        
        # Compute metrics
        m = compute_depth_metrics(depth_pred, gt_depth, max_depth=10.0)
        all_metrics.append(m)
        
        if (i+1) % 50 == 0:
            print(f'  {i+1}/{n} processed')
    
    # Aggregate
    aggregated = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics]
        aggregated[key] = float(np.mean(vals))
        aggregated[f'{key}_std'] = float(np.std(vals))
    
    return aggregated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--precision', required=True)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='None for full set (654)')
    args = parser.parse_args()
    
    engine_path = f'engines/dav2_{args.model}_{args.precision}.engine'
    print(f'Evaluating {args.model} {args.precision}...')
    
    dataset = load_from_disk('data/nyu_v2/validation')
    metrics = evaluate(engine_path, dataset, max_samples=args.max_samples)
    
    print(f'  AbsRel: {metrics["abs_rel"]:.4f}')
    print(f'  RMSE: {metrics["rmse"]:.4f}')
    print(f'  delta1: {metrics["delta1"]:.4f}')
    
    out_path = f'results/raw/{args.model}_{args.precision}_accuracy.json'
    with open(out_path, 'w') as f:
        json.dump({
            'model': args.model,
            'precision': args.precision,
            **metrics,
        }, f, indent=2)
    print(f'  Saved to {out_path}')

if __name__ == '__main__':
    main()