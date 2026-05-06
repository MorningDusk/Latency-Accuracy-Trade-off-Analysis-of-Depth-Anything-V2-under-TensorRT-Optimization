import numpy as np
import torch

def compute_depth_metrics(pred_disparity, gt_depth, valid_mask=None, max_depth=10.0, min_depth=1e-3):
    if isinstance(pred_disparity, torch.Tensor):
        pred_disparity = pred_disparity.cpu().numpy()
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.cpu().numpy()
    
    if valid_mask is None:
        valid_mask = (gt_depth > min_depth) & (gt_depth < max_depth)
    
    pred_d = pred_disparity[valid_mask].astype(np.float64)
    gt_d = gt_depth[valid_mask].astype(np.float64)
    
    gt_disparity = 1.0 / gt_d
    
    s, t = align_disparity_robust(pred_d, gt_disparity)
    pred_disparity_aligned = s * pred_d + t
    
    pred_disparity_aligned = np.clip(pred_disparity_aligned, 1e-6, None)
    pred_depth_aligned = 1.0 / pred_disparity_aligned
    pred_depth_aligned = np.clip(pred_depth_aligned, min_depth, max_depth)
    
    abs_rel = np.mean(np.abs(pred_depth_aligned - gt_d) / gt_d)
    rmse = np.sqrt(np.mean((pred_depth_aligned - gt_d) ** 2))
    log10 = np.mean(np.abs(np.log10(pred_depth_aligned) - np.log10(gt_d)))
    
    thresh = np.maximum(pred_depth_aligned / gt_d, gt_d / pred_depth_aligned)
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)
    
    return {
        'abs_rel': float(abs_rel),
        'rmse': float(rmse),
        'log10': float(log10),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
    }

def align_disparity_robust(pred, target):
    t_pred = np.median(pred)
    s_pred = np.median(np.abs(pred - t_pred)) + 1e-8
    t_target = np.median(target)
    s_target = np.median(np.abs(target - t_target)) + 1e-8
    
    scale = s_target / s_pred
    shift = t_target - scale * t_pred
    
    return scale, shift