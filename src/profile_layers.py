import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import json
import os
import argparse
import re
from collections import defaultdict
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from infer_trt import TRTInference

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class LayerProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.layer_times = defaultdict(list)
    
    def report_layer_time(self, layer_name, ms):
        self.layer_times[layer_name].append(ms)
    
    def get_summary(self):
        summary = {}
        for name, times in self.layer_times.items():
            arr = np.array(times)
            summary[name] = {
                'mean_ms': float(arr.mean()),
                'std_ms': float(arr.std()),
                'count': len(times),
                'total_ms': float(arr.sum()),
            }
        return summary
    
    def reset(self):
        self.layer_times.clear()

def categorize_layer(layer_name):
    name_lower = layer_name.lower()
    
    # === DPT Head (가장 먼저 체크) ===
    # Depth Anything V2의 DPT decoder 구성요소
    if any(k in name_lower for k in [
        'scratch', 'refinenet', 'output_conv', 
        'depth_head', 'fusion_block', 'reassemble',
        'resconfunit', 'projection_'
    ]):
        return 'DPT_Head'
    
    # === Backbone Attention ===
    # 표준 이름 + TRT Myelin fused 이름
    if any(k in name_lower for k in [
        # 표준 ONNX 이름
        'attn', 'attention', 'qkv', 'softmax',
        '/q_proj', '/k_proj', '/v_proj', 'proj_drop',
        # TRT Myelin: Multi-Head Attention fused
        '_gemm_mha', 'mha_v2',
        # TRT Myelin: Softmax fused 
        # (Max → Sub → Exp → Sum → Div → Mul 패턴)
        'maxrsubexpsumdivmul', 'maxsubexpsum',
    ]):
        return 'Backbone_Attention'
    
    # === Backbone FFN/MLP ===
    if any(k in name_lower for k in [
        # 표준 이름
        'mlp', 'fc1', 'fc2', 'ffn', 'gelu',
        'feedforward', 'intermediate',
        # TRT Myelin: Fully-Connected fused
        '__myl_fc', '_fc_myl',
    ]):
        return 'Backbone_FFN'
    
    # === Backbone Norm ===
    if any(k in name_lower for k in [
        # 표준 이름
        'layernorm', 'layer_norm', 'norm1', 'norm2', 
        '/ln_', 'batchnorm',
        # TRT Myelin: LayerNorm fused
        # (Mean → Sub → Square → Mean → Add ε → Sqrt → Div → ×γ → +β)
        'muladdmeansubmul', 'meansubmulmean',
    ]):
        return 'Backbone_Norm'
    
    # === Patch Embed ===
    if any(k in name_lower for k in [
        'patch_embed', 'cls_token', 'pos_embed',
        'patch_embedding', 'positional_embedding'
    ]):
        return 'Backbone_PatchEmbed'
    
    # === Backbone 기타 (transformer block 내) ===
    if any(k in name_lower for k in [
        'blocks.', 'block_', 'encoder.', 'transformer'
    ]):
        return 'Backbone_Other'
    
    # TRT Myelin이 합친 작은 op들 (chained __mye 패턴)
    # 예: __mye64100+__mye64088+__mye64076_myl1_98
    if '__mye' in name_lower and '+' in layer_name:
        return 'Backbone_Other'
    
    # === Tensor 조작 ===
    if any(k in name_lower for k in [
        'reshape', 'transpose', 'concat', 'slice', 
        'gather', 'cast', 'unsqueeze', 'squeeze',
    ]):
        return 'Tensor_Ops'
    
    return 'Other'


def profile_engine(engine_path, num_iter=100, num_warmup=20):
    print(f'\n=== Profiling {engine_path} ===')
    
    # Engine 로드
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # IProfiler 등록
    profiler = LayerProfiler()
    context.profiler = profiler
    
    # I/O 메모리 할당
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    # FP16 engine이어도 host 입력은 FP32로 줘도 됨 (TRT가 변환)
    d_input = cuda.mem_alloc(int(np.prod(input_shape) * 4))
    d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))
    stream = cuda.Stream()
    
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    # 더미 입력
    dummy = np.random.randn(*input_shape).astype(np.float32)
    cuda.memcpy_htod(d_input, dummy)
    
    # Warm-up (프로파일러 비활성화)
    context.profiler = None
    for _ in range(num_warmup):
        context.execute_async_v3(stream.handle)
    stream.synchronize()
    
    # 본 측정 (프로파일러 활성화)
    # 주의: profiler 콜백은 동기 실행에서만 작동. execute_v2 사용.
    context.profiler = profiler
    
    # 동기 실행을 위한 binding 리스트
    bindings = [int(d_input), int(d_output)]
    
    for _ in range(num_iter):
        # execute_v2는 동기. profiler 콜백이 정확히 작동.
        context.execute_v2(bindings)
    
    summary = profiler.get_summary()
    print(f'  Profiled {len(summary)} layers, {num_iter} iterations each.')
    
    return summary, num_iter


def aggregate_by_category(layer_summary):
    by_category = defaultdict(lambda: {
        'total_ms_per_inference': 0.0,
        'layer_count': 0,
        'layers': []
    })
    
    for layer_name, stats in layer_summary.items():
        category = categorize_layer(layer_name)
        # mean_ms는 한 inference당 평균이므로 그대로 더함
        by_category[category]['total_ms_per_inference'] += stats['mean_ms']
        by_category[category]['layer_count'] += 1
        by_category[category]['layers'].append({
            'name': layer_name,
            'mean_ms': stats['mean_ms'],
        })
    
    return dict(by_category)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vits', 
                        choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--precisions', nargs='+', 
                        default=['fp32', 'fp16'])
    parser.add_argument('--num_iter', type=int, default=100)
    args = parser.parse_args()
    
    os.makedirs('results/raw', exist_ok=True)
    os.makedirs('results/profile', exist_ok=True)
    
    all_results = {}
    
    for prec in args.precisions:
        engine_path = f'engines/dav2_{args.model}_{prec}.engine'
        if not os.path.exists(engine_path):
            print(f'Skipping (engine not found): {engine_path}')
            continue
        
        summary, n_iter = profile_engine(engine_path, num_iter=args.num_iter)
        by_cat = aggregate_by_category(summary)
        
        # Total latency 검증 (sanity check)
        total_per_inf = sum(c['total_ms_per_inference'] for c in by_cat.values())
        print(f'  Total latency from profiler: {total_per_inf:.3f} ms')
        
        # Save full layer-level data
        full_path = f'results/profile/{args.model}_{prec}_layers.json'
        with open(full_path, 'w') as f:
            json.dump({
                'model': args.model,
                'precision': prec,
                'num_iter': n_iter,
                'layers': summary,
            }, f, indent=2)
        
        # Save aggregated by category
        cat_path = f'results/profile/{args.model}_{prec}_categories.json'
        with open(cat_path, 'w') as f:
            # JSON 직렬화 위해 layers는 분리
            cat_summary = {
                cat: {
                    'total_ms_per_inference': data['total_ms_per_inference'],
                    'layer_count': data['layer_count'],
                    'percentage': data['total_ms_per_inference'] / total_per_inf * 100
                                   if total_per_inf > 0 else 0,
                }
                for cat, data in by_cat.items()
            }
            json.dump({
                'model': args.model,
                'precision': prec,
                'total_ms': total_per_inf,
                'categories': cat_summary,
            }, f, indent=2)
        
        # 출력
        print(f'\n  Category breakdown ({args.model}/{prec}):')
        for cat, data in sorted(by_cat.items(), 
                                 key=lambda x: -x[1]['total_ms_per_inference']):
            pct = data['total_ms_per_inference'] / total_per_inf * 100
            print(f'    {cat:25s}: {data["total_ms_per_inference"]:7.3f} ms '
                  f'({pct:5.2f}%) - {data["layer_count"]} layers')
        
        all_results[prec] = by_cat
    
    print(f'\nResults saved to results/profile/')


if __name__ == '__main__':
    main()