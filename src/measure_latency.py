import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import time
import json
import os
import argparse
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
import pynvml
from infer_trt import TRTInference

def measure_latency(engine_path, num_warmup=50, num_iter=200):
    trt_engine = TRTInference(engine_path)
    input_shape = trt_engine.input_shape
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    cuda.memcpy_htod(trt_engine.d_input, dummy_input)
    
    # CUDA events
    start_event = cuda.Event()
    end_event = cuda.Event()
    
    # Warm-up
    for _ in range(num_warmup):
        trt_engine.context.execute_async_v3(trt_engine.stream.handle)
    trt_engine.stream.synchronize()
    
    latencies = []
    for _ in range(num_iter):
        start_event.record(trt_engine.stream)
        trt_engine.context.execute_async_v3(trt_engine.stream.handle)
        end_event.record(trt_engine.stream)
        end_event.synchronize()
        latency_ms = end_event.time_since(start_event)
        latencies.append(latency_ms)
    
    latencies = np.array(latencies)
    return {
        'mean_ms': float(latencies.mean()),
        'median_ms': float(np.median(latencies)),
        'std_ms': float(latencies.std()),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(latencies.min()),
        'max_ms': float(latencies.max()),
        'fps': float(1000.0 / latencies.mean()),
        'num_iter': num_iter,
        'num_warmup': num_warmup,
    }

def measure_memory_and_power(engine_path, num_iter=100):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    trt_engine = TRTInference(engine_path)
    input_shape = trt_engine.input_shape
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    cuda.memcpy_htod(trt_engine.d_input, dummy_input)
    
    # Warm-up
    for _ in range(20):
        trt_engine.context.execute_async_v3(trt_engine.stream.handle)
    trt_engine.stream.synchronize()
    
    power_samples = []
    
    mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    
    for i in range(num_iter):
        trt_engine.context.execute_async_v3(trt_engine.stream.handle)
        if i % 5 == 0:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_samples.append(power_mw / 1000.0)  # to W
    
    trt_engine.stream.synchronize()
    mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    
    pynvml.nvmlShutdown()
    
    return {
        'vram_used_MB': float((mem_after - mem_before) / 1024 / 1024),
        'vram_total_MB': float(mem_after / 1024 / 1024),
        'power_mean_W': float(np.mean(power_samples)),
        'power_max_W': float(np.max(power_samples)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--precision', required=True, choices=['fp32', 'fp16', 'int8'])
    parser.add_argument('--num_iter', type=int, default=200)
    args = parser.parse_args()
    
    engine_path = f'engines/dav2_{args.model}_{args.precision}.engine'
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f'Engine not found: {engine_path}')
    
    print(f'Measuring {args.model} {args.precision}...')
    
    lat = measure_latency(engine_path, num_iter=args.num_iter)
    mem = measure_memory_and_power(engine_path)
    
    result = {
        'model': args.model,
        'precision': args.precision,
        'engine_path': engine_path,
        **lat,
        **mem,
    }
    
    # GPU info
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    result['gpu_name'] = pynvml.nvmlDeviceGetName(handle)
    pynvml.nvmlShutdown()
    
    out_path = f'results/raw/{args.model}_{args.precision}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f'  Mean: {lat["mean_ms"]:.2f}ms ({lat["fps"]:.1f} FPS)')
    print(f'  P95: {lat["p95_ms"]:.2f}ms')
    print(f'  VRAM: {mem["vram_used_MB"]:.0f}MB')
    print(f'  Power: {mem["power_mean_W"]:.1f}W avg')
    print(f'  Saved to {out_path}')

if __name__ == '__main__':
    main()