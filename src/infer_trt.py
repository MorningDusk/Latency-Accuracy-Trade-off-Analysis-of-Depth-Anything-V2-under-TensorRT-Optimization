import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
import numpy as np
import cv2
import torch
from load_pytorch import load_model
from verify_onnx import preprocess

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # I/O binding information
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        # memory allocation
        self.d_input = cuda.mem_alloc(
            int(np.prod(self.input_shape) * 4)  # fp32 size
        )
        self.d_output = cuda.mem_alloc(
            int(np.prod(self.output_shape) * 4)
        )
        self.stream = cuda.Stream()
        
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
    
    def infer(self, img_np):
        # img_np: [1, 3, H, W] float32
        img_np = np.ascontiguousarray(img_np)
        cuda.memcpy_htod_async(self.d_input, img_np, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        return output

def main():
    img = preprocess('test_images/sample.jpg')
    
    # FP16 TRT
    trt_engine = TRTInference('engines/dav2_vits_fp16.engine')
    trt_out = trt_engine.infer(img)
    
    # PyTorch
    pt_model = load_model()
    img_tensor = torch.from_numpy(img).cuda()
    with torch.no_grad():
        pt_out = pt_model(img_tensor).cpu().numpy()
    
    # compare
    diff = np.abs(trt_out - pt_out)
    print(f'TRT FP16 vs PyTorch:')
    print(f'  Max abs diff: {diff.max():.4f}')
    print(f'  Mean abs diff: {diff.mean():.4f}')
    print(f'  Relative error: {(diff.mean() / np.abs(pt_out).mean() * 100):.2f}%')
    
    # visualize
    depth = trt_out[0]
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite('results/raw/trt_fp16_output.png', depth_colored)
    print('Saved TRT FP16 output.')

if __name__ == '__main__':
    main()