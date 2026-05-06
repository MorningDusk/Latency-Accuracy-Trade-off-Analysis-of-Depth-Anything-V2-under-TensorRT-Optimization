import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, precision='fp16'):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError('ONNX parsing failed')
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30)  # 4GB
    )
    
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'fp32':
        pass  # default
    elif precision == 'int8':
        raise NotImplementedError('INT8 will be added later')
    
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError('Engine build failed')
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f'Engine saved to {engine_path}')

if __name__ == '__main__':
    os.makedirs('engines', exist_ok=True)
    
    # FP32 baseline
    build_engine(
        'onnx_models/dav2_vits_sim.onnx',
        'engines/dav2_vits_fp32.engine',
        precision='fp32'
    )
    
    # FP16
    build_engine(
        'onnx_models/dav2_vits_sim.onnx',
        'engines/dav2_vits_fp16.engine',
        precision='fp16'
    )