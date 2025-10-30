/src/model/trt_engine.py

# sederhana: pembuatan engine dengan TensorRT (requres trt bindings)
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def build_engine_from_onnx(onnx_path: str, max_batch_size: int = 1):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Failed to parse ONNX")

    builder.max_batch_size = max_batch_size
    engine = builder.build_cuda_engine(network)
    return engine
