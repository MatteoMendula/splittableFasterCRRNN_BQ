# Import needed libraries and define the evaluate function
import pycuda.driver as cuda
import pycuda.autoinit
import time 
import torchvision.transforms as transforms
import tensorrt as trt
import numpy as np
import argparse 
from PIL import Image
import torch

host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []

class SplitDet():
    def __init__(self, engine_path, precision):
        print("building engine...")
        print("engine_path:", engine_path)
        self.engine_path = engine_path
        self.precision = precision
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, 'rb') as f:
            serialized_engine = f.read()
        print("deserializing engine...")
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.batch_size = self.engine.max_batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print("allocating buffers...")

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

    def preprocess_image(self, image):
        image = self.transform(image)
        if self.precision == 'fp16':
            input = torch.tensor(image, dtype=torch.float16)
        else:
            input = torch.tensor(image, dtype=torch.float32)
        return input.unsqueeze(0).ravel().cpu()
    
    def postprocess(self, output):
        tensor = output[0].reshape(self.batch_size, 1, 124, 164)
        # parse tensor to int8
        tensor = tensor.astype(np.int8)
        result = (tensor, output[1], output[2], output[3], output[4], output[5])
        return result
    
    def inference(self, image):
        input = self.preprocess_image(image)
        np.copyto(host_inputs[0], input)
        stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        t1 = time.time()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
        cuda.memcpy_dtoh_async(host_outputs[2], cuda_outputs[2], stream)
        cuda.memcpy_dtoh_async(host_outputs[3], cuda_outputs[3], stream)
        cuda.memcpy_dtoh_async(host_outputs[4], cuda_outputs[4], stream)
        cuda.memcpy_dtoh_async(host_outputs[5], cuda_outputs[5], stream)
        stream.synchronize()
        t2 = time.time()
        result = self.postprocess(host_outputs)
        return result, t2-t1