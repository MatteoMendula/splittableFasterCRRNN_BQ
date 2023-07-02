import pycuda.driver as cuda
import pycuda.autoinit
import time
import os
import torch
import tensorrt as trt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import sys 
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

batch = 1
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []

def PrepareEngine(engine_path):
    with open(engine_path, 'rb') as f:
        serialized_engine = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # create buffer
    for index, binding in enumerate(engine):
        print('bingding shape:', binding, engine.get_binding_shape(binding))
        print('bingding type:', str(binding))
        # print('tensor shape:', binding, engine.get_tensor_shape(binding))
        # print("name", engine.get_tensor_name(index))
        # size = trt.volume(engine.get_tensor_shape(binding)) * batch

        size = trt.volume(engine.get_binding_shape(binding)) * batch
        host_mem = cuda.pagelocked_empty(shape=[size],dtype=np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(cuda_mem))
        # if engine.get_tensor_mode(binding)==trt.TensorIOMode.INPUT:
        if 'input' in str(binding).lower():
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)

    return engine

def Inference(tensor_x, engine, stream, context):
    context.push()
    np.copyto(host_inputs[0], tensor_x)
    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    # print("execute times "+str(time.time()-start_time))
    latency = time.time()-start_time
    output = host_outputs[0]
    context.pop()
    return (np.argmax(output), latency)


def evaluate_trt(engine_path, dataloader):
    correct = 0
    total = 0
    latency = 0
    engine = PrepareEngine(engine_path)
    stream = cuda.Stream()
    context = engine.create_execution_context()
    print("engine prepared")    
    time.sleep(3)
    for input, label in dataloader:
        input_batch = input.numpy()
        testing_shit = Inference(input_batch.ravel(), engine, stream, context)
        correct += 1 if testing_shit[0] == label.item() else 0
        latency += testing_shit[1]
        total += 1

    return (correct/total, latency/total)

if __name__=='__main__':
        trt_file = "stupidmodel13.trt"
        test_dataset = CIFAR10(root='data/',download=True, train=False, transform=transforms.ToTensor())

        torch.manual_seed(43)
        batch_size=1
        test_loader = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)
        test_acc = evaluate_trt(trt_file, test_loader)
        print("StupidModel TRT accuracy: {:.2f}%".format(100 * test_acc[0]))
        print("Avg latency: ", test_acc[1])
