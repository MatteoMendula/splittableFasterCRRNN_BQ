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

def predict(context, stream, d_input, batch, bindings, output, d_output): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    return output

# def single_prediction(engine_path, input, real_label):
#     cuda.init()
#     f = open(engine_path, "rb")
#     stream = cuda.Stream()
#     result = []
#     try:
#         runtime = trt.Runtime(trt.Logger(trt.Logger.VERBOSE))
#         engine = runtime.deserialize_cuda_engine(f.read())
#         labels = real_label.numpy()
#         output = np.empty([labels.shape[0], 10], dtype = np.float32)
#         d_input = cuda.mem_alloc(1 * input_batch.nbytes)
#         d_output = cuda.mem_alloc(1 * output.nbytes)
#         bindings = [int(d_input), int(d_output)]

#     print("input", input)
#     return result
     

def evaluate_trt(engine_path, dataloader):
    cuda.init()
    f = open(engine_path, "rb")
    stream = cuda.Stream()
    try:
        runtime = trt.Runtime(trt.Logger(trt.Logger.VERBOSE))
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        total = 0
        correct = 0

        for input, label in dataloader:
            input_batch = input.numpy()
            labels = label.numpy()
            output = np.empty([labels.shape[0], 10], dtype = np.float32)

            # Now allocate input and output memory, give TRT pointers (bindings) to it:
            d_input = cuda.mem_alloc(1 * input_batch.nbytes)
            d_output = cuda.mem_alloc(1 * output.nbytes)
            bindings = [int(d_input), int(d_output)]

            testing_shit = predict(context, stream, d_input, input_batch, bindings, output, d_output)
            for i, pred in enumerate(testing_shit):
                pred_label = (-pred).argsort()[0]
                correct += 1 if pred_label == label[i] else 0

            total += len(labels)

        return correct/total
    finally:
        f.close()

    
    # with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.VERBOSE)) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:


if __name__=='__main__':
        trt_file = "stupidmodel13.trt"
        test_dataset = CIFAR10(root='data/',download=True, train=False, transform=transforms.ToTensor())

        torch.manual_seed(43)
        batch_size=1
        test_loader = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)

        # input, real_label = next(iter(test_loader))   
        # single_prediction_result = single_prediction(trt_file, input, real_label)
        # print("single_prediction_result", single_prediction_result)

        test_acc = evaluate_trt(trt_file, test_loader)
        print("StupidModel TRT accuracy: {:.2f}%".format(100 * test_acc))
