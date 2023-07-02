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

trt_file = "mobilenetv2_base.trt"
# Define main data directory
DATA_DIR = './data/imagenette2-320'
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_dataloader = data.DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)

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

def evaluate_trt(engine_path, dataloader, batch_size):



    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.VERBOSE)) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
        total = 0
        correct = 0
        for images, labels in dataloader:
            input_batch = images.numpy()
            labels = labels.numpy()
            output = np.empty([batch_size, 10], dtype = np.float32)

            # Now allocate input and output memory, give TRT pointers (bindings) to it:
            d_input = cuda.mem_alloc(1 * input_batch.nbytes)
            d_output = cuda.mem_alloc(1 * output.nbytes)
            bindings = [int(d_input), int(d_output)]

            stream = cuda.Stream()
            preds = predict(context, stream, d_input, input_batch, bindings, output, d_output)
            pred_labels = []
            for pred in preds:
                pred_label = (-pred).argsort()[0]
                pred_labels.append(pred_label)

            total += len(labels)
            correct += (pred_labels == labels).sum()

    return correct/total

if __name__=='__main__':
        batch_size = 64
        test_acc = evaluate_trt(trt_file, val_dataloader, batch_size)
        print("Mobilenetv2 TRT Baseline accuracy: {:.2f}%".format(100 * test_acc))
