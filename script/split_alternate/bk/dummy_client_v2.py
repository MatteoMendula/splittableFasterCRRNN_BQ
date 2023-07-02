import torch
import os
import sys
import time

from split_models import model_1
from params import PARAMS, CURR_DATE, DESIRED_CLASSES
from torchdistill.common.main_util import load_ckpt
from torchdistill.models.registry import get_model

from torchdistill.common import yaml_util
from sc2bench.models.detection.registry import load_detection_model
from sc2bench.models.detection.wrapper import get_wrapped_detection_model

from PIL import Image
import torchvision.transforms as transforms

import torch_tensorrt

def load_model(model_config, device):
    if 'detection_model' not in model_config:
        print('No detection model specified in config')
        return load_detection_model(model_config, device)    # <--- here
    print('Detection model specified in config')
    return get_wrapped_detection_model(model_config, device)

def get_student_model(yaml_file = PARAMS['FASTER_RCNN_YAML']):
    if yaml_file is None:
        return None

    config = yaml_util.load_yaml_file(os.path.expanduser(yaml_file))
    models_config = config['models']
    student_model_config = models_config['student_model']
    student_model = load_model(student_model_config, PARAMS['DETECTION_DEVICE']).eval()

    return student_model

if __name__ == '__main__':
    student_model = get_student_model(PARAMS['FASTER_RCNN_YAML'])
    client_model = model_1.ClientModel(student_model).eval().cuda()
    edge_model = model_1.ServerModel(student_model).eval().cuda()

    #TODO: image things
    print("--------------")
    print('all loaded :D') 
    print("--------------")

    client_model.eval()

    print(" --------------- Running inference ---------------")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open("car800x1280.jpg")
    image = transform(image)
    my_image = torch.unsqueeze(torch.Tensor(image), 0)
    input = torch.tensor(my_image, dtype=torch.float32).cuda()

    print("input.shape", input.shape)
    out_head = client_model(input)
    print("out_head:", out_head)
    print(out_head[0].dtype)
    out_edge = edge_model(*out_head)
    print("out_edge:", out_edge)

    # dummy_input = torch.randn(1, 3, 300, 300)
    # torch.onnx.export(client_model, dummy_input, "client320x240.onnx")

    # image_size = [(int(out_head[0][0][0][340]), int(out_head[0][0][1][340]))]
    # original_image_size = [(int(out_head[0][0][2][340]), int(out_head[0][0][3][340]))]
    # secondary_image_size = torch.Size([int(out_head[0][0][4][340]), int(out_head[0][0][5][340])])
    # out_tensor = torch.tensor(out_head[:, :, :, :-1], dtype=torch.float32)

    # start_time = time.time()
    # out_edge = edge_model(out_tensor, image_size, original_image_size, secondary_image_size)
    # print("time tail", time.time() - start_time)
    # print("out decoder", out_edge)

    # image_size = [(int(out_head[0][0][0][340]), int(out_head[0][0][1][340]))]
    # original_image_size = [(int(out_head[0][0][2][340]), int(out_head[0][0][3][340]))]
    # secondary_image_size = torch.Size([int(out_head[0][0][4][340]), int(out_head[0][0][5][340])])

    # image_size = [(736, 1333)]
    # original_image_size = [(743, 1344)]
    # secondary_image_size = torch.Size([736, 1344])

    # image_size = [(800, 1066)]
    # original_image_size = [(240, 320)]
    # secondary_image_size = torch.Size([800, 1088])

    # print("image_size", image_size)
    # print("original_image_size", original_image_size)
    # print("secondary_image_size", secondary_image_size)

    # torch.save(out_head, 'out_head_original.pt')
    # loaded_tensor = torch.load('out_head_original.pt')

    # print("loaded_tensor", loaded_tensor)

    # # object_size = sys.getsizeof(out_head)
    # # print("outencoder object_size", object_size)
    # # # print("type out_head", type(out_head[0]))
    # out_tail_loaded = edge_model(*out_head)
    # start_time = time.time()
    # out_tail_loaded = edge_model(out_head[0], image_size, original_image_size, secondary_image_size)
    # print("time tail", time.time() - start_time)
    # print("out decoder", out_tail_loaded)

    # image_size = [(int(loaded_tensor[0][0][0][340]), int(loaded_tensor[0][0][1][340]))]
    # original_image_size = [(int(loaded_tensor[0][0][2][340]), int(loaded_tensor[0][0][3][340]))]
    # secondary_image_size = torch.Size([int(loaded_tensor[0][0][4][340]), int(loaded_tensor[0][0][5][340])])
    # out_tensort = torch.tensor(loaded_tensor[:, :, :, :-1], dtype=torch.float32)

    # out_tail_loaded = edge_model(out_tensort, image_size, original_image_size, secondary_image_size)
    # print("out decoder", out_tail_loaded)

    # output_names = []
    # output_names.append("features_tensort")
    # output_names.append("features_scale")
    # output_names.append("features_zero_point")
    # # output_names.app"d("original_image_sizes_0")
    # output_names.append("original_image_sizes_1")
    # output_names.append("secondary_image_size_0")
    # output_names.append("secondary_image_size_1")

    # torch.onnx.export(client_model,                 # model being run
    #               input,                                # model input (or a tuple for multiple inputs)
    #               "client320x240.onnx",                    # where to save the model (can be a file or file-like object)
    #               export_params=True,               # store the trained parameter weights inside the model file
    #               opset_version=11,                 # the ONNX version to export the model to
    #               do_constant_folding=True,         # whether to execute constant folding for optimization
    #               input_names = ['input'],          # the model's input names
    #             #   output_names = output_names,        # the model's output names
    #             #   dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}}    # variable length axes
    # )    

    # torch.onnx.export(
    #     client_model,
    #     input,
    #     "client320x240.onnx",
    #     verbose=False,
    #     input_names = ['input'],
    #     opset_version=13,
    #     do_constant_folding = False
    # )
    
    # 
    # torch.onnx.export(edge_model,                 # model being run
    #               out_head,                                # model input (or a tuple for multiple inputs)
    #               "edge320x240.onnx",                    # where to save the model (can be a file or file-like object)
    #               export_params=True,               # store the trained parameter weights inside the model file
    #               opset_version=13,                 # the ONNX version to export the model to
    #               do_constant_folding=True,         # whether to execute constant folding for optimization
    #               input_names = ['input'],          # the model's input names
    #             #   output_names = output_names,        # the model's output names
    #             #   dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}}    # variable length axes
    # )    
    
    # import onnx
    # onnx_model = onnx.load("client.onnx")
    # onnx.checker.check_model(onnx_model)

    # import onnxruntime as ort
    # # # # import numpy as np

    # ort_sess = ort.InferenceSession('client.onnx')
    # outputs = ort_sess.run(None, {'input': input.detach().numpy()})
    # print("outputs", outputs)

    # print("outputs shape", outputs[0].shape)
    # print("size outputs", sys.getsizeof(outputs))
    # print("outputs len", len(outputs))
    # print("outputs shape", outputs[0].shape)

    # for i in range(len(outputs)):
    #     print("output", i, outputs[i])
    #     if i == 0:
    #         print("output shape", i, outputs[i].shape)

    # quantized_tensor = torch.from_numpy(outputs[0])
    # scale = torch.from_numpy(outputs[1])
    # zero_point = torch.from_numpy(outputs[2])

    # image_sizes = [(outputs[4].item(), outputs[3].item())]
    # original_image_sizes = [(outputs[6].item(), outputs[5].item())]
    # secondary_image_size = torch.Size([outputs[8].item(), outputs[7].item()])

    # print("quantized_tensor", quantized_tensor)
    # print("scale", scale)
    # print("zero_point", zero_point)
    # print("image_sizes", image_sizes)
    # print("original_image_sizes", original_image_sizes)
    # print("secondary_image_size", secondary_image_size)

    # out_tail_loaded = edge_model(quantized_tensor, scale, zero_point, image_sizes, original_image_sizes, secondary_image_size)
    # print("out_tail_loaded", out_tail_loaded)

    # output 0 [[[[182 180 180 ... 180 180  92]
    # [179  95  91 ...  97 104   7]
    # [181 114 110 ...  97 104   4]
    # ...
    # [180  96  96 ...  97 103   3]
    # [179  98  96 ...  97 103   6]
    # [ 93   2   0 ...   0   3   4]]]]
    # output shape 0 (1, 1, 340, 196)

    # output 1 0.556861
    # output 2 92

    # output 3 1333
    # output 4 749

    # output 5 1280
    # output 6 720

    # output 7 1344
    # output 8 768

    # # server model expects features, targets, image_sizes, original_image_sizes, secondary_image_size
    # # outputs = edge_model(features)

    # print("----------------- loading out head from trt output file")
    # loaded_tensor_original = torch.load('out_head_original.pt')
    # loaded_tensor = torch.load('out_head.pt')[0]
    # print("loaded_tensor", loaded_tensor)
    # print("loaded_tensor original shape", loaded_tensor_original.shape)
    # print("loaded_tensor", loaded_tensor)
    # print("loaded_tensor shape", loaded_tensor.shape)

    # diff = loaded_tensor_original - torch.tensor(loaded_tensor)
    # print("diff:", diff)
    # print("sum diff:", torch.sum(diff))

    # image_size = [(int(loaded_tensor[0][0][0][340]), int(loaded_tensor[0][0][1][340]))]
    # original_image_size = [(int(loaded_tensor[0][0][2][340]), int(loaded_tensor[0][0][3][340]))]
    # secondary_image_size = torch.Size([int(loaded_tensor[0][0][4][340]), int(loaded_tensor[0][0][5][340])])

    # image_size = [(736, 1333)]
    # original_image_size = [(743, 1344)]
    # secondary_image_size = torch.Size([736, 1344])

    # out_tensort = torch.tensor(loaded_tensor[:, :, :, :-1], dtype=torch.float32)

    # print("image_size", image_size)
    # print("original_image_size", original_image_size)
    # print("secondary_image_size", secondary_image_size)
    # print("out_tensort", out_tensort)

    # out_tail_loaded = edge_model(out_tensort, image_size, original_image_size, secondary_image_size)
    # print("out decoder", out_tail_loaded)

    # sum diff: tensor(-3406882.5000, grad_fn=<SumBackward0>)
    # sum diff: tensor(-38816.5859, grad_fn=<SumBackward0>)
