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
import torchvision.transforms.functional as F

from PIL import Image
import torchvision.transforms as transforms 

def jpeg_size_compression(image_path, quality=10):
    from io import BytesIO # "import StringIO" directly in python2
    from PIL import Image
    import os
    im1 = Image.open(image_path)

    # here, we create an empty string buffer    
    buffer = BytesIO()
    im1.save(buffer, "JPEG", quality=quality)
    # get buffer size in bytes
    buffer.seek(0, os.SEEK_END)
    return buffer.tell()

def load_model(model_config, device):
    if 'detection_model' not in model_config:
        print('No detection model specified in config')
        return load_detection_model(model_config, device)    # <--- here
    print('Detection model specified in config')
    return get_wrapped_detection_model(model_config, device)

def get_coco_object_dictionary():
    import os
    file_with_coco_names = "category_names.txt"

    if not os.path.exists(file_with_coco_names):
        print("Downloading COCO annotations.")
        import urllib
        import zipfile
        import json
        import shutil
        urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "cocoanno.zip")
        with zipfile.ZipFile("cocoanno.zip", "r") as f:
            f.extractall()
        print("Downloading finished.")
        with open("annotations/instances_val2017.json", 'r') as COCO:
            js = json.loads(COCO.read())
        class_names = [category['name'] for category in js['categories']]
        open("category_names.txt", 'w').writelines([c+"\n" for c in class_names])
        os.remove("cocoanno.zip")
        shutil.rmtree("annotations")
    else:
        class_names = open("category_names.txt").readlines()
        class_names = [c.strip() for c in class_names]
    return class_names

def plot_results(best_results, inputs, classes_to_labels):
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...

    ax.imshow(torch.transpose(inputs, 0, 2).transpose(0, 1))
    # ...with detections
    bboxes = best_results[0].cpu().detach().numpy().tolist()
    classes = best_results[1].cpu().detach().numpy().tolist()
    confidences = best_results[2].cpu().detach().numpy().tolist()
    for idx in range(len(bboxes)):
        if confidences[idx] < 0.7:
            continue
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

def get_student_model(yaml_file = PARAMS['FASTER_RCNN_YAML']):
    if yaml_file is None:
        return None

    config = yaml_util.load_yaml_file(os.path.expanduser(yaml_file))
    models_config = config['models']
    student_model_config = models_config['student_model']
    student_model = load_model(student_model_config, PARAMS['DETECTION_DEVICE']).eval()

    return student_model

if __name__ == '__main__':
    from copy import deepcopy
    student_model = get_student_model(PARAMS['FASTER_RCNN_YAML'])
    student_model_encoder = deepcopy(student_model)
    encoder = student_model.backbone.body.bottleneck_layer.encoder
    client_model = model_1.ClientModel(encoder).eval().cuda()
    edge_model = model_1.ServerModel(student_model).eval().cuda()

    #TODO: image things
    print("--------------")
    print('all loaded :D') 
    print("--------------")

    print(client_model)
    
    # ClientModel(
    # (transform): GeneralizedRCNNTransform(
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     Resize(min_size=(800,), max_size=1333, mode='bilinear')
    # )
    # (encoder): Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (2): ReLU(inplace=True)
    #     (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    #     (4): Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=False)
    #     (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (6): Conv2d(64, 256, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=False)
    #     (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (8): ReLU(inplace=True)
    #     (9): Conv2d(256, 64, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=False)
    #     (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (11): Conv2d(64, 1, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=False)
    # )
    # )

    print(" --------------- Running inference ---------------")
    client_model = client_model.eval().to("cuda")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_name = "./images/kitchen.jpg"
    image_name = "./images/car320x240.jpg"
    image_name = "./images/car.jpg"
    image_name = "./images/car800x1280.jpg"
    image_name = "./images/car800x1280.jpg"
    image_name = "./images/kitti_1.png"
    image = Image.open(image_name)
    image = transform(image)
    # image = torch.unsqueeze(torch.Tensor(image), 0)
    input = torch.tensor(image, dtype=torch.float32).cuda()
    image_sizes = [(input.shape[1], input.shape[2])]
    original_image_sizes = image_sizes
    secondary_image_size = torch.Size([input.shape[1], input.shape[2]])
    print("-------------------")

    out_head = client_model(input)

    # print("type out_head:", type(out_head))
    print("out_head[0]:", out_head[0])
    print("out_head[0].shape:", out_head[0].shape)
    print("out_head[0].dtype:", out_head[0].dtype)
    print("out_head:", out_head[1])
    print("out_head:", out_head[2])

    jpeg_compression_size = jpeg_size_compression(image_name, 80)
    print("JPEG compression size:", jpeg_compression_size)
    # detemine size of x in bytes
    check_size_tensor = out_head[0].to(torch.int8)
    image_size = check_size_tensor.element_size() * check_size_tensor.nelement()
    print("head tensor size:", image_size)
    print("[split compression rate]", jpeg_compression_size / image_size)

    print(" --------------- end inference ---------------")
    print(" --------------- start export head ---------------")
    traced_model = torch.jit.trace(client_model, input)   
    print("model traced")
    torch.jit.save(traced_model, './exported_models/head.pt')
    torch.onnx.export(traced_model,  # model being run
                    input,  # model input (or a tuple for multiple inputs)
                    "./exported_models/head.onnx",  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=13,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model's input names
                    output_names=['output0',],  # the model's output names]
                )
    
    print(" --------------- end export head ---------------")
    print("head[0]", out_head[0])
    print("head[1]", out_head[1])
    print("head[2]", out_head[2])
    print("image_sizes", image_sizes)
    print("original_image_sizes", original_image_sizes)
    print("secondary_image_size", secondary_image_size)

    image_sizes = torch.tensor(image_sizes[0], dtype=torch.float32).cuda()
    original_image_sizes = torch.tensor(original_image_sizes[0], dtype=torch.float32).cuda()
    secondary_image_size = torch.tensor(secondary_image_size, dtype=torch.float32).cuda()

    print(" --------------- start export edge ---------------")
    # traced_model = torch.jit.trace(edge_model, (out_head[0], out_head[1], out_head[2], image_sizes, original_image_sizes, secondary_image_size))
    # print("model traced")
    # m = torch.jit.script( model_1.ServerModel(student_model).eval())
    # torch.jit.save(m, './exported_models/edge.pt')
    print(" --------------- end export edge ---------------")

    quantized_tensor = out_head[0]
    out_edge = edge_model(quantized_tensor, out_head[1], out_head[2], image_sizes, original_image_sizes, secondary_image_size)
    print("out_edge:", out_edge)
    classes_to_labels= get_coco_object_dictionary()   
    plot_results(out_edge, input.cpu(), classes_to_labels)


