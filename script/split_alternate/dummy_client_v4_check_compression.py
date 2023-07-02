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
from io import BytesIO
from PIL import Image
import os
import torchvision.transforms as transforms

def jpeg_size_compression(image_path, quality=40):
    image = Image.open(image_path)
    # compress image
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    image.save("./images/compressed_{}.jpg".format(quality), "JPEG", quality=quality)
    buffer.seek(0, os.SEEK_END)
    bytes_size = buffer.tell()
    return bytes_size

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
    ax.imshow(torch.transpose(inputs.squeeze(0), 0, 2).transpose(0, 1))
    # ...with detections
    bboxes = best_results[0]["boxes"].cpu().detach().numpy().tolist()
    classes = best_results[0]["labels"].cpu().detach().numpy().tolist()
    confidences = best_results[0]["scores"].cpu().detach().numpy().tolist()
    for idx in range(len(bboxes)):
        if confidences[idx] < 0.7:
            continue

        if classes[idx] > len(classes_to_labels):
            continue

        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

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

def split_preprocess_image(image):
    image = transform(image)
    my_image = torch.unsqueeze(torch.Tensor(image), 0)
    input = torch.tensor(my_image, dtype=torch.float32).cuda()
    return input

def get_original_image_size(image_path):
    image = Image.open(image_path)
    return image.size

def resize_image(image_path, desired_min_size):
    image = Image.open(image_path)
    width, height = image.size
    min_size = min(width, height)
    scale_factor = desired_min_size / float(min_size)
    scaled_width = int(round(width * scale_factor))
    scaled_height = int(round(height * scale_factor))
    image = image.resize((scaled_width,scaled_height), Image.ANTIALIAS)
    return image

def parse_to_onnx(model, input = False):
    if input is False:
        input = [torch.randn((1,3,300,300)).to("cuda")]
    model = model.eval().to("cuda")
    print("---------------------------------")
    print("onnx", input.shape)
    print("---------------------------------")
    traced_model = torch.jit.trace(model, input)   
    torch.onnx.export(traced_model,  # model being run
                        input,  # model input (or a tuple for multiple inputs)
                        "./exported_models/head_ubicomp.onnx",  # where to save the model (can be a file or file-like object)
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=13,  # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=['input'],  # the model's input names
                        output_names=['output0', 'output1', 'output2'],  # the model's output names]
                    )

if __name__ == '__main__':
    student_model = get_student_model(PARAMS['FASTER_RCNN_YAML'])
    client_model = model_1.ClientModel(student_model).eval().cuda()
    edge_model = model_1.ServerModel(student_model).eval().cuda()
    image_name = "./images/kitti_1.png"
    image_name = "./images/kitchen.jpg"
    # resize image to desired height
    desired_min_size = 400
    jpeg_compression_quality = 20

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    input_resized = resize_image(image_name, desired_min_size)
    input_resized.save("./images/input_resized.jpg")
    input = split_preprocess_image(input_resized)
    parse_to_onnx(client_model, input)

    out_head = client_model(input)
    out_edge = edge_model(*out_head)
    classes_to_labels= get_coco_object_dictionary()

    jpeg_compression_size = jpeg_size_compression(image_name, jpeg_compression_quality)
    # get size in bytes of tensor out_head
    check_size_tensor = out_head[0].to(torch.int8)
    image_size = check_size_tensor.element_size() * check_size_tensor.nelement()
    print("---------------------------------")
    print("original image size", get_original_image_size(image_name))
    print("---------------------------------")
    print("input_resized shape", transform(input_resized).shape)
    print("---------------------------------")
    print("input shape", input.shape)
    print("---------------------------------")
    print("out_head[0] shape", out_head[0].shape)
    print("---------------------------------")
    print("head size", image_size)
    print("jpeg_compression_size", jpeg_compression_size)
    print("[split compression rate]", image_size / jpeg_compression_size )
    print("---------------------------------")
    plot_results(out_edge, input.cpu(), classes_to_labels)

