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
    print(inputs.squeeze(0).shape)
    ax.imshow(torch.transpose(inputs.squeeze(0), 0, 2).transpose(0, 1))
    # ...with detections
    bboxes = best_results[0]["boxes"].cpu().detach().numpy().tolist()
    classes = best_results[0]["labels"].cpu().detach().numpy().tolist()
    confidences = best_results[0]["scores"].cpu().detach().numpy().tolist()
    for idx in range(len(bboxes)):
        if confidences[idx] < 0.7:
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
    image_name = "./images/kitchen.jpg"
    image_name = "./images/car320x240.jpg"
    image_name = "./images/car.jpg"
    image_name = "./images/car800x1280.jpg"
    image_name = "./images/car800x1280.jpg"
    image_name = "./images/kitti_1.png"

    image = Image.open(image_name)
    image = transform(image)
    my_image = torch.unsqueeze(torch.Tensor(image), 0)
    input = torch.tensor(my_image, dtype=torch.float32).cuda()

    print("input.shape", input.shape)
    out_head = client_model(input)
    print("out_head:", out_head)
    print(out_head[0].dtype)
    out_edge = edge_model(*out_head)
    print("out_edge:", out_edge)
    classes_to_labels= get_coco_object_dictionary()   
    plot_results(out_edge, input.cpu(), classes_to_labels)

    print("client_model:", client_model)