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

from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,FasterRCNN_MobileNet_V3_Large_FPN_Weights,fasterrcnn_mobilenet_v3_large_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from pycocotools.coco import COCO
from copy import deepcopy
from pycocotools.cocoeval import COCOeval
import cv2
from typing import Union
import warnings
import os
import json
warnings.filterwarnings('ignore')

import itertools
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

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

def get_student_model(yaml_file = PARAMS['FASTER_RCNN_YAML']):
    if yaml_file is None:
        return None

    config = yaml_util.load_yaml_file(os.path.expanduser(yaml_file))
    models_config = config['models']
    student_model_config = models_config['student_model']
    student_model = load_model(student_model_config, PARAMS['DETECTION_DEVICE']).eval()

    return student_model


transform = transforms.Compose([
    transforms.ToTensor()
])

def evaluate_coco(img_path, set_name, image_ids, coco, model, weights, threshold=0.05):
    results = []

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']
        #print(image_path)
        #weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        image = Image.open(image_path)
        image = transform(image)
        # image = torch.unsqueeze(torch.Tensor(image), 0)
        input = torch.tensor(image, dtype=torch.float32).cuda()

        print("image_path: ", image_path)

        image_sizes = [(input.shape[1], input.shape[2])]
        original_image_sizes = image_sizes
        secondary_image_size = torch.Size([input.shape[1], input.shape[2]])
        image_sizes = torch.tensor(image_sizes[0], dtype=torch.float32).cuda()
        original_image_sizes = torch.tensor(original_image_sizes[0], dtype=torch.float32).cuda()
        secondary_image_size = torch.tensor(secondary_image_size, dtype=torch.float32).cuda()

        #x[0].cuda()
        #print(x.size())
        #features, regression, classification, anchors = model(x)

        if image.shape[0] != 3:
            continue

        out_head = model[0](input)
        pred = model[1](out_head[0], out_head[1], out_head[2], image_sizes, original_image_sizes, secondary_image_size)

        rois = pred[0]
        class_ids = pred[1]
        scores = pred[2]

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                #print(image_result)
                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f"{set_name}_bbox_results.json"

    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

def _eval(coco_gt, image_ids, pred_json_path):
    cat_num=[1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72]
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    for e in cat_num:
        print("----------------------------------------------")
        print("Category "+ str(e))
        print("----------------------------------------------")
        coco_eval.params.catIds = [e]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    print("----------------------------------------------")
    print("Overall")
    print("----------------------------------------------")
    E = COCOeval(coco_gt, coco_pred, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f}".format(E.stats[0]))


if __name__ == '__main__':
    path_to_annotations="/home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/coco/"
    coco_annotation_file_path = path_to_annotations+"annotations/instances_val2017.json"
    VAL_GT = path_to_annotations+"annotations/instances_val2017.json"
    VAL_IMGS = path_to_annotations+"val2017/"
    MAX_IMAGES = 50000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    SET_NAME="yoshi"
    use_cuda = True
    override_prev_results = True   
    if override_prev_results or not os.path.exists(f"{SET_NAME}_bbox_results.json"):
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        student_model = get_student_model(PARAMS['FASTER_RCNN_YAML'])
        student_model_encoder = deepcopy(student_model)
        encoder = student_model.backbone.body.bottleneck_layer.encoder
        client_model = model_1.ClientModel(encoder).eval().cuda()
        edge_model = model_1.ServerModel(student_model).eval().cuda()

        model = (client_model, edge_model)

        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model, weights)

    _eval(coco_gt, image_ids, "{}_bbox_results.json".format(SET_NAME))
