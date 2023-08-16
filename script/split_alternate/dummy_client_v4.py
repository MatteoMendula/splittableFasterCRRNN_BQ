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

# resize image to desired height
desired_height = 400
jpeg_compression_quality = 40

def jpeg_size_compression(image_path, quality=40, desired_height=400):
    image = Image.open(image_path)

    hpercent = (desired_height/float(image.size[1]))
    wsize = int((float(image.size[0])*float(hpercent)))
    # wsize = int(desired_height * 16. / 9.)
    # resize image and save
    image = image.resize((wsize,desired_height), Image.ANTIALIAS)

    # here, we create an empty string buffer
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    image.save("./images/compressed_{}.jpg".format(quality), "JPEG", quality=quality)
    # get buffer size in bytes
    buffer.seek(0, os.SEEK_END)
    return buffer.tell()

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
        y = y + 15
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x - 50, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig("./kitti_detection.jpg", bbox_inches='tight', pad_inches=0)
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

def parse_to_onnx(model, input = False):
    if input is False:
        input = [torch.randn((1,3,300,300)).to("cuda")]
    model = model.eval().to("cuda")
    print("onnx", input.shape)
    traced_model = torch.jit.trace(model, input)   
    torch.onnx.export(traced_model,  # model being run
                        input,  # model input (or a tuple for multiple inputs)
                        "./exported_models/head_demo.onnx",  # where to save the model (can be a file or file-like object)
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

    print("--------------")
    print('all loaded :D') 
    print("--------------")

    print(" --------------- Running inference ---------------")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_name = "./images/car.jpg"
    image_name = "./images/car4k.jpg"
    image_name = "./images/baseball.jpg"
    image_name = "./images/car400x234.jpg"
    image_name = "./images/car320x240.jpg"
    image_name = "./images/car800x1280.jpg"
    image_name = "./images/kitchen.jpg"
    image_name = "./images/matte.jpg"
    image_name = "./images/kitti_1.png"

    image = Image.open(image_name)

    # determining the height ratio
    hpercent = (desired_height/float(image.size[1]))
    wsize = int((float(image.size[0])*float(hpercent)))
    # resize image and save
    # wsize = int(desired_height * 16. / 9.)
    # image = image.resize((wsize,desired_height), Image.ANTIALIAS)

    image = transform(image)
    my_image = torch.unsqueeze(torch.Tensor(image), 0)
    input = torch.tensor(my_image, dtype=torch.float32).cuda()

    # save onnx
    parse_to_onnx(client_model, input)

    out_head = client_model(input)
    original_tensor = out_head[0].to(torch.float32)
    tensor = out_head[0].to(torch.uint8)
    tensor = tensor.to(torch.float32)
    results = [tensor, out_head[1], out_head[2], out_head[3], out_head[4], out_head[5]]
    out_edge = edge_model(*results)


    print("--------------")
    print("original_tensor", original_tensor)
    print("tensor", tensor)
    print("--------------")
    print("input shape", input.shape)
    print("out_head[0] shape", out_head[0].shape)
    print(out_head[0])
    print(out_head[1])
    print(out_head[2])
    print(out_head[3])
    print(out_head[4])
    print(out_head[5])
    


    classes_to_labels= get_coco_object_dictionary()
    plot_results(out_edge, input.cpu(), classes_to_labels)

    jpeg_compression_size = jpeg_size_compression(image_name, jpeg_compression_quality, desired_height)

    check_size_tensor = out_head[0].to(torch.int8)
    image_size = check_size_tensor.element_size() * check_size_tensor.nelement()
    print("[split compression rate]", image_size / jpeg_compression_size )

