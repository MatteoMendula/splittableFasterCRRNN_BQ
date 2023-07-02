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
import torchvision.datasets as datasets
from tqdm import tqdm

def jpeg_size_compression(image_path, quality=90):
    from io import BytesIO # "import StringIO" directly in python2
    from PIL import Image
    import os
    im1 = Image.open(image_path)

    # here, we create an empty string buffer    
    buffer = BytesIO()
    im1.save(buffer, "JPEG", quality=quality)
    im1.save("./images/compressed_{}.jpg".format(quality), "JPEG", quality=quality)
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

def parse_to_onnx(model, input = False):
    if input is False:
        input = [torch.randn((1,3,300,300)).to("cuda")]
    model = model.eval().to("cuda")
    traced_model = torch.jit.trace(model, input)    
    torch.onnx.export(traced_model,  # model being run
                        input,  # model input (or a tuple for multiple inputs)
                        "./exported_models/head_juliano.onnx",  # where to save the model (can be a file or file-like object)
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=13,  # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=['input'],  # the model's input names
                        output_names=['output0', 'output1', 'output2'],  # the model's output names]
                    )

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

        if classes[idx] > len(classes_to_labels):
            continue

        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

if __name__ == '__main__':
    student_model = get_student_model(PARAMS['FASTER_RCNN_YAML'])
    client_model = model_1.ClientModel(student_model).eval().cuda()
    edge_model = model_1.ServerModel(student_model).eval().cuda()
    classes_to_labels= get_coco_object_dictionary()   
    print("--------------")
    print('all loaded :D') 
    print("--------------")

    path2data="/home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/coco/train2017"
    path2json="/home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/coco/annotations/instances_train2017.json"

    transforms_thing = transforms.Compose([transforms.ToTensor()])

    coco_train = datasets.CocoDetection(root = path2data,
                                annFile = path2json,
                                transform = transforms_thing)

    loss_ce = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        edge_model.parameters(),
        lr=0.001,
        momentum=0.9
    )


    train_loader = torch.utils.data.DataLoader(coco_train, batch_size=1, shuffle=True, num_workers=0)
    # initialize loss
    loss = torch.tensor(0).to(torch.float32).cuda()
    print(" --------------- Startin training ---------------")
    epochs = 10
    import matplotlib.pyplot as plt
    for epoch in range(epochs):
        step = 0
        for input, targets in tqdm(train_loader):
            inputs = input.cuda()

            out_head = client_model(inputs)
            out_edge = edge_model(*out_head)

            # plot_results(out_edge, inputs.cpu(), classes_to_labels)
            # print("out_head:", out_head)
            # print("out_edge:", out_edge)
            # print("targets", targets)

            ground_truth_boxes = torch.tensor([el["bbox"] for el in targets]).to(torch.float32).cuda()
            ground_truth_labels = torch.tensor([el["category_id"] for el in targets]).to(torch.float32).cuda()

            predicted_boxes = out_edge[0]["boxes"].to(torch.float32).cuda()
            predicted_labels = out_edge[0]["labels"].to(torch.float32).cuda()

            # calculate loss for boxes and labels with same size
            min_size = min(len(ground_truth_boxes), len(predicted_boxes))
            max_size = max(len(ground_truth_boxes), len(predicted_boxes))

            if len(predicted_boxes) == 0 and len(ground_truth_boxes) == 0:
                continue
            else:
                # if min_size > 0:
                l_mse = loss_mse(predicted_boxes[:min_size], ground_truth_boxes[:min_size])
                l_ce = loss_ce(predicted_labels[:min_size], ground_truth_labels[:min_size])

                # penalize missing and extra boxes
                l_mse += 1 * abs(min_size - max_size)
                l_ce += 1 * abs(min_size - max_size)

                loss = (l_mse + l_ce)
                
            if step % 100 == 0:
                print("step:", step)
                print("loss: ", loss.item(), loss)
                print("len(predicted_boxes):", len(predicted_boxes))
                print("len(ground_truth_boxes):", len(ground_truth_boxes))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device='cuda:0')

            step += 1

        edge_model.save("edge_model.pt")
            
        
