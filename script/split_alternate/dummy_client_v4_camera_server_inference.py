import os
from split_models import model_1
from params import PARAMS, CURR_DATE, DESIRED_CLASSES

from torchdistill.common import yaml_util
from sc2bench.models.detection.registry import load_detection_model
from sc2bench.models.detection.wrapper import get_wrapped_detection_model
from PIL import Image
import os
import time
import torch
from io import BytesIO 
from PIL import Image
import os
from flask import Flask, request, jsonify
import numpy as np

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

student_model = get_student_model(PARAMS['FASTER_RCNN_YAML'])
edge_model = model_1.ServerModel(student_model).eval().cuda()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    file_1 = request.files['0']
    file_2 = request.files['1']
    file_3 = request.files['2']
    file_4 = request.files['3']
    file_5 = request.files['4']
    file_6 = request.files['5']

    buffer_1 = BytesIO()
    buffer_1.write(file_1.read())
    buffer_1.seek(0)
    numpy_array_1 = np.load(buffer_1)
    torch_tensor_1 = torch.from_numpy(numpy_array_1).to(torch.float32)

    buffer_2 = BytesIO()
    buffer_2.write(file_2.read())
    buffer_2.seek(0)
    numpy_array_2 = np.load(buffer_2)
    torch_tensor_2 = torch.from_numpy(numpy_array_2).to(torch.float32)

    buffer_3 = BytesIO()
    buffer_3.write(file_3.read())
    buffer_3.seek(0)
    numpy_array_3 = np.load(buffer_3)
    torch_tensor_3 = torch.from_numpy(numpy_array_3).to(torch.float32)

    buffer_4 = BytesIO()
    buffer_4.write(file_4.read())
    buffer_4.seek(0)
    numpy_array_4 = np.load(buffer_4)
    torch_tensor_4 = torch.from_numpy(numpy_array_4).to(torch.float32)

    buffer_5 = BytesIO()
    buffer_5.write(file_5.read())
    buffer_5.seek(0)
    numpy_array_5 = np.load(buffer_5)
    torch_tensor_5 = torch.from_numpy(numpy_array_5).to(torch.float32)

    buffer_6 = BytesIO()
    buffer_6.write(file_6.read())
    buffer_6.seek(0)
    numpy_array_6 = np.load(buffer_6)
    torch_tensor_6 = torch.from_numpy(numpy_array_6).to(torch.float32)

    out_head = (torch_tensor_1, torch_tensor_2, torch_tensor_3, torch_tensor_4, torch_tensor_5, torch_tensor_6)

    prediction = edge_model(*out_head)

    my_prediction = {}
    my_prediction['detection'] = {}
    my_prediction['detection']["boxes"] = prediction[0]["boxes"].cpu().detach().numpy().tolist()
    my_prediction['detection']["labels"] = prediction[0]["labels"].cpu().detach().numpy().tolist()
    my_prediction['detection']["scores"] = prediction[0]["scores"].cpu().detach().numpy().tolist()

    return jsonify(my_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)

    


  