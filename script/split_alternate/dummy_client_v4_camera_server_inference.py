import os
from split_models import model_1
from params import PARAMS, CURR_DATE, DESIRED_CLASSES

from torchdistill.common import yaml_util
from sc2bench.models.detection.registry import load_detection_model
from sc2bench.models.detection.wrapper import get_wrapped_detection_model
import os
import torch
from io import BytesIO 
import os
from flask import Flask, request, jsonify
import numpy as np
import json

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
    
    file = request.files['file']

    json_data = json.loads(request.files['data'].read().decode('utf-8'))
    scale = float(json_data['1'])
    zero_point = float(json_data['2'])

    buffer_0 = BytesIO()
    buffer_0.write(file.read())
    buffer_0.seek(0)
    numpy_array_0 = np.load(buffer_0)
    torch_tensor_0 = torch.from_numpy(numpy_array_0).to(torch.float32)

    numpy_array_1 = np.array(scale)
    torch_tensor_1 = torch.from_numpy(numpy_array_1).to(torch.float32)

    numpy_array_2 = np.array(zero_point)
    torch_tensor_2 = torch.from_numpy(numpy_array_2).to(torch.float32)

    torch_tensor_3 = torch.tensor([480., 640.]).to(torch.float32)
    torch_tensor_4 = torch.tensor([480., 640.]).to(torch.float32)
    torch_tensor_5 = torch.tensor([480., 640.]).to(torch.float32)

    out_head = (torch_tensor_0, torch_tensor_1, torch_tensor_2, torch_tensor_3, torch_tensor_4, torch_tensor_5)

    prediction = edge_model(*out_head)

    my_prediction = {}
    my_prediction['detection'] = {}
    my_prediction['detection']["boxes"] = prediction[0]["boxes"].cpu().detach().numpy().tolist()
    my_prediction['detection']["labels"] = prediction[0]["labels"].cpu().detach().numpy().tolist()
    my_prediction['detection']["scores"] = prediction[0]["scores"].cpu().detach().numpy().tolist()

    return jsonify(my_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)

    


  