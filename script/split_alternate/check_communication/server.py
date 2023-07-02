from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,FasterRCNN_MobileNet_V3_Large_FPN_Weights,fasterrcnn_mobilenet_v3_large_fpn

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
preprocess = weights.transforms()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return "Hello world!"

@app.route('/furcifer_check_communication', methods=['POST'])
def receive_data():
    if 'np_array' not in request.files:
        return 'No tensor found in the request', 400

    array_list = request.files['np_array']  # Assuming the tensor is stored under the key 'tensor'
    # Convert the list back to a Torch tensor
    tensor = np.array(array_list)
    # Process the received tensor or perform any desired operations

    # Return a response if needed
    return jsonify({'message': 'Tensor received successfully'})

@app.route('/furcifer_check_communication_jpeg', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image found in the request', 400

    image_file = request.files['image']
    image = Image.open(image_file)

    x = [preprocess(image).cuda()]
    # save the image to disk
    image.save('image.jpg')
    
    return jsonify({'message': 'Image received successfully'})

if __name__ == '__main__':
    app.run(host="0.0.0.0")