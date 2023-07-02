import numpy as np
import requests
import json
import time
import argparse
from PIL import Image
from io import BytesIO

parser = argparse.ArgumentParser(description='Parser jpge communication check')
parser.add_argument('-c', '--compression', default=100, type=int)
args = vars(parser.parse_args())

compression = args["compression"]
image = Image.open('kitchen.jpg')

latency = []
fails = 0

N_TESTS = 1000

def send_tensor(tensor = np.array([[1, 2, 3], [4, 5, 6]]), url = "http://10.42.0.121:5000/furcifer_check_communication_jpeg"):
    start_time = time.time()
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=compression)
    buffer.seek(0)
    files = {'image': buffer}
    response = requests.post(url, files=files, timeout=1)
    return [time.time() - start_time]

if __name__ == '__main__':
    for i in range(N_TESTS):
        try:
            lat = send_tensor()
            latency += lat
        except:
            fails += 1
        print("sending: ", i)
    mean = sum(latency) / len(latency)
    variance = sum([((x - mean) ** 2) for x in latency]) / len(latency)
    res = variance ** 0.5
    print("Mean: " + str(mean))
    print("Variance: " + str(variance))
    print("Standard deviation: " + str(res))
    print("Fails: " + str(fails))
