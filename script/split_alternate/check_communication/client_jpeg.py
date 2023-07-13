import numpy as np
import requests
import json
import time
import argparse
from PIL import Image
from io import BytesIO

parser = argparse.ArgumentParser(description='Parser jpge communication check')
parser.add_argument('-c', '--compression', default=100, type=int)
parser.add_argument('-s', '--sleep_time', default=0.5, type=float)
parser.add_argument('-f', '--forever', default=False, type=bool)
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
    return_value = {}
    return_value["error"] = True
    try:
        response = requests.post(url, files=files, timeout=1)
        response.raise_for_status()
        return_value["latency"] = [time.time() - start_time]
        return_value["error"] = False
    except Exception as errh:
        print("HTTP Error: ", errh)
    return return_value

if __name__ == '__main__':

    # this is meant just to create noise
    if args["forever"]:
        while True:
            print("sending noise")
            lat = send_tensor()
            time.sleep(args["sleep_time"])

    for i in range(N_TESTS):
        print("sending: ", i)
        lat = send_tensor()
        if not lat["error"]:
            latency += lat["latency"]
        else:
            fails += 1
    if len(latency) > 0:
        mean = sum(latency) / len(latency)
        variance = sum([((x - mean) ** 2) for x in latency]) / len(latency)
        res = variance ** 0.5
        print("Mean: " + str(mean))
        print("Variance: " + str(variance))
        print("Standard deviation: " + str(res))
        with open("out_jpeg.txt", "a") as text_file:
            text_file.write("----------- compression {} -----------\n".format(compression))
            text_file.write("mean: " + str(mean) + "\n")
            text_file.write("variance: " + str(variance) + "\n")
            text_file.write("standard deviation: " + str(res) + "\n")
    print("Fails: " + str(fails))
    with open("Output.txt", "a") as text_file:
        text_file.write("fails: " + str(fails) + "\n")

