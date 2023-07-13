import numpy as np
import requests
import time
from io import BytesIO
import argparse


parser = argparse.ArgumentParser(description='Parser jpge communication check')
parser.add_argument('-f', '--forever', default=False, type=bool)
parser.add_argument('-s', '--sleep_time', default=0.5, type=float)
args = vars(parser.parse_args())

np_array = np.load('reshaped.npy')
np_array = np_array.astype("int8")
array_bytes = np_array.tobytes()

# Create an in-memory buffer using BytesIO
buffer = BytesIO(array_bytes)
files = {'np_array': buffer}

latency = []
fails = 0
N_TESTS = 1000

def send_tensor(np_array = np.array([[1, 2, 3], [4, 5, 6]]), url = "http://10.42.0.121:5000/furcifer_check_communication"):
    start_time = time.time()
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
        with open("out_split.txt", "a") as text_file:
            text_file.write(" ----------------- \n")
            text_file.write("mean: " + str(mean) + "\n")
            text_file.write("variance: " + str(variance) + "\n")
            text_file.write("standard deviation: " + str(res) + "\n")
    print("Fails: " + str(fails))
