import numpy as np
import requests
import json
import time
from io import BytesIO

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
    print("Fails: " + str(fails))
