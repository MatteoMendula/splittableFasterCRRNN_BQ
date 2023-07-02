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
