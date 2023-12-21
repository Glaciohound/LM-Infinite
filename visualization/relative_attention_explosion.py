import matplotlib.pyplot as plt
import numpy as np
import sys
import json

input_file = sys.argv[1]

with open(input_file, "r") as f:
    data = []
    for line in f:
        line = line.replace("nan", "NaN")
        line_data = json.loads(line.strip())
        data.append(line_data)

data = np.array(data)
data = np.flip(data, 1)
n = 8192

all_bound = []
for i in range(0, data.shape[0]):
    x = data[i]
    bound = [x[0]]
    for y in x[1:n]:
        bound.append(max(bound[-1], y))
    # plt.plot(bound)
    all_bound.append(np.array(bound))

mean = np.mean(all_bound, axis=0)
std = np.std(all_bound, axis=0, ddof=1)
plt.plot(mean, color="blue")
plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, color="blue")

# plt.legend()
plt.show()
plt.close()
