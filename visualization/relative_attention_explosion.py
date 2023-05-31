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

for i in range(0, data.shape[0]):
    step = 128
    smoothed_weights = np.nanmean(data[i].reshape(-1, step), -1)
    x = np.arange(0, smoothed_weights.shape[0]) * step
    plt.plot(x, smoothed_weights[::-1], label=str(i))

# plt.legend()
plt.show()
plt.close()
