# plot a PCA scatter of embeddings

import matplotlib.pyplot as plt
import sys
import pickle

input_file = sys.argv[1]
layer_i = int(sys.argv[2])

with open(input_file, "rb") as f:
    data = pickle.load(f)[layer_i]

# n = data.shape[1]
n = 1000
print(data.shape)

colors = [
    (i/n, 0, 1-i/n)
    for i in range(n)
]

for sequence in data:
    plt.scatter(sequence[:n, 0], sequence[:n, 1], s=100, c=colors, alpha=0.01)
plt.show()
