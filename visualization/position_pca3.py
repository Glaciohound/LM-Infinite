# plot a PCA scatter of embeddings

import matplotlib.pyplot as plt
import sys
import pickle

input_file = sys.argv[1]
layer_i = int(sys.argv[2])
data_from = int(sys.argv[3])
data_to = int(sys.argv[4])
s = float(sys.argv[5])
alpha = float(sys.argv[6])

with open(input_file, "rb") as f:
    data = pickle.load(f)[layer_i]

n = 1000
data_from = min(data_from, n)
print("# of sequences", data.shape[0])
print("# of tokens per sequence", data.shape[1])
print("# of dimensions per token", data.shape[2])
print("visualizing token ranges from {} to {}".format(data_from, data_to))

colors = [
    (i/n, 0, 1-i/n)
    for i in range(data_from, data_to)
]


for sequence in data:
    plt.scatter(sequence[data_from: data_to, 0],
                sequence[data_from: data_to, 1],
                s=s, c=colors, alpha=alpha,
                edgecolors='none')
plt.show()
