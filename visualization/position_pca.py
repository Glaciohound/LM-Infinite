# plot a PCA scatter of embeddings

import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.decomposition import PCA
import numpy as np

input_file = sys.argv[1]

with open(input_file, "rb") as f:
    data = pickle.load(f)

data = np.stack(data)

data = data[:16, :4096].transpose(1, 0, 2)
data = data.reshape(-1, *data.shape[2:])
pca = PCA(n_components=2)
pca.fit(data)
transformed = pca.transform(data)

n = data.shape[0]

colors = [
    (i/n, 0, 1-i/n)
    for i in range(n)
]

plt.scatter(transformed[:, 0], transformed[:, 1], s=0.1, c=colors)
plt.show()
