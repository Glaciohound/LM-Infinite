# plot a PCA scatter of embeddings

import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.decomposition import PCA

input_file = sys.argv[1]

with open(input_file, "rb") as f:
    data = pickle.load(f)

data = data[:4096]
pca = PCA(n_components=2)
pca.fit(data)
transformed = pca.transform(data)

n = data.shape[0]

colors = [
    (i/n, 0, 1-i/n)
    for i in range(n)
]

plt.scatter(transformed[:, 0], transformed[:, 1], s=2, c=colors)
plt.show()
