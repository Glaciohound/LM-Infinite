import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
from sklearn.linear_model import LinearRegression

filename = sys.argv[1]
color = sys.argv[2]

with open(f"_logs/{filename}.pkl", "rb") as f:
    data = pickle.load(f)["nll_stats_token"]

x = np.arange(64000, dtype=float) + 1
y = np.array([
    data[_i]["mean"] for _i in x
])

start = 3
num_fit = 50
model = LinearRegression()
model.fit(
    np.log(x[start: start+num_fit]).reshape(-1, 1),
    np.log((y[start: start+num_fit]-y[-10000:].mean()).clip(min=0.0001))
)
print(model.coef_, np.exp(model.intercept_), y[-10000:].mean())

yp = np.exp(model.predict(np.log(x).reshape(-1, 1))) + \
    y[-1000:].mean()
s = (10 * np.exp(-x / 500)).clip(min=0.001)
plt.scatter(x, y, s=s, color=color)
plt.plot(x, yp, color="gray")

plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.98)
plt.xscale("log")
plt.show()
