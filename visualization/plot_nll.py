import matplotlib.pyplot as plt
import pickle
import sys
smooth_gamma = float(sys.argv[1])
files = sys.argv[2::2]
labels = sys.argv[3::2]

for _filename, _label in zip(files, labels):
    with open(f"_logs/{_filename}", "rb") as f:
        data = pickle.load(f)["nll_stats_token"]
        smoothed = [0]
        for _d in data.values():
            smoothed.append(
                _d["mean"] * (1 - smooth_gamma) +
                smoothed[-1] * smooth_gamma
            )
        smoothed.pop(0)
        plt.plot(smoothed, label=_label, linewidth=1.5)

plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.98)
plt.legend()
plt.show()
