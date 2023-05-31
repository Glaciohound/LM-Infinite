import sys
import time
import pickle
from data.get_data import get_zero_scrolls
from data.zero_scrolls_metrics import zero_scrolls_metrics_dict

dir_prefix = sys.argv[1]
num_trials = int(sys.argv[2])
task_name = sys.argv[3]
split = sys.argv[4]
file_list = [
    f"{dir_prefix}{i}/results.pkl" for i in range(0, num_trials)
]

dataset = get_zero_scrolls(task_name, split, None, None)
metric = zero_scrolls_metrics_dict[task_name]

all_target = []
all_output = []
all_ids = []
score_sum = {}
for file_name in file_list:
    with open(file_name, "rb") as f:
        results = pickle.load(f)
    all_ids.extend(results["all_ids"])
    all_target.extend(results["all_target"])
    all_output.extend(results["all_output"])

time.sleep(1)
gathered_ids = []
predictions = []
references = []
for _id, _target, _output in zip(all_ids, all_target, all_output):
    if _id not in gathered_ids:
        gathered_ids.append(_id)
        predictions.append(_output)
        references.append([_target])
    else:
        references[-1].append(_target)

if split == "validation":
    scores = metric(predictions=predictions, references=references)
else:
    scores = None

assert all_ids == [_d["id"] for _d in dataset]
print("checksum passed", dir_prefix, task_name, split, scores)

combined_results = {
    "all_id": gathered_ids,
    "all_output": predictions,
    "all_target": references,
    "scores": scores,
}

with open(f"{dir_prefix}0/combined_results.pkl", "wb") as f:
    pickle.dump(combined_results, f)
