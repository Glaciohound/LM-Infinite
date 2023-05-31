import os
import sys
import pickle
from data.generation_metrics import generation_overall_metric
from pprint import pprint

dir_prefix = sys.argv[1]
indices = sys.argv[2].split(",")
evaluate_positions = list(map(int, sys.argv[3].split(",")))
os.makedirs(f"{dir_prefix}combined", exist_ok=True)

all_results = []
for i in indices:
    with open(f"{dir_prefix}{i}/results.pkl", "rb") as f:
        individual_results = pickle.load(f)[0]
    print(i, len(individual_results))
    all_results.extend(individual_results)

scores = generation_overall_metric(all_results, evaluate_positions)
pprint(scores)

with open(f"{dir_prefix}combined/combined_results.pkl", "wb") as f:
    pickle.dump([all_results, scores], f)
