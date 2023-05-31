import sys
from tqdm import tqdm
import json

input_file = sys.argv[1]
output_dir = sys.argv[2]

print("Input file:", input_file)
print("Output dir:", output_dir)

with open(input_file, "r") as f:
    data = list(map(json.loads, tqdm(f.readlines())))

print("Data length:", len(data))

print("Starts data splitting")
splitted_data = {}
for _datum in tqdm(data):
    text = _datum["text"]
    set_name = _datum["meta"]["pile_set_name"]
    if set_name not in splitted_data:
        splitted_data[set_name] = []
    splitted_data[set_name].append(text)

print("Starts writing splitted data")
for set_name, set_data in tqdm(splitted_data.items()):
    with open(f"{output_dir}/{set_name}.jsonl", "w") as f:
        for _datum in set_data:
            f.write(json.dumps(_datum) + "\n")
