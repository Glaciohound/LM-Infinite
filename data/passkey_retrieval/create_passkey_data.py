# This code aims to create synthetic data to evaluate capabilities of LLMs that have longer context
import json
from tqdm import tqdm
import nltk
import re
import argparse
import os
import numpy as np
import sys
from pathlib import Path
import random

from transformers import AutoTokenizer
sys.path.append(".")
from transformers import LlamaTokenizer


TASK_PREFIX = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information there."

DEFAULT_CONTENT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

KEY_CONTENT = "The pass key is {KEY}. Remember it. {KEY} is the pass key."

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def clean_string(s):
    cleaned = re.sub(r"(\(.*?\)|\[.*?\]|<.*?>|\{.*?\})", "", s)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()  # remove leading/trailing spaces


class CreatePassKeyTask:
    def __init__(self, tokenizer, data=None):
        """
        data: List containing inputs
        """
        self.data = data # if not None, will use random sentences instead of "DEFAULT_CONTENT" to fill the prompt
        self.tokenizer = tokenizer

        if self.data is None:
            self.distractor = DEFAULT_CONTENT

            self.descriptor_len = len(self.tokenizer(TASK_PREFIX))
            self.distractor_len = len(self.tokenizer.encode(self.distractor))
            self.key_info_len = len(self.tokenizer.encode(KEY_CONTENT))

    def create_task_retrieve(self, max_token_length, num_examples=100,
                             first_half=False):

        assert max_token_length > (self.key_info_len + self.descriptor_len)

        num_distractors = (max_token_length - self.key_info_len - self.descriptor_len) // self.distractor_len

        assert num_distractors >= 2

        rng = np.random.RandomState(seed=max_token_length)

        samples = []
        for _ in range(num_examples):
            random_answer = rng.randint(1,10000000)
            answer_sentence = KEY_CONTENT.format(KEY=random_answer)

            if first_half:
                insert_location = rng.randint(0, num_distractors // 2) # //2 to be far from the generation
            else:
                insert_location = rng.randint(0, num_distractors)
            input_ = [TASK_PREFIX] + [self.distractor] * insert_location + [answer_sentence] + [self.distractor] * (num_distractors - insert_location)

            samples.append({
                "input": " ".join(input_),
                "target": str(random_answer)

            })

        return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-length", type=int, default=256)
    parser.add_argument("--dump-file-path", type=str)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--first-half", action="store_true")
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    task_creator = CreatePassKeyTask(tokenizer)

    samples = task_creator.create_task_retrieve(
        args.token_length, first_half=args.first_half)

    print(f"Length of the dataset: {len(samples)}")

    random_length = len(tokenizer.encode(
        random.choice(samples)['input']))
    print(f"Length of random sample: {random_length}")

    output_dir = Path(args.dump_file_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'test.jsonl', "w") as f:
        for entry in samples:
            jout = json.dumps(entry) + "\n"
            f.write(jout)
