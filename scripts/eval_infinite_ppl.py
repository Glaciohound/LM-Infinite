import torch
import os
import random
from tqdm import tqdm
import pickle
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from data.get_data import get_data
from models.get_model import get_model
from utils.arguments import parse_args


class StreamingLoader:
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = torch.zeros(0, dtype=torch.long)
        self.attention_mask = torch.zeros(0, dtype=torch.bool)

    def __iter__(self):
        first = True
        while True:
            random.shuffle(self.data)
            for datum in self.data:
                encoded = self.tokenizer(
                    datum, return_tensors="pt", add_special_tokens=first)
                first = False
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                self.input_ids = torch.cat([self.input_ids, input_ids[0]])
                self.attention_mask = torch.cat(
                    [self.attention_mask, attention_mask[0]]
                )
                while self.input_ids.shape[0] > self.max_length:
                    yield self.input_ids[:self.max_length], \
                        self.attention_mask[:self.max_length]
                    self.input_ids = self.input_ids[self.max_length:]
                    self.attention_mask = self.attention_mask[self.max_length:]


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        deepspeed.init_distributed()
    print("local_rank: ", local_rank, "world_size: ", world_size)
    print(args)

    data = get_data(args.dataset, args.dataset_dir,
                    args.dataset_group, args.split, args.structured_prompt,
                    args.max_data_num, args.start_data_from)

    print("loading model...")
    model = get_model(
        args.model, args.tokenizer_path, args.max_length, args.truncation_side,
        args.fp16, args.load_in_4bit, args.device_map,
        args.use_lambda_attention,
        args.local_branch, args.global_branch,
        args.limit_distance, args.triangle_offset, args.constant_answer,
        args.top_k_attention, args.top_k_insert_at,
        args.top_k_from_layer, args.top_k_to_layer)
    dataloader = iter(StreamingLoader(data, model.tokenizer, args.max_length))

    deepspeed_config = HfDeepSpeedConfig(args.deepspeed_config)
    print(deepspeed_config)
    ds_engine = deepspeed.initialize(
        model=model, config=args.deepspeed_config
    )[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(model)

    print("starts evaluating...")
    all_nll = torch.zeros(0, dtype=torch.float16)
    pbar = tqdm(range(args.streaming_length // args.max_length + 1))
    past_key_values = None
    for step_i in pbar:
        input_ids, attention_mask = next(dataloader)
        with torch.no_grad():
            input_ids = input_ids.to(device)[None]
            attention_mask = attention_mask.to(device)[None]
            pbar.set_description(f"input length: {input_ids.shape[1]}")
            output = model.forward_features(
                input_ids, attention_mask, use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = output["past_key_values"]
        all_nll = torch.cat([all_nll, output["token_nll_list"][0].cpu()])
        del output

        with open(f"{args.log_dir}/stats.pkl", "wb") as f:
            pickle.dump(all_nll, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
