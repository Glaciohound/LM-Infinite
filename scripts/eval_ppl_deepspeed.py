import torch
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle
from torch.utils.data import DataLoader
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from data.get_data import get_data
from models.get_model import get_model
from utils.arguments import parse_args


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
    print("# total entries: ", len(data))

    print("loading model...")
    model = get_model(
        args.model, args.tokenizer_path, args.max_length, args.truncation_side,
        args.fp16, args.load_in_4bit, args.device_map,
        args.use_lambda_attention,
        args.local_branch, args.global_branch,
        args.limit_distance, args.triangle_offset, args.constant_answer,
        args.top_k_attention, args.top_k_insert_at,
        args.top_k_from_layer, args.top_k_to_layer)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                            collate_fn=model.tokenize)

    deepspeed_config = HfDeepSpeedConfig(args.deepspeed_config)
    print(deepspeed_config)
    ds_engine = deepspeed.initialize(
        model=model, config=args.deepspeed_config
    )[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(model)

    print("starts evaluating...")
    all_nll = []
    pbar = tqdm(dataloader)
    for input_ids, attention_mask in pbar:
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pbar.set_description(f"input length: {input_ids.shape[1]}")
            output = model.forward_features(
                input_ids, attention_mask, False, False,
            )
        all_nll.extend([_nll_list.cpu().numpy() for _nll_list in
                        output["token_nll_list"]])
        print("Shape:", output["logits"].shape)
        print("Start: ", output["token_nll_list"][0][:20])
        print("End:", output["token_nll_list"][0][-20:])

        nll_stats_sequence = defaultdict(list)
        nll_stats_token = defaultdict(list)
        for nll_seq in all_nll:
            nll_stats_sequence[len(nll_seq)].append(nll_seq.mean())
            for token_i, token_nll in enumerate(nll_seq):
                nll_stats_token[token_i].append(token_nll)

        nll_stats_sequence = {
            length: {"mean": np.nanmean(np.array(record)),
                     "var": np.nanvar(np.array(record))}
            for length, record in nll_stats_sequence.items()
        }
        nll_stats_token = {
            length: {"mean": np.nanmean(np.array(record)),
                     "var": np.nanvar(np.array(record))}
            for length, record in nll_stats_token.items()
        }

        with open(f"{args.log_dir}/full_stats.pkl", "wb") as f:
            pickle.dump({
                "nll_stats_sequence": nll_stats_sequence,
                "nll_stats_token": nll_stats_token,
                "all_nll": all_nll
            }, f)

        with open(f"{args.log_dir}/stats.pkl", "wb") as f:
            pickle.dump({
                "nll_stats_sequence": nll_stats_sequence,
                "nll_stats_token": nll_stats_token,
            }, f)

    nll_mean = np.array([np.nanmean(_nll) for _nll in all_nll])
    print("nll_mean: ", nll_mean.mean())


if __name__ == "__main__":
    args = parse_args()
    main(args)
