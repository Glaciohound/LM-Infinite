import torch
import os
from tqdm import tqdm
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from torch.profiler import profile, record_function, ProfilerActivity

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

    print("loading model...")
    model = get_model(
        args.model, args.tokenizer_path, args.max_length, args.truncation_side,
        args.fp16, args.load_in_4bit, args.device_map,
        args.use_lambda_attention,
        args.local_branch, args.global_branch,
        args.limit_distance, args.triangle_offset, args.constant_answer,
        args.top_k_attention, args.top_k_insert_at,
        args.top_k_from_layer, args.top_k_to_layer)

    deepspeed_config = HfDeepSpeedConfig(args.deepspeed_config)
    print(deepspeed_config)
    ds_engine = deepspeed.initialize(
        model=model, config=args.deepspeed_config
    )[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(model)

    print("starts evaluating...")
    with profile(activities=[ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True) as prof:
        for _ in tqdm(range(args.max_data_num)):
            input_ids = torch.randint(0, 1000, (1, args.max_length)).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
            with torch.no_grad():
                _ = model.forward_features(
                    input_ids, attention_mask, False, False,
                )

    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage", row_limit=10))


if __name__ == "__main__":
    args = parse_args()
    main(args)
