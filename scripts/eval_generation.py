import torch
import os
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from data.get_data import get_data
from data.generation_metrics import generation_overall_metric
from models.get_model import get_model
from utils.arguments import parse_args


def main(args):
    # argument setting and logging
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and args.model != "constant_model":
        deepspeed.init_distributed()
    print("local_rank: ", local_rank, "world_size: ", world_size)
    print(args)

    # load data and model
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
        args.limit_distance, args.triangle_offset, args.constant_answer)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    if args.model != "constant_model":
        deepspeed_config = HfDeepSpeedConfig(args.deepspeed_config)
        print(deepspeed_config)
        ds_engine = deepspeed.initialize(
            model=model, config=args.deepspeed_config
        )[0]
        model = ds_engine.module
        model.eval()
    print(model)

    # prediction
    print("starts evaluating...")
    all_results = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            input_ids, attention_mask = model.tokenize(batch)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            generation_results = {}
            for position in args.evaluate_positions:
                if input_ids.shape[1] <= position:
                    continue
                generation = model.generate(
                    input_ids[:, :position],
                    attention_mask[:, :position],
                    args.max_generation_length,
                    args.suppress_tokens, args.do_sample
                )[0][0]
                generation_results[position] = {
                    "generation": generation,
                    "target": model.tokenizer.decode(
                        input_ids[0, position:
                                  position+args.max_generation_length]
                    )
                }
                print(generation_results[position])

            print(generation_results)
            all_results.append({
                "generation_results": generation_results,
            })
            if args.evaluate_metrics:
                scores = generation_overall_metric(
                    all_results, args.evaluate_positions
                )
                print(scores)

            with open(os.path.join(args.log_dir, "results.pkl"), "wb") as f:
                pickle.dump([all_results, scores], f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
