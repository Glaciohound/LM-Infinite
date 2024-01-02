import torch
import os
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from data.get_data import get_data
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
    max_generation_length = args.max_generation_length if \
        args.max_generation_length is not None else \
        data["metadata"]["recommended_length"]
    metric_func = data["metadata"]["metric_func"]
    print("dataset size: ", len(data["data"]))
    print("generation length: ", max_generation_length)

    print("loading model...")
    model = get_model(
        args.model, args.tokenizer_path, args.max_length, args.truncation_side,
        args.fp16, args.load_in_4bit, args.device_map,
        args.use_lambda_attention,
        args.local_branch, args.global_branch,
        args.limit_distance, args.triangle_offset, args.constant_answer)
    dataloader = DataLoader(data["data"],
                            batch_size=args.batch_size, shuffle=False)

    if args.model != "constant_model":
        HfDeepSpeedConfig(args.deepspeed_config)
        ds_engine = deepspeed.initialize(
            model=model, config=args.deepspeed_config
        )[0]
        model = ds_engine.module
        model.eval()
    print(model)

    # prediction
    print("starts evaluating...")
    all_target = []
    all_output = []
    all_ids = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            input_ids, attention_mask = model.tokenize(
                batch["prompt"])
            input_length = input_ids.shape[1]
            if input_length < 4096:
                continue
            target = batch["output"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output, output_ids = model.generate(
                input_ids, attention_mask, max_generation_length, args.min_new_tokens,
                args.suppress_tokens, args.do_sample)
            # print(model.tokenizer.convert_ids_to_tokens(
            #     output_ids[0, input_ids.shape[1]:]))
            print("Prediction:", output[0].strip())
            print("References:", target[0])
            all_target.extend(target)
            all_output.extend(output)
            all_ids.extend(batch["id"])
            print(args.dataset_group, "score :",
                  metric_func(all_output, all_target, batch))

            with open(os.path.join(args.log_dir, "results.pkl"), "wb") as f:
                pickle.dump({
                    "all_target": all_target,
                    "all_output": all_output,
                    "all_ids": all_ids,
                }, f)

    # evaluation
    if args.evaluate_metrics:
        scores = metric_func(all_output, all_target, batch)
    else:
        def remove_tag(_text):
            if "<Summary>" in _text:
                _text = _text[_text.index("<Summary>") + len("<Summary>"):]
            if "</Summary>" in _text:
                _text = _text[:_text.index("</Summary>")]
            if "<Answer>" in _text:
                _text = _text[_text.index("<Answer>") + len("<Answer>"):]
            if "</Answer>" in _text:
                _text = _text[:_text.index("</Answer>")]
            return _text

        all_output = [remove_tag(_text) for _text in all_output]
        scores = None

    print("group:", args.dataset_group, "scores:", scores)
    with open(os.path.join(args.log_dir, "results.pkl"), "wb") as f:
        pickle.dump({
            "all_target": all_target,
            "all_output": all_output,
            "all_ids": all_ids,
            "scores": scores,
        }, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
