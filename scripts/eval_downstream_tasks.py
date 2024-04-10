import torch
import os
from tqdm import tqdm
import pickle
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import StoppingCriteria

from data.get_data import get_data
from models.get_model import get_model
from utils.arguments import parse_args


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


def save_results(split, all_target, all_output, all_ids, all_datum_id, scores):
    with open(os.path.join(args.log_dir,
                           f"results_{split}.pkl"), "wb") as f:
        pickle.dump({
            "all_target": all_target,
            "all_output": all_output,
            "all_ids": all_ids,
            "all_datum_ids": all_datum_id,
            "scores": scores,
        }, f)


def load_results(split):
    cache_file = os.path.join(args.log_dir, f"results_{split}.pkl")
    if os.path.exists(cache_file):
        print("Loading results from cache:", cache_file)
        with open(cache_file, "rb") as f:
            results = pickle.load(f)
        return results["all_target"], results["all_output"], \
            results["all_ids"], results["all_datum_ids"], results["scores"]
    else:
        print("No cache file found.")
        return [], [], [], [], None


class StoppingWords(StoppingCriteria):
    stopping_words = ["</Summary>", "</Answer>", "\n"]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor, **kwargs) -> bool:
        return any([
            self.tokenizer.decode(one_input).endswith(word)
            for word in self.stopping_words
            for one_input in input_ids
        ])


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

    # load data
    data = get_data(args.dataset, args.dataset_dir,
                    args.dataset_group, args.split, args.structured_prompt,
                    args.max_data_num, args.start_data_from)
    # load existing evaluation results
    all_target, all_output, all_ids, all_datum_ids, scores = load_results(
        args.split
    )
    max_generation_length = args.max_generation_length if \
        args.max_generation_length is not None else \
        data["metadata"]["recommended_length"]
    metric_func = data["metadata"]["metric_func"]
    print("dataset size: ", len(data["data"]))
    print("generation length: ", max_generation_length)

    # load model
    print("loading model...")
    model = get_model(
        args.model, args.tokenizer_path, args.max_length, args.truncation_side,
        args.fp16, args.load_in_4bit, args.device_map,
        args.use_lambda_attention,
        args.local_branch, args.global_branch,
        args.limit_distance, args.triangle_offset, args.constant_answer,
        args.top_k_attention, args.top_k_insert_at,
        args.top_k_from_layer, args.top_k_to_layer,
    )
    if args.model != "constant_model":
        HfDeepSpeedConfig(args.deepspeed_config)
        ds_engine = deepspeed.initialize(
            model=model, config=args.deepspeed_config
        )[0]
        model = ds_engine.module
        model.eval()
    print(model)

    # evaluate
    print("starts evaluating...")
    pbar = tqdm(data["data"])
    start_data_from = args.start_data_from or 0
    for datum_i, datum in enumerate(pbar):
        if datum_i in all_datum_ids:
            continue
        target = datum["target"]
        input_ids, attention_mask = model.tokenize(
            datum["prompt"])
        input_length = input_ids.shape[1]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            output, _ = model.generate(
                input_ids, attention_mask,
                max_generation_length,
                args.min_new_tokens,
                args.suppress_tokens, args.do_sample,
                [StoppingWords(model.tokenizer)]
            )

        if not args.silent:
            print("Length:", input_length, "Prediction:", output[0])
            print("References:", target)
        all_output.append(remove_tag(output[0].strip()))
        all_target.append(target)
        all_ids.append(datum["id"])
        all_datum_ids.append(datum_i + start_data_from)

        if not args.silent and args.dataset == "passkey_retrieval":
            answer_prefix = datum["prompt"][
                :datum["prompt"].index(target) + len(target)
            ]
            distance_till_end = input_length - \
                model.tokenize(answer_prefix)[0].shape[1]
            print("Distance till end:", distance_till_end)

        scores = metric_func(all_output, all_target, datum) if \
            args.evaluate_metrics else None
        score_text = f"{args.dataset} {args.dataset_group} "\
            f"{args.split}: {scores}"
        save_results(args.split, all_target, all_output, all_ids,
                     all_datum_ids, scores)
        pbar.set_description(score_text)

    print(score_text)


if __name__ == "__main__":
    args = parse_args()
    main(args)
