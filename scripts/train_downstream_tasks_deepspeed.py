import pickle
import torch
import os
from itertools import chain
from tqdm import tqdm
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from datasets import Dataset
import torch.multiprocessing
# from peft import get_peft_model, LoraConfig, BitsAndBytesConfig

from data.get_data import get_data
from models.get_model import get_model
from utils.arguments import parse_args


def construct_training_dataset(dataset, tokenizer, no_end_token):
    training_dataset = []
    for datum in tqdm(dataset):
        for _t in datum["target"]:
            if not no_end_token:
                text = datum["prompt"] + _t + "</Answer>"
                target_text = "<Answer>" + _t + "</Answer>"
            else:
                text = datum["prompt"] + _t
                target_text = _t
            input_ids, attention_mask = tokenizer(text)
            target_length = tokenizer(target_text)[0].shape[1] - 1
            training_dataset.append({
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "target_position": input_ids.shape[1] - target_length,
                "labels": 0,
            })
    training_dataset = Dataset.from_list(training_dataset)
    return training_dataset


def main(args):
    # argument setting and logging
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        deepspeed.init_distributed()
    print("local_rank: ", local_rank, "world_size: ", world_size)
    print(args)

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
    # HfDeepSpeedConfig(args.deepspeed_config)
    deepspeed_config = HfDeepSpeedConfig(args.deepspeed_config)
    deepspeed_config.config["train_batch_size"] = args.batch_size
    deepspeed_config.config["gradient_accumulation_steps"] = \
        args.batch_size // world_size // \
        deepspeed_config.config["train_micro_batch_size_per_gpu"]

    # training dataset
    if os.path.exists(args.dataset_cache):
        print("loading dataset from cache...")
        with open(args.dataset_cache, "rb") as f:
            dataset = pickle.load(f)
        # dataset = Dataset.from_list([dataset[0]] * 1000)
    else:
        dataset = get_data(
            args.dataset, args.dataset_dir,
            args.dataset_group, args.split, args.structured_prompt,
            args.max_data_num,
            args.start_data_from)
        dataset = construct_training_dataset(
            dataset["data"], model.tokenize, args.no_end_token)
        if local_rank == 0:
            print("saving dataset to cache...")
            with open(args.dataset_cache, "wb") as f:
                pickle.dump(dataset, f)
    print("dataset size: ", len(dataset))

    # deepspeed initialization
    model_engine, optimizer, dataloader, _ = deepspeed.initialize(
        model=model.model, config=deepspeed_config.config,
        training_data=dataset
    )
    _, client_sd = model_engine.load_checkpoint(args.log_dir)
    last_step = client_sd['step'] if client_sd is not None else 0
    model.model = model_engine.module
    model.train()
    print(model)

    # training
    print("start training...")
    # pbar = tqdm(dataloader)
    pbar = tqdm(chain(*[iter(dataloader)] * args.num_train_epochs),
                total=len(dataloader) * args.num_train_epochs)
    for step_i, datum in enumerate(pbar):
        if step_i < last_step:
            continue
        target_position = datum.pop("target_position")
        input_ids = datum["input_ids"]
        input_ids = torch.tensor(
            datum["input_ids"], device=device, dtype=int).unsqueeze(0)
        attention_mask = torch.tensor(
            datum["attention_mask"], device=device, dtype=bool).unsqueeze(0)

        with torch.no_grad():
            prompt_output = model(
                input_ids[:, :target_position-1],
                attention_mask[:, :target_position-1],
                use_cache=True
            )

        output = model(
            input_ids[:, target_position-1:],
            attention_mask,
            past_key_values=prompt_output["past_key_values"],
        )

        losses = [
            _nll[1:].mean()
            for _nll in output["token_nll_list"]
        ]
        loss = sum(losses) / len(losses)
        pbar.set_description(
            f"loss:{loss.item():.4f}, length:{input_ids.shape[1]}")
        print()

        model_engine.backward(loss)
        model_engine.step()

        if (step_i+1) % args.save_steps == 0:
            ckpt_id = step_i
            model_engine.save_checkpoint(
                args.log_dir, ckpt_id, save_latest=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
