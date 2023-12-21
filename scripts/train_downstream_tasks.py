import torch
import torch.nn.functional as F
import os
import pickle
from tqdm import tqdm
# from peft import get_peft_model, LoraConfig
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from data.get_data import get_data
from models.get_model import get_model
from utils.arguments import parse_args


def construct_training_dataset(dataset, tokenizer):
    training_dataset = []
    for datum in tqdm(dataset):
        for _t in datum["target"]:
            text = datum["prompt"] + _t + "</Answer>"
            target_text = "<Answer>" + _t + "</Answer>"
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


class TargetTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        target_position = inputs.pop("target_position")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = model(
            input_ids, attention_mask, False, False
        )
        logits = outputs["logits"]
        losses = []
        for datum_i, _target_pos in enumerate(target_position):
            nll = F.cross_entropy(
                logits[datum_i, _target_pos-1:-1],
                input_ids[datum_i, _target_pos:],
                reduction="none"
            )
            losses.append(nll.mean())
        print(input_ids.shape[1], losses[0])
        loss = torch.stack(losses).mean()
        return (loss, outputs) if return_outputs else loss

    def training_step(self, *args, **kwargs):
        output = super().training_step(*args, **kwargs)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any()
                                       or torch.isinf(param.grad).any())
                if not valid_gradients:
                    print("invalid gradient for ", name)
                    param.grad = None
        return output


def main(args):
    # argument setting and logging
    # torch.autograd.set_detect_anomaly(True)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # load model
    if local_rank == 0:
        print(args)
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
    model.model.to(torch.float16)
    tokenizer = model.tokenizer

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
            dataset["data"], model.tokenize)
        if local_rank == 0:
            print("saving dataset to cache...")
            with open(args.dataset_cache, "wb") as f:
                pickle.dump(dataset, f)
    print("dataset size: ", len(dataset))

    training_args = TrainingArguments(
        run_name=args.log_dir.split("/")[-1],
        optim="adamw_torch",
        output_dir=args.log_dir,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=(args.batch_size - 1) // 4 + 1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=0,
        # max_grad_norm=0.1,
        lr_scheduler_type="constant",
        per_device_train_batch_size=1,
        logging_steps=1,
        fsdp="full_shard",
        fsdp_config={
            "activation_checkpointing": True,
            "limit_all_gathers": True,
            "use_orig_params": "true",
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer"
        },
        gradient_checkpointing=False,
        # bf16=True,
        fp16=True,
        remove_unused_columns=False,
    )

    trainer = TargetTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
