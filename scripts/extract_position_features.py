import torch
import pickle
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from sklearn.decomposition import PCA

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
        args.limit_distance, args.triangle_offset, args.constant_answer)
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
    all_features = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        10: [],
        20: [],
    }
    pbar = tqdm(dataloader)
    for input_ids, attention_mask in pbar:
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pbar.set_description(f"input length: {input_ids.shape[1]}")
            output = model.forward_features(
                input_ids, attention_mask, False, True,
            )
        for layer_i, features in all_features.items():
            features.append(output.hidden_states[layer_i][0].cpu().numpy())

    for layer_i, features in all_features.items():
        all_features[layer_i] = np.stack(features, axis=0)

    all_transformed = {}

    for layer_i, features in tqdm(all_features.items()):
        n, L, d = features.shape
        data = features.reshape(-1, d)
        pca = PCA(n_components=2)
        pca.fit(data)
        transformed = pca.transform(data)
        transformed = transformed.reshape(n, L, 2)
        all_transformed[layer_i] = transformed

    with open(os.path.join(args.log_dir, "features.pkl"), "wb") as f:
        pickle.dump(all_transformed, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
