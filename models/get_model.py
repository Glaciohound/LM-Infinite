from .gpt_j import GPT_J_Model
from .llama import LLAMA_Model
from .mpt_7b import MPT_7B_Model
from .constant import ConstantModel


def get_model(
    model_name_or_path, tokenizer_path, max_length, truncation_side,
    fp16, load_in_4bit, device_map,
    use_lambda_attention,
    local_branch, global_branch,
    limit_distance, triangle_offset, constant_answer,
    top_k_attention
):
    hack_args = (use_lambda_attention, local_branch, global_branch,
                 limit_distance, triangle_offset)
    if model_name_or_path == "EleutherAI/gpt-j-6b":
        model = GPT_J_Model(
            model_name_or_path, max_length, fp16, truncation_side,
            *hack_args
        )
    elif model_name_or_path == "decapoda-research/llama-7b-hf":
        model = LLAMA_Model(
            model_name_or_path, model_name_or_path,
            max_length, truncation_side,
            load_in_4bit, device_map, *hack_args
        )
    elif "llama-2" in model_name_or_path:
        model = LLAMA_Model(
            model_name_or_path, tokenizer_path, max_length, truncation_side,
            load_in_4bit, device_map, *hack_args,
            top_k_attention
        )
    elif model_name_or_path.startswith("mosaicml/mpt-7b"):
        model = MPT_7B_Model(
            model_name_or_path, max_length, truncation_side, *hack_args)
    elif model_name_or_path == "constant_model":
        return ConstantModel(max_length, truncation_side, constant_answer)
    else:
        raise NotImplementedError()
    return model
