from datasets import load_dataset
from .f1_metric import compute_f1
from .qasper import dialogue_prompts, plain_prompts


def get_narrative_qa(dataset_name, split, max_data_num, start_data_from):
    dataset = load_dataset("narrativeqa", split=split)
    if dataset_name == "narrative_qa":
        prompt_set = dialogue_prompts
    elif dataset_name == "narrative_qa_plain":
        prompt_set = plain_prompts
    else:
        raise Exception("Invalid dataset name.")

    system_prompt = prompt_set["system_prompt"]
    header = prompt_set["header"]
    concise_prompt = prompt_set["concise_prompt"]

    detailed_dataset = []
    for _datum in dataset:
        full_text = _datum["document"]["text"]

        question = _datum["question"]["text"]
        prompt = f"{system_prompt} {header} {full_text} "\
            f"Question: {question} {concise_prompt}"
        target = [answer["text"] for answer in _datum["answers"]]
        detailed_dataset.append({
            "prompt": prompt,
            "target": target,
            "id": _datum["document"]["id"],
            "title": _datum["document"]["summary"]["title"],
        })

    dataset = detailed_dataset
    if start_data_from is not None:
        dataset = dataset[start_data_from:]
    if max_data_num is not None:
        dataset = dataset[:max_data_num]

    return {
        "data": dataset,
        "metadata": {
            "metric_func": compute_f1,
            "recommended_length": 30,
        }
    }
