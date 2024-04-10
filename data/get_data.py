import json
from tqdm import tqdm

system_prompt = "<s>[INST] <<SYS>> \n " \
    "You are a helpful, concise and honest assistant. <</SYS>> \n "
concise_prompt = "Do not provide any explanation. [/INST] \n"


def get_pile_dataset(dataset_dir, dataset_group, split,
                     max_data_num, start_data_from):
    file_name = f"{dataset_dir}/{split}/{dataset_group}.jsonl"
    with open(file_name, "r") as f:
        dataset = list(map(json.loads, tqdm(f.readlines())))

    if dataset_group == "ArXiv":
        dataset = list(filter(lambda x: ":" in x, dataset))

    dataset = sorted(dataset, key=lambda x: len(x))
    if start_data_from is not None:
        dataset = dataset[start_data_from:]
    if max_data_num is not None:
        dataset = dataset[:max_data_num]
    lengths = list(map(lambda x: len(x), dataset))
    print("Max length:", max(lengths))
    print("Min length:", min(lengths))
    print("Avg length:", sum(lengths) / len(lengths))
    return dataset


def get_passkey_retrieval(dataset_dir, dataset_group, structured_prompt,
                          max_data_num, start_data_from):
    from .passkey_retrieval.passkey_retrieval_accuracy import \
        passkey_retrieval_accuracy
    filename = f"{dataset_dir}/{dataset_group}/test.jsonl"
    with open(filename, "r") as f:
        dataset = list(map(json.loads, tqdm(f.readlines())))
    if start_data_from is not None:
        dataset = dataset[start_data_from:]
    if max_data_num is not None:
        dataset = dataset[:max_data_num]
    for _id, _datum in enumerate(dataset):
        _datum["prompt"] = _datum["input"] + \
            " What is the pass key? The pass key is "
        if structured_prompt:
            _datum["prompt"] = system_prompt + _datum["input"] + concise_prompt
        _datum["id"] = _id
    return {
        "data": dataset,
        "metadata": {
            "metric_func": passkey_retrieval_accuracy,
            "recommended_length": None,
        }
    }


def get_needle_in_a_haystack(
        dataset_dir, dataset_group, structured_prompt,
        max_data_num, start_data_from):
    from .needle_in_a_haystack.evaluators.openai import OpenAIEvaluator
    filename = f"{dataset_dir}/{dataset_group}-test.jsonl"
    with open(filename, "r") as f:
        dataset = list(map(json.loads, tqdm(f.readlines())))
    if start_data_from is not None:
        dataset = dataset[start_data_from:]
    if max_data_num is not None:
        dataset = dataset[:max_data_num]
    for _id, _datum in enumerate(dataset):
        _datum["prompt"] = _datum["input"] + _datum["question"]
        if structured_prompt:
            _datum["prompt"] = system_prompt + _datum["input"] + concise_prompt
        _datum["id"] = _id
    agent = OpenAIEvaluator(
        true_answer=dataset[0]["target"],
        question_asked=dataset[0]["question"]
    )
    return {
        "data": dataset,
        "metadata": {
            "metric_func": agent.evaluate_batch_responses,
            "recommended_length": None,
        }
    }


def get_data(dataset_name, dataset_dir, dataset_group,
             split, structured_prompt,
             max_data_num, start_data_from):
    if dataset_name == "the_pile":
        dataset = get_pile_dataset(
            dataset_dir, dataset_group, split,
            max_data_num, start_data_from)
    elif dataset_name == "passkey_retrieval":
        dataset = get_passkey_retrieval(
            dataset_dir, dataset_group, structured_prompt,
            max_data_num, start_data_from)
    elif dataset_name == "needle_in_a_haystack":
        dataset = get_needle_in_a_haystack(
            dataset_dir, dataset_group, structured_prompt,
            max_data_num, start_data_from)
    elif dataset_name == "tau/zero_scrolls":
        from .zero_scrolls.get_zero_scrolls import get_zero_scrolls
        dataset = get_zero_scrolls(
            dataset_group, split,
            max_data_num, start_data_from)
    elif dataset_name.startswith("qasper"):
        from .qasper import get_qasper
        dataset = get_qasper(
            dataset_name, split, max_data_num, start_data_from)
    elif dataset_name.startswith("narrative_qa"):
        from .narrative_qa import get_narrative_qa
        dataset = get_narrative_qa(
            dataset_name, split, max_data_num, start_data_from)
    else:
        raise NotImplementedError()

    return dataset
