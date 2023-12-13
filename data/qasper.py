from datasets import load_dataset
from .f1_metric import compute_f1

dialogue_prompts = {
    "header": "You are given a scientific article and a question. Answer the " \
        "question as concisely as you can, using a single phrase or sentence if " \
        "possible. If the question cannot be answered based on the information in the " \
        "article, write \"unanswerable\". If the question is a yes/no question, answer " \
        "\"yes\", \"no\", or \"unanswerable\". Article:",
    "system_prompt": "<s>[INST] <<SYS>> \n " \
        "You are a helpful, concise and honest assistant. <</SYS>> \n ",
    "concise_prompt": "Please highlight your final answer with a single pair of " \
        "<Answer> </Answer> tags. Do not provide any explanation. [/INST] \n <Answer>"
}

plain_prompts = {
    "header": "You are given a scientific article and a question. Answer the " \
        "question as concisely as you can, using a single phrase or sentence if " \
        "possible. If the question cannot be answered based on the information in the " \
        "article, write \"unanswerable\". If the question is a yes/no question, answer " \
        "\"yes\", \"no\", or \"unanswerable\". Article:",
    "system_prompt": "You are a helpful, concise and honest assistant.",
    "concise_prompt": "Please highlight your final answer with a single pair of " \
        "<Answer> </Answer> tags. Do not provide any explanation. <Answer>"
}


def get_qasper(dataset_name, split, max_data_num, start_data_from):
    dataset = load_dataset("qasper", split=split)
    if dataset_name == "qasper":
        prompt_set = dialogue_prompts
    elif dataset_name == "qasper_plain":
        prompt_set = plain_prompts
    elif dataset_name == "qasper_free":
        prompt_set = free_prompts
    else:
        raise Exception("Invalid dataset name.")
    system_prompt = prompt_set["system_prompt"]
    header = prompt_set["header"]
    concise_prompt = prompt_set["concise_prompt"]

    def get_answer(answer_item):
        if answer_item["yes_no"] is not None:
            return {
                True: "yes",
                False: "no",
            }[answer_item["yes_no"]]
        elif answer_item["unanswerable"]:
            return "unanswerable"
        elif answer_item["free_form_answer"] != "":
            return answer_item["free_form_answer"]
        elif len(answer_item["extractive_spans"]) > 0:
            return " ".join(answer_item["extractive_spans"])
        else:
            raise Exception("No answer found.")

    detailed_dataset = []
    for _datum in dataset:
        full_text = ""
        for sec_name, paragraphs in zip(_datum["full_text"]["section_name"],
                                        _datum["full_text"]["paragraphs"]):
            paragraphs = " ".join(paragraphs)
            full_text += f"{sec_name} {paragraphs} "

        for question, answers in zip(_datum["qas"]["question"],
                                     _datum["qas"]["answers"]):
            prompt = f"{system_prompt} {header} {full_text} "\
                f"Question: {question} {concise_prompt}"
            target = list(map(get_answer, answers["answer"]))
            detailed_dataset.append({
                "prompt": prompt,
                "target": target,
                "id": _datum["id"],
                "title": _datum["title"],
                "abstract": _datum["abstract"],
                "question": _datum["qas"]["question"],
                "question_id": _datum["qas"]["question_id"],
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
