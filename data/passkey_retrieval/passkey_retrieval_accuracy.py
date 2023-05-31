import re


def passkey_retrieval_accuracy(predictions, references, input_batch):
    def extract_answer(text):
        answer = re.findall(r'\d+', text)
        if len(answer) == 0:
            return 0
        else:
            return int(answer[0])

    predictions = [extract_answer(pred) for pred in predictions]
    references = [extract_answer(ref) for ref in references]
    accuracy = sum(pred == ref for pred, ref in zip(predictions, references)
                   ) / len(predictions)
    return {
        "accuracy": accuracy
    }
