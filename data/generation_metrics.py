import evaluate


def generation_overall_metric(all_results, evaluate_positions):
    all_scores = {}
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    for position in evaluate_positions:
        all_scores[position] = {}

        position_predictions = [
            _result["generation_results"][position]["generation"]
            for _result in all_results
            if position in _result["generation_results"]
        ]
        position_predictions = [
            _l if _l != "" else "." for _l in position_predictions
        ]
        position_references = [
            _result["generation_results"][position]["target"]
            for _result in all_results
            if position in _result["generation_results"]
        ]
        if len(position_predictions) == 0:
            continue
        all_scores[position]["rouge"] = rouge.compute(
            predictions=position_predictions,
            references=position_references,
        )
        all_scores[position]["bleu"] = bleu.compute(
            predictions=position_predictions,
            references=position_references,
        )
    return all_scores
