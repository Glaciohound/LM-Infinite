import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_Base(nn.Module):
    def __init__(self, max_length, truncation_side):
        super().__init__()
        self.max_length = max_length
        self.truncation_side = truncation_side
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def tokenize(self, text_batch):
        encoded_input = self.tokenizer(
            text_batch, return_tensors='pt',
            padding=True, truncation=False,
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        if self.max_length is not None and \
                encoded_input["input_ids"].shape[1] > self.max_length:
            # Truncate the input
            # In this task we almost only use batch size 1 so it's fine to do
            # truncation by our own
            if self.truncation_side == "left":
                input_ids = input_ids[:, -self.max_length:]
                attention_mask = attention_mask[:, -self.max_length:]
            elif self.truncation_side == "right":
                input_ids = input_ids[:, :self.max_length]
                attention_mask = attention_mask[:, :self.max_length]
            elif self.truncation_side == "center":
                input_ids = torch.cat([
                    input_ids[:, :self.max_length // 2],
                    input_ids[:, -self.max_length // 2:]
                ], dim=1)
                attention_mask = torch.cat([
                    attention_mask[:, :self.max_length // 2],
                    attention_mask[:, -self.max_length // 2:]
                ], dim=1)

        return input_ids, attention_mask

    def forward_features(self, input_ids, attention_mask,
                         output_attentions=False, output_hidden_states=False,
                         use_cache=False, past_key_values=None):
        output = self.model(input_ids=input_ids,
                            use_cache=use_cache,
                            attention_mask=attention_mask,
                            return_dict=True,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            past_key_values=past_key_values
                            )
        logits = output["logits"]
        batch_size, length, _ = logits.shape
        token_nll = F.cross_entropy(
            logits[:, :-1].reshape(batch_size*(length-1), -1),
            input_ids[:, 1:].reshape(-1),
            reduction="none"
        ).reshape(batch_size, -1)
        token_nll_list = [
            _nll[:_mask.sum()-1]
            for _nll, _mask in zip(token_nll, attention_mask)
        ]
        output["token_nll_list"] = token_nll_list
        return output

    def forward(self, *args, **kwargs):
        return self.forward_features(
            *args, **kwargs
        )

    def generate(self, input_ids, attention_mask,
                 max_generation_length, min_new_tokens,
                 suppress_tokens, do_sample, stopping_criteria):
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_generation_length,
            min_new_tokens=min_new_tokens,
            suppress_tokens=suppress_tokens,
            stopping_criteria=stopping_criteria,
        )
        decoded = [
            self.tokenizer.decode(_o[_a.sum():], skip_special_tokens=True)
            for _a, _o in zip(attention_mask, output_ids)
        ]
        return decoded, output_ids
