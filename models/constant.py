import torch
from .model_base import Model_Base


class ConstantTokenizer():

    def convert_ids_to_tokens(self, ids):
        return None


class ConstantModel(Model_Base):

    def __init__(
        self, max_length, truncation_side, constant_answer
    ):
        super().__init__(max_length, truncation_side)
        self.constant_answer = constant_answer
        self.tokenizer = ConstantTokenizer()

    def generate(self, input_ids, attention_mask, max_generation_length,
                 suppress_tokens, do_sample):
        return [self.constant_answer], torch.ones(1, 20)

    def tokenize(self, text_batch):
        return torch.ones(1, 10), torch.ones(1, 10)
