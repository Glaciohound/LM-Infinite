import torch
import math
import warnings
from einops import rearrange
from .model_base import Model_Base
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# Legacy code from mpt_7b.py, for reference only:
#
# def scaled_multihead_dot_product_attention(query, key, value, n_heads, past_key_value=None, softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False):
#     q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
#     kv_n_heads = 1 if multiquery else n_heads
#     k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
#     v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)
#     if past_key_value is not None:
#         if len(past_key_value) != 0:
#             k = torch.cat([past_key_value[0], k], dim=3)
#             v = torch.cat([past_key_value[1], v], dim=2)
#         past_key_value = (k, v)
#     (b, _, s_q, d) = q.shape
#     s_k = k.size(-1)
#     if softmax_scale is None:
#         softmax_scale = 1 / math.sqrt(d)
#     attn_weight = q.matmul(k) * softmax_scale
#     if attn_bias is not None:
#         _s_q = max(0, attn_bias.size(2) - s_q)
#         _s_k = max(0, attn_bias.size(3) - s_k)
#         attn_bias = attn_bias[:, :, _s_q:, _s_k:]
#         if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
#             raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
#         attn_weight = attn_weight + attn_bias
#     min_val = torch.finfo(q.dtype).min
#     if key_padding_mask is not None:
#         if attn_bias is not None:
#             warnings.warn('Propogating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unneccessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
#         attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
#     if is_causal and (not q.size(2) == 1):
#         s = max(s_q, s_k)
#         causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
#         causal_mask = causal_mask.tril()
#         causal_mask = causal_mask.to(torch.bool)
#         causal_mask = ~causal_mask
#         causal_mask = causal_mask[-s_q:, -s_k:]
#         attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     if dropout_p:
#         attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)
#     out = attn_weight.to(v.dtype).matmul(v)
#     out = rearrange(out, 'b h s d -> b s (h d)')
#     if needs_weights:
#         return (out, attn_weight, past_key_value)
#     return (out, None, past_key_value)


def attn_forward_factory(use_lambda_attention, local_branch, global_branch,
                         limit_distance, triangle_offset):

    def scaled_multihead_dot_product_attention(
            query, key, value, n_heads, past_key_value=None,
            softmax_scale=None, attn_bias=None, key_padding_mask=None,
            is_causal=False, dropout_p=0.0, training=False,
            needs_weights=False, multiquery=False):
        q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
        kv_n_heads = 1 if multiquery else n_heads
        k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
        v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)
        if past_key_value is not None:
            if len(past_key_value) != 0:
                k = torch.cat([past_key_value[0], k], dim=3)
                v = torch.cat([past_key_value[1], v], dim=2)
            past_key_value = (k, v)
        (b, _, s_q, d) = q.shape
        s_k = k.size(-1)
        out = q
        for head_i in range(n_heads):
            if softmax_scale is None:
                softmax_scale = 1 / math.sqrt(d)
            attn_weight = q[:, head_i].matmul(k[:, head_i]) * softmax_scale
            assert attn_bias.shape[2] == 1, "it seems to be implemented this way"
            head_attn_bias = attn_bias[:, head_i].repeat(1, s_q, 1)
            if limit_distance is not None and limit_distance < s_k:
                limited_offset = attn_bias[0, head_i, 0].clone()
                limited_offset[limit_distance-1:] = limited_offset[
                    :-limit_distance+1].clone()
                limited_offset = limited_offset[-s_q:]
                limited_offset[:limit_distance-1] = 0
                limited_mask = torch.ones_like(head_attn_bias).tril(-limit_distance+s_k-s_q)
                head_attn_bias.masked_fill_(limited_mask.bool(), 0)
                limited_mask = limited_mask * limited_offset[None, :, None]
                head_attn_bias = head_attn_bias + limited_mask
            if attn_bias is not None:
                _s_q = max(0, attn_bias.size(2) - s_q)
                _s_k = max(0, attn_bias.size(3) - s_k)
                head_attn_bias = head_attn_bias[:, _s_q:, _s_k:]
                if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
                    raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
                attn_weight = attn_weight + head_attn_bias
            min_val = torch.finfo(q.dtype).min
            if key_padding_mask is not None:
                if attn_bias is not None:
                    warnings.warn('Propogating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unneccessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
                attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
            if is_causal and (not q.size(2) == 1):
                s = max(s_q, s_k)
                causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
                causal_mask = causal_mask.tril()
                causal_mask = causal_mask.to(torch.bool)
                causal_mask = ~causal_mask
                if use_lambda_attention:
                    lambda_mask = torch.ones_like(causal_mask)
                    lambda_mask = lambda_mask.tril(-local_branch)
                    lambda_mask[:, :global_branch] = 0
                    causal_mask = causal_mask + lambda_mask
                causal_mask = causal_mask[-s_q:, -s_k:]
                attn_weight = attn_weight.masked_fill(causal_mask.view(1, s_q, s_k), min_val)
            # from IPython import embed; embed(); exit()
            attn_weight = torch.softmax(attn_weight, dim=-1)
            if dropout_p:
                attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)
            out[:, head_i] = attn_weight.to(v.dtype).matmul(v[:, head_i])
        out = rearrange(out, 'b h s d -> b s (h d)')
        if needs_weights:
            return (out, attn_weight, past_key_value)
        return (out, None, past_key_value)

    return scaled_multihead_dot_product_attention


class MPT_7B_Model(Model_Base):
    def __init__(self, model_name_or_path, max_length, truncation_side,
                 use_lambda_attention, efficient_implementation,
                 local_branch, global_branch,
                 limit_distance, triangle_offset):
        super().__init__(max_length, truncation_side)
        self.config = AutoConfig.from_pretrained(model_name_or_path,
                                                 trust_remote_code=True)
        self.config.max_seq_len = max_length
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=self.config,
            trust_remote_code=True
        )
        self.use_lambda_attention = use_lambda_attention
        self.local_branch = local_branch
        self.global_branch = global_branch
        self.limit_distance = limit_distance
        self.triangle_offset = triangle_offset

        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        for hidden_layer in self.model.transformer.blocks:
            attn = hidden_layer.attn
            attn.attn_fn = attn_forward_factory(
                use_lambda_attention, local_branch, global_branch,
                limit_distance, triangle_offset
            )
