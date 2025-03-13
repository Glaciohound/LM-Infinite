import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
# import deepspeed
# from transformers.deepspeed import HfDeepSpeedConfig

# from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# from .model_base import Model_Base
# from .lambda_attention import lambda_matmul


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(vec, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    vec_embed = (vec * cos) + (rotate_half(vec) * sin)
    return vec_embed


# Efficient implementation using `models/lambda_attention.py`
def attn_forward_factory(
    self, layer_i, top_k_from_layer, gamma, shuffle_policy
):

    def limited_distance_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        dtype = query_states.dtype
        device = query_states.device

        if key_states.shape[-3] != self.num_heads:
            group_ratio = self.num_heads // key_states.size(-3)
            key_states = key_states.repeat_interleave(group_ratio, -3)
            value_states = value_states.repeat_interleave(group_ratio, -3)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            cache_kwargs = dict()
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        kv_seq_len = key_states.shape[-2]
        key_position_ids = torch.arange(kv_seq_len, device=device)[None]

        # inv_freq controls the dtype of rotation phase, which can be large
        self.rotary_emb.inv_freq = self.rotary_emb.inv_freq.to(torch.float32)
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos, sin = self.rotary_emb(value_states, key_position_ids)
        rot_query_states = apply_rotary_pos_emb(
            query_states, cos, sin, position_ids)
        rot_key_states = apply_rotary_pos_emb(
            key_states, cos, sin, key_position_ids)

        logits = rot_query_states.matmul(rot_key_states.transpose(-1, -2)) / np.sqrt(self.head_dim)
        logits = (logits + 50000).tril() - 50000

        # TODO: experimental
        if layer_i >= top_k_from_layer:
            quantile = gamma
            cap_value = torch.quantile(logits[0, 0, -1], quantile, dim=-1)

            if shuffle_policy == "disentangled":
                head_dim = self.head_dim
                dtype = query_states.dtype
                mean_k = key_states.mean(-2).unsqueeze(-2)
                # mean_q = query_states.mean(-1)
                key_axis = apply_rotary_pos_emb(
                    query_states,
                    cos, sin, position_ids[:, 1000, None]
                ).matmul(
                    (key_states - mean_k).transpose(-1, -2)
                ) / np.sqrt(head_dim)
                distance_axis = rot_query_states.matmul(
                    apply_rotary_pos_emb(
                        mean_k, cos, sin, position_ids).transpose(-1, -2)
                    ) / np.sqrt(head_dim)
                approx_logits = key_axis + distance_axis
                approx_logits = (approx_logits + 50000).tril() - 50000
                replaced_logits = torch.where(
                    logits > cap_value, logits,
                    approx_logits)

                # import matplotlib.pyplot as plt
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # axes[0].imshow(logits[0, 2].softmax(-1).cpu().numpy(), vmin=0, vmax=0.1)
                # axes[1].imshow(replaced_logits[0, 2].softmax(-1).cpu().numpy(), vmin=0, vmax=0.1)
                # plt.savefig("approx.pdf", dpi=800)
                # from IPython import embed; embed(); exit()

                logits = replaced_logits

            elif shuffle_policy == "average":
                # midddle value
                logits = torch.where(
                    logits > cap_value, logits,
                    logits[0, 0, -1].mean(),
                    # -1000
                )
                logits = (logits + 50000).tril() - 50000

            else:
                raise ValueError("Invalid shuffle policy")


        attn_weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.contiguous()
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return limited_distance_forward


