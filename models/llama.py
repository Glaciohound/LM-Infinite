from typing import Optional, Tuple
import torch
import math
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .model_base import Model_Base
from .lambda_attention import lambda_matmul


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


def detailed_lambda_attention(
    query_states, key_states, value_states,
    rot_query_states, rot_key_states,
    stationary_query_states, stationary_key_states,
    cos, sin,
    attention_mask, head_dim, device, dtype,
    q_len, kv_seq_len,
    global_branch, local_branch, limit_distance,
    top_k_attention, top_k_insert_at, top_k_from_layer, top_k_to_layer,
    shuffle_policy, layer_i
):
    # key/value_states.shape == [bs, heads, kv_seq_len, head_dim]
    # query_states.shape == [bs, heads, q_len, head_dim]
    # cos/sin.shape == [1, kv_seq_len, head_dim]
    attn_weights = rot_query_states.matmul(
        rot_key_states.transpose(-1, -2)) / math.sqrt(head_dim)
    # FIXME: experimental
    if len(shuffle_policy) > 0 and \
            shuffle_policy[0].startswith("k_q_simplify") and (
                str(layer_i) in shuffle_policy[0].split("=")[1].split(":")
                or "all" in shuffle_policy[0].split("=")[1].split(":")
            ):
        assert top_k_attention is None
        mean_cos = cos[0].mean(0)
        mean_sin = sin[0].mean(0)
        mean_k = key_states[0].mean(1)
        mean_q = query_states[0].mean(1)
        mean_rot_q = mean_q * mean_cos + rotate_half(mean_q) * mean_sin
        semantic = mean_rot_q[:, None].matmul(
            key_states[0].transpose(-1, -2)) / math.sqrt(head_dim)
        simple_logits = semantic.repeat(1, kv_seq_len, 1)
        position_ids = torch.arange(kv_seq_len, device=device)[None]
        position = apply_rotary_pos_emb(
                mean_q[None, :, None], cos, sin, position_ids)[0].matmul(
                    mean_k[..., None]
                ).squeeze(-1) / math.sqrt(head_dim)
        for i in range(kv_seq_len):
            simple_logits[:, i, :i+1] += position[:, :i+1].flip(-1)
        attn_weights = simple_logits[None]
        # attn_weights = torch.randn_like(attn_weights)

    # attn_weights.shape == [bs, heads, q_len, kv_seq_len]
    dtype = attn_weights.dtype
    _min = torch.finfo(dtype).min

    attn_stationary = stationary_query_states.matmul(
        stationary_key_states.transpose(-1, -2)
    ) / math.sqrt(head_dim)

    if limit_distance is not None:
        attn_weights = attn_weights.triu(-limit_distance+1+kv_seq_len-q_len)
        # lower-triangular limited distance zone
        attn_weights += attn_stationary.tril(
            -limit_distance+kv_seq_len-q_len
        )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, 0]
        attn_weights = torch.max(
            attn_weights,
            torch.tensor(_min, device=device)
        )
    lambda_mask = torch.ones_like(attn_weights).to(bool)
    lambda_mask = lambda_mask.tril(
        -local_branch+kv_seq_len-q_len)
    lambda_mask[..., :global_branch] = False
    last_line_attn_weights = attn_weights[..., -1, None, :].clone()
    attn_weights.masked_fill_(
        lambda_mask, _min)
    # last_line_attn_weights.shape == [bs, heads, 1, kv_seq_len]

    col_high = - top_k_insert_at + kv_seq_len
    if top_k_attention is not None and \
            top_k_from_layer <= layer_i and top_k_to_layer > layer_i and \
            col_high >= global_branch + top_k_attention:

        # virtual attention to calculate the top-k
        if "top_agnostic" in shuffle_policy or \
                "top_max" in shuffle_policy:
            agnostic_query_states = \
                (query_states * cos[0, top_k_insert_at]) + \
                (rotate_half(query_states) * sin[0, top_k_insert_at])
            agnostic_attention = agnostic_query_states[
                ..., -1, None, :].matmul(
                    key_states.transpose(-1, -2)
                ) / math.sqrt(head_dim)
            if "certain_heads=" in ",".join(shuffle_policy):
                certain_heads = next(filter(
                    lambda x: "certain_heads=" in x,
                    shuffle_policy))
                certain_heads = certain_heads.split("=")[1].split(":")
                certain_heads = list(map(
                    lambda x: int(x.split("-")[1]),
                    filter(
                        lambda x: x.split("-")[0] == str(layer_i),
                        certain_heads
                    )
                ))
                virtual_attention = last_line_attn_weights.clone()
                for head_i in certain_heads:
                    virtual_attention[:, head_i] = \
                        agnostic_attention[:, head_i]
            else:
                virtual_attention = agnostic_attention
            if "top_max" in shuffle_policy:
                virtual_attention = virtual_attention.max(
                    last_line_attn_weights)
        elif "top_positional_limited" in shuffle_policy:
            virtual_attention = last_line_attn_weights
        else:
            raise ValueError("Invalid shuffle_policy")

        virtual_attention = virtual_attention[..., global_branch: col_high]
        _, top_k_indices = torch.topk(
            virtual_attention, top_k_attention, dim=-1)
        # top_k_indices.shape == [bs, heads, 1, top_k_attention]

        if "reorder_top_k" in shuffle_policy or \
                "squash_top_k" in shuffle_policy:
            # Reorder the key, query, and value states
            # shape == [bs, heads, top_k_attention, head_dim]
            if "squash_top_k" in shuffle_policy:
                top_k_indices, _ = top_k_indices.sort(-1)
                top_k_indices = top_k_indices.flip(-1)
            reordered_key_states = torch.take_along_dim(
                key_states,
                top_k_indices.flip(-1)[:, :, 0, :, None]+global_branch,
                dim=2)
            reordered_key_states = (
                reordered_key_states*cos[0, :top_k_attention] +
                rotate_half(reordered_key_states)*sin[0, :top_k_attention]
            )
            reordered_query_states = (
                query_states * cos[0, top_k_attention+top_k_insert_at-1] +
                rotate_half(query_states) *
                sin[0, top_k_attention+top_k_insert_at-1]
            )
            reordered_value_states = torch.take_along_dim(
                value_states,
                top_k_indices.flip(-1)[:, :, 0, :, None]+global_branch,
                dim=2
            )

            # calculate the reordered attention
            reordered_attention = \
                reordered_query_states.matmul(
                    reordered_key_states.transpose(-1, -2)
                ).squeeze(2) / math.sqrt(head_dim)
            # reordered_attention.shape == [bs, heads, top_k_attention]
            attn_weights[..., -1, global_branch:col_high] = \
                _min
            attn_weights[..., -1, col_high-top_k_attention:col_high] = \
                reordered_attention

            # calculate the vanilla output
            output1 = torch.matmul(
                F.softmax(
                    attn_weights[..., :-1], dim=-1,
                    dtype=torch.float32).to(dtype),
                value_states
            ) if q_len > 1 else value_states[:, :, :0]

            # calculate the reordered output
            value_states = value_states.clone()
            value_states[:, :, col_high-top_k_attention: col_high] = \
                reordered_value_states
            output2 = torch.matmul(
                F.softmax(
                    attn_weights[:, :, -1:],
                    dim=-1, dtype=torch.float32).to(dtype),
                value_states
            )

            return torch.cat((output1, output2), dim=-2)

        elif "select_top_k" in shuffle_policy:
            top_k_mask = torch.ones_like(virtual_attention).to(bool)
            top_k_mask.scatter_(-1, top_k_indices, 0)
            attn_weights[..., -1, global_branch: col_high] = \
                virtual_attention.masked_fill(
                    top_k_mask, _min
                ).squeeze(2)
        else:
            raise ValueError("Invalid shuffle_policy")

    return torch.matmul(
        F.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(dtype),
        value_states
    )


# Efficient implementation using `models/lambda_attention.py`
def attn_forward_factory(
    self, use_lambda_attention, local_branch, global_branch,
    limit_distance,
    top_k_attention, top_k_insert_at, top_k_from_layer, top_k_to_layer,
    shuffle_policy, layer_i
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
        if limit_distance is None:
            stationary_key_states = rot_key_states
            stationary_query_states = rot_query_states
        else:
            stationary_key_states = key_states
            effective_limit_distance = min(limit_distance, kv_seq_len-1)
            stationary_query_states = \
                (query_states * cos[0, effective_limit_distance]) + \
                (rotate_half(query_states) * sin[0, effective_limit_distance])

        # # FIXME: remove this
        # import pickle
        # filename = f"key_query_{layer_i}.pkl"
        # dict_to_save = {
        #     "query_states": query_states.cpu().numpy(),
        #     "key_states": key_states.cpu().numpy(),
        #     "cos": cos.cpu().numpy(),
        #     "sin": sin.cpu().numpy(),
        # }
        # with open(filename, "wb") as f:
        #     pickle.dump(dict_to_save, f)

        headwise_limit = 33000  # magic number set for A100 GPU
        # If use_lambda_attention, we can use an efficient implementation
        if use_lambda_attention and top_k_attention is not None:
            if q_len > headwise_limit:
                for head_i in range(self.num_heads):
                    query_states[:, head_i, :-1] = (
                        lambda_matmul(
                            rot_key_states[:, head_i],
                            stationary_key_states[:, head_i],
                            rot_query_states[:, head_i],
                            stationary_query_states[:, head_i],
                            local_branch, global_branch
                        ) / math.sqrt(self.head_dim)
                    ).softmax().matmul(value_states[:, head_i])[
                        :, :-1]
            else:
                query_states[:, :, :-1] = (
                    lambda_matmul(
                        rot_key_states,
                        stationary_key_states,
                        rot_query_states,
                        stationary_query_states,
                        local_branch, global_branch
                    ) / math.sqrt(self.head_dim)
                 ).softmax().matmul(value_states)[:, :, :-1]

            query_states[:, :, -1] = detailed_lambda_attention(
                query_states[:, :, -1, None], key_states, value_states,
                rot_query_states[:, :, -1, None], rot_key_states,
                stationary_query_states[:, :, -1, None], stationary_key_states,
                cos, sin,
                None if position_ids is None else
                attention_mask[:, :, -1, None] if attention_mask is not None
                else None,
                self.head_dim, device, dtype,
                1, kv_seq_len,
                global_branch, local_branch,
                limit_distance,
                top_k_attention, top_k_insert_at,
                top_k_from_layer, top_k_to_layer,
                shuffle_policy, layer_i
            ).squeeze(2)

        elif use_lambda_attention:
            if q_len > headwise_limit:
                for head_i in range(self.num_heads):
                    query_states[:, head_i] = (
                        lambda_matmul(
                            rot_key_states[:, head_i],
                            stationary_key_states[:, head_i],
                            rot_query_states[:, head_i],
                            stationary_query_states[:, head_i],
                            local_branch, global_branch
                        ) / math.sqrt(self.head_dim)
                    ).softmax().matmul(value_states[:, head_i])
            else:
                query_states = (
                    lambda_matmul(
                        rot_key_states,
                        stationary_key_states,
                        rot_query_states,
                        stationary_query_states,
                        local_branch, global_branch
                    ) / math.sqrt(self.head_dim)
                 ).softmax().matmul(value_states)

        # If not use_lambda_attention, we use a costlier implementation
        else:
            query_states = detailed_lambda_attention(
                query_states, key_states,
                value_states,
                rot_query_states, rot_key_states,
                stationary_query_states,
                stationary_key_states,
                cos, sin, attention_mask, self.head_dim, device, dtype,
                q_len, kv_seq_len,
                global_branch, local_branch,
                limit_distance,
                top_k_attention, top_k_insert_at,
                top_k_from_layer, top_k_to_layer,
                shuffle_policy, layer_i
            )
            # for head_i in range(self.num_heads):
            #     query_states[:, head_i] = detailed_lambda_attention(
            #         query_states[:, head_i], key_states[:, head_i],
            #         value_states[:, head_i],
            #         rot_query_states[:, head_i], rot_key_states[:, head_i],
            #         stationary_query_states[:, head_i],
            #         stationary_key_states[:, head_i],
            #         cos, sin, attention_mask, self.head_dim, device, dtype,
            #         q_len, kv_seq_len,
            #         global_branch, local_branch,
            #         limit_distance,
            #         top_k_attention, top_k_insert_at,
            #         top_k_from_layer, top_k_to_layer,
            #         shuffle_policy, layer_i
            #     )

        attn_output = query_states
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


class LLAMA_Model(Model_Base):
    wrap_module = LlamaDecoderLayer

    def __init__(
        self, model_name_or_path, tokenizer_path, max_length, truncation_side,
        load_in_4bit, device_map,
        use_lambda_attention, local_branch, global_branch,
        limit_distance,
        top_k_attention, top_k_insert_at, top_k_from_layer, top_k_to_layer,
        shuffle_policy
    ):
        super().__init__(max_length, truncation_side)
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_4bit=load_in_4bit, device_map=device_map
        )
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # hack arguments
        self.use_lambda_attention = use_lambda_attention
        self.local_branch = local_branch
        self.global_branch = global_branch
        self.limit_distance = limit_distance
        self.top_k_attention = top_k_attention
        self.top_k_insert_at = top_k_insert_at
        self.top_k_from_layer = top_k_from_layer
        self.top_k_to_layer = top_k_to_layer
        self.shuffle_policy = shuffle_policy

        for layer_i, hidden_layer in enumerate(self.model.model.layers):
            attn = hidden_layer.self_attn
            attn.forward = attn_forward_factory(
                attn, use_lambda_attention, local_branch, global_branch,
                limit_distance,
                top_k_attention, top_k_insert_at,
                top_k_from_layer, top_k_to_layer,
                shuffle_policy, layer_i
            )

        if use_lambda_attention:
            self.model.model._prepare_decoder_attention_mask = \
                lambda *args, **kwargs: None

    def to(self, device):
        if self.load_in_4bit:
            return self

        self.device = device
        self.model.to(device)
        return self


def convert_llama_model(model, local_branch, global_branch):
    for layer_i, hidden_layer in enumerate(model.model.layers):
        attn = hidden_layer.self_attn
        attn.forward = attn_forward_factory(
            attn, True, local_branch, global_branch, local_branch, 0,
            None, None, None, None, layer_i
        )
    return model
