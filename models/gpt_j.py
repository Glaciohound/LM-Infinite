from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import GPTJForCausalLM, AutoTokenizer
from transformers.models.gptj.modeling_gptj import \
    create_sinusoidal_positions, is_torch_fx_proxy, apply_rotary_pos_emb

from .model_base import Model_Base
from .lambda_attention import lambda_matmul


# See https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
# for original implementation
def attn_forward_factory(
    self, use_lambda_mask, local_branch, global_branch,
    limit_distance, triangle_offset
):
    def _attn(
        query_rot,
        query_stationary,
        key_rot,
        key_stationary,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query_rot.size(-2), key_rot.size(-2)

        # If use_lambda_mask, we can use an efficient implementation
        if use_lambda_mask:
            headwise_limit = 33000  # magic number set for A100 GPU
            if query_length > headwise_limit:
                for head_i in range(self.num_attention_heads):
                    attn = (
                        lambda_matmul(
                            key_rot[:, head_i],
                            key_stationary[:, head_i],
                            query_rot[:, head_i],
                            query_stationary[:, head_i],
                            local_branch, global_branch
                        ) / self.scale_attn
                    ).softmax().to(value.dtype).dropout(self.attn_dropout)
                    if head_mask is not None:
                        attn = attn * head_mask[:, head_i]
                    query_rot[:, head_i] = attn.matmul(value[:, head_i])
                # At this size, we can't return the full attention
                attn_weights = None
            else:
                attn = (
                    lambda_matmul(
                        key_rot,
                        key_stationary,
                        query_rot,
                        query_stationary,
                        local_branch, global_branch
                    ) / self.scale_attn
                ).softmax().to(value.dtype).dropout(self.attn_dropout)
                if head_mask is not None:
                    attn = attn * head_mask
                attn_weights = attn.attn
                query_rot = attn.matmul(value)

        # If not use_lambda_mask, we use a costlier implementation
        else:
            all_attn_weights = []
            for head_i in range(self.num_attention_heads):
                attn_weights = torch.matmul(
                    query_rot[:, head_i], key_rot[:, head_i].transpose(-1, -2))
                mask_value = torch.finfo(attn_weights.dtype).min
                mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype
                                          ).to(attn_weights.device)
                if limit_distance is not None:
                    attn_weights = attn_weights.triu(
                        -limit_distance+1+key_length-query_length
                    ) + torch.matmul(
                        query_stationary[:, head_i],
                        key_stationary[:, head_i].transpose(-1, -2)
                    ).tril(-limit_distance+key_length-query_length)
                causal_mask = self.bias[
                    :, :, key_length - query_length: key_length, : key_length]
                attn_weights = torch.where(causal_mask[:, 0], attn_weights, mask_value)

                attn_weights = attn_weights / self.scale_attn

                if attention_mask is not None:
                    # Apply the attention mask
                    attn_weights = attn_weights + attention_mask[:, 0]

                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
                attn_weights = attn_weights.to(value.dtype)
                attn_weights = self.attn_dropout(attn_weights)

                # Mask heads if we want to
                if head_mask is not None:
                    attn_weights = attn_weights * head_mask[:, head_i]

                query_rot[:, head_i] = torch.matmul(attn_weights, value[:, head_i])
                all_attn_weights.append(attn_weights)

            attn_weights = torch.stack(all_attn_weights, dim=1)

        attn_output = query_rot.to(value.dtype)

        return attn_output, attn_weights

    def forward(
            hidden_states: torch.FloatTensor,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
        ) -> Union[
            Tuple[torch.Tensor, Tuple[torch.Tensor]],
            Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
        ]:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

            # shape = (batch_size, seq_len, num_heads, feature_dim)
            query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
            key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
            value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)
            kv_seq_len = key.size(1)
            q_len = query.size(1)

            if is_torch_fx_proxy(position_ids) or torch.jit.is_tracing():
                # The logic to conditionally copy to GPU could not be traced, so we do this
                # every time in the torch.fx case
                embed_positions = get_embed_positions(self.embed_positions, position_ids)
            else:
                embed_positions = self._get_embed_positions(position_ids)

            repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
            sincos = torch.gather(embed_positions, 1, repeated_position_ids)
            # shape = (batch_size, seq_len, feature_dim)
            sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

            def rotate_vector(_vector, _sin, _cos):
                if self.rotary_dim is not None:
                    _vector_rot = _vector[:, :, :, : self.rotary_dim]
                    _vector_pass = _vector[:, :, :, self.rotary_dim:]
                    _vector_rot = apply_rotary_pos_emb(_vector_rot, _sin, _cos)
                    _vector = torch.cat([_vector_rot, _vector_pass], dim=-1)
                else:
                    _vector = apply_rotary_pos_emb(_vector, _sin, _cos)
                return _vector

            key_rot = rotate_vector(key, sin, cos)
            query_rot = rotate_vector(query, sin, cos)
            if limit_distance is None:
                key_stationary = key_rot
                query_stationary = query_rot
            else:
                key_stationary = rotate_vector(
                    key, sin[:, 0, None], cos[:, 0, None])
                effective_limit_distance = min(limit_distance, kv_seq_len-1) \
                    if limit_distance is not None else kv_seq_len-1
                query_stationary = rotate_vector(
                    query, sin[:, effective_limit_distance, None],
                    cos[:, effective_limit_distance, None])

            # shape = (batch_size, num_heads, seq_len, feature_dim)
            key_rot = key_rot.permute(0, 2, 1, 3)
            query_rot = query_rot.permute(0, 2, 1, 3)
            key_stationary = key_stationary.permute(0, 2, 1, 3)
            query_stationary = query_stationary.permute(0, 2, 1, 3)

            # FIXME: the following needs to be corrected if for generation
            if layer_past is not None:
                past_key = layer_past[0]
                past_value = layer_past[1]
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            if use_cache is True:
                present = (key, value)
            else:
                present = None

            # compute self-attention: V x Softmax(QK^T)
            attn_output, attn_weights = _attn(
                query_rot, query_stationary, key_rot, key_stationary,
                value, attention_mask, head_mask)

            attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
            attn_output = self.out_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)

            return outputs  # a, present, (attentions)

    return forward


class GPT_J_Model(Model_Base):
    def __init__(
        self, model_name_or_path, max_length, fp16, truncation_side,
        use_lambda_mask, local_branch, global_branch,
        limit_distance, triangle_offset
    ):
        super().__init__(max_length, truncation_side)
        if fp16:
            print("using low_resource_mode and fp16")
            self.model = GPTJForCausalLM.from_pretrained(
                model_name_or_path, revision="float16",
                torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
        else:
            self.model = GPTJForCausalLM.from_pretrained(
                model_name_or_path,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.max_position_embeddings = max_length
        self.truncation_side = truncation_side

        # hack arguments
        self.fp16 = fp16
        self.use_lambda_mask = use_lambda_mask
        self.local_branch = local_branch
        self.global_branch = global_branch
        self.limit_distance = limit_distance
        self.triangle_offset = triangle_offset

        for hidden_layer in self.model.transformer.h:
            attn = hidden_layer.attn
            attn.embed_positions = create_sinusoidal_positions(
                max_length, attn.rotary_dim or attn.embed_dim)
            # adding lambda-mask
            if use_lambda_mask:
                attention_mask = None
            else:
                attention_mask = torch.tril(
                    torch.ones((max_length, max_length),
                               dtype=torch.bool)
                )
                attention_mask = attention_mask.view(
                    1, 1, max_length, max_length)
            attn.register_buffer(
                "bias", attention_mask,
                persistent=False,
            )

            attn.forward = attn_forward_factory(
                attn, use_lambda_mask, local_branch, global_branch,
                limit_distance, triangle_offset
            )
