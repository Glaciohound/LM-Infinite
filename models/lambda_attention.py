import torch


def pad_sequence_to_length(tensor, length, value):
    return torch.cat(
        (tensor,
         torch.zeros(*tensor.shape[:-2],
                     length-tensor.shape[-2], tensor.shape[-1],
                     dtype=tensor.dtype
                     ).to(tensor.device)
         ), dim=-2
    )


def blockwise_sequence(sequence, block_size):
    pad_to_length = ((sequence.shape[-2]-1) // block_size + 1
                     ) * block_size
    padded = pad_sequence_to_length(
        sequence, pad_to_length, 0
    )
    blockwise = padded.view(
        *sequence.shape[:-2],
        pad_to_length // block_size, block_size, -1
    )
    return blockwise


def shift_and_pair(blockwise):
    return torch.cat((
        torch.cat((
            blockwise[..., -1, None, :, :],
            blockwise[..., :-1, :, :],
        ), -3),
        blockwise
    ), -2)


class Lambda_Attention_Matrix:
    def __init__(self, key_rot, key_stationary, query_rot, query_stationary,
                 local_branch, global_branch):
        query_length = query_rot.shape[-2]
        key_length = key_rot.shape[-2]
        embed_dim = key_rot.shape[-1]
        dtype = key_rot.dtype
        device = key_rot.device
        min_value = torch.finfo(dtype).min

        if query_length < key_length:
            assert query_rot.shape[0] == 1
            self.mode = "single_query"
        elif query_length <= local_branch:
            self.mode = "short_seq"
        else:
            self.mode = "long_seq"

        attn_stationary = torch.matmul(
            query_stationary,
            key_stationary[..., :global_branch, :].transpose(-1, -2)
        )
        attn_stationary = torch.where(
            torch.ones(attn_stationary.shape[-2:], dtype=torch.bool).to(
                device).triu(-local_branch+1+key_length-query_length),
            min_value, attn_stationary
        )

        if self.mode == "short_seq":
            attn_rot = torch.matmul(
                query_rot, key_rot.transpose(-1, -2)
            )
            attn_rot = torch.where(
                torch.ones(attn_rot.shape[-2:], dtype=torch.bool).to(
                    device).triu(1),
                min_value, attn_rot
            )
            self.attn = attn_rot
            self.attn[..., :, :global_branch] = torch.where(
                attn_stationary > min_value / 2,
                attn_stationary,
                attn_rot[..., :, :global_branch]
            )
        elif self.mode == "single_query":
            attn_rot = torch.matmul(
                query_rot,
                key_rot[..., max(0, key_length-local_branch):,
                        :].transpose(-1, -2)
            )
            self.attn = torch.cat((
                attn_stationary, attn_rot), -1)
        else:
            pad_to_length = ((query_length-1) // local_branch + 1
                             ) * local_branch
            patch_size = pad_to_length - query_length
            segmented_query_rot = blockwise_sequence(query_rot, local_branch)
            segmented_key_rot = blockwise_sequence(key_rot, local_branch)
            segmented_key_rot = shift_and_pair(segmented_key_rot)
            attn_rot = torch.matmul(
                segmented_query_rot, segmented_key_rot.transpose(-1, -2)
            )
            attn_rot = torch.where(
                torch.ones(
                    (local_branch, 2*local_branch), dtype=torch.bool
                ).to(device).triu(1).tril(local_branch).logical_not(),
                min_value, attn_rot
            )
            attn_rot[..., 0, :, :local_branch] = min_value
            if patch_size != 0:
                attn_rot[..., -1, -patch_size:, :] = min_value
                attn_rot[..., -1, :, -patch_size:] = min_value
            attn_rot = attn_rot.view(
                query_rot.shape[:-2] + (-1, local_branch*2)
            )
            attn_stationary = pad_sequence_to_length(
                attn_stationary, pad_to_length, min_value
            )
            self.pad_to_length = pad_to_length
            self.attn = torch.cat((
                attn_stationary, attn_rot), -1)

        self.query_length = query_length
        self.key_length = key_length
        self.min_value = min_value
        self.global_branch = global_branch
        self.local_branch = local_branch
        self.embed_dim = embed_dim

    def __truediv__(self, scalar):
        self.attn.div_(scalar).clamp_(min=self.min_value)
        return self

    def __mul__(self, scalar):
        self.attn.mul_(scalar).clamp_(min=self.min_value)
        return self

    def local_branch_add(self, other):
        self.attn[..., -self.local_branch*2:].add_(other).clamp_(
            min=self.min_value)
        return self

    def global_branch_add(self, other):
        self.attn[..., :-self.local_branch*2].add_(other).clamp_(
            min=self.min_value)
        return self

    def dropout(self, dropout):
        self.attn = dropout(self.attn)
        return self

    def softmax(self):
        self.attn = self.attn.softmax(-1)
        return self

    def to(self, destination):
        self.attn = self.attn.to(destination)
        return self

    def matmul(self, value):
        if self.mode == "short_seq":
            output = torch.matmul(self.attn, value)
        elif self.mode == "single_query":
            output = torch.matmul(
                self.attn,
                torch.cat((
                    value[..., :self.global_branch, :],
                    value[..., max(0, self.key_length-self.local_branch):, :]
                ), -2)
            )
        else:
            segmented_value = shift_and_pair(blockwise_sequence(
                value, self.local_branch))
            output_stationary = torch.matmul(
                self.attn[..., :self.query_length, :self.global_branch],
                value[..., :self.global_branch, :]
            )
            output_rot = torch.matmul(
                self.attn[..., self.global_branch:].view(
                    self.attn.shape[:-2] +
                    (-1, self.local_branch, self.local_branch*2)
                ),
                segmented_value
            ).view(
                self.attn.shape[:-2] + (self.pad_to_length, -1)
            )
            output = output_stationary + output_rot[..., :self.query_length, :]
        del self.attn
        return output


lambda_matmul = Lambda_Attention_Matrix
