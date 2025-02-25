import torch
from torch import Tensor, nn
from torch.nn import GLU, Conv1d, Dropout, GELU
from typing import Iterable, Optional, Tuple, final
import torch.nn.functional as F
import pdb

def generate_batch(batch_size, max_length, dim):
    actual_lengths = torch.randint(low=1, high=max_length+1, size=(batch_size,))
    input_data = torch.zeros(batch_size, max_length, dim)
    input_mask = torch.zeros(batch_size, max_length, dtype=torch.int64)
    
    for i in range(batch_size):
        length = actual_lengths[i]
        input_data[i, :length, :] = torch.randn(length, dim)
        input_mask[i, :length] = 1
    
    return input_data, input_mask


class MultiHeadAttention(nn.Module):
    # come from https://github.com/openai/whisper/blob/main/whisper/model.py
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state, bias=True)
        self.key = nn.Linear(n_state, n_state, bias=True)
        self.value = nn.Linear(n_state, n_state, bias=True)
        self.out = nn.Linear(n_state, n_state, bias=True)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk += mask

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


def _compute_new_paddingTensor(
    seqs: Tensor, padding_mask: Optional[Tensor], kernel_size: int, stride: int
) -> Optional[Tensor]:
    if padding_mask is None:
        return padding_mask

    pad = kernel_size // 2
    seq_lens = padding_mask.sum(-1)

    seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1

    seq_lens = seq_lens.floor().to(torch.int64) 
    batch_seq_len=seqs.size(1)
    batch_size = seq_lens.size(0)

    indices = torch.arange(batch_seq_len, device=seq_lens.device).expand(batch_size, -1)
    lengths = seq_lens.unsqueeze(1).expand(-1, batch_seq_len)
    return indices < lengths


class TransformerAdaptorLayer(nn.Module):
    def __init__(
        self,
        model_dim=1024,
        kernel_size=8,
        stride=8,
        dropout_p=0.1
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.residual_layer_norm = nn.LayerNorm(model_dim)
        self.residual_conv = Conv1d(
                    model_dim,
                    model_dim * 2,
                    kernel_size,
                    stride,
                    padding=kernel_size // 2
                )
        self.residual_activation = GLU(dim=1)
        self.self_attn_layer_norm = nn.LayerNorm(model_dim)
        self.self_attn_conv = Conv1d(
                    model_dim,
                    model_dim * 2,
                    kernel_size,
                    stride,
                    padding=kernel_size // 2
                )
        self.self_attn_activation = GLU(dim=1)
        self.self_attn = MultiHeadAttention(n_state=model_dim, n_head=16)

        self.self_attn_dropout = Dropout(dropout_p)
        self.ffn_layer_norm = nn.LayerNorm(model_dim)

        self.ffn = nn.Sequential(
                nn.Linear(model_dim,model_dim*2),
                GELU(approximate='none'),
                nn.Linear(model_dim*2,model_dim)
                )
        self.ffn_dropout = Dropout(dropout_p)

    def forward(
        self,
        seqs: Tensor,
        seqs_len: Optional[Tensor]=None,
        padding_mask: Optional[Tensor]=None,
        self_attn_mask: Optional[Tensor]=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        residual = self.residual_layer_norm(seqs) 
        residual = residual.transpose(1,2)
        residual = self.residual_conv(residual)
        residual = self.residual_activation(residual)

        # (N, M, S) -> (N, S, M)
        residual = residual.transpose(1, 2)

        seqs = self.self_attn_layer_norm(seqs)

        # Apply pooling before feeding to the multihead-attention layer.
        # (N, S, M) -> (N, M, S)
        seqs = seqs.transpose(1, 2)
        seqs = self.self_attn_conv(seqs)


        seqs = self.self_attn_activation(seqs)

        # (N, M, S) -> (N, S, M)
        seqs = seqs.transpose(1, 2)

        if seqs_len is None:  ## for whisper
            padding_mask = _compute_new_paddingTensor(
                seqs, padding_mask, self.kernel_size, self.stride
            )
            seqs_len = padding_mask.sum(-1)

        # The rest of the computation is identical to a vanilla Transformer
        # encoder layer.
        B,T,D = seqs.shape
        masks,padding_mask = self.creat_attention_mask(T,seqs_len)
        device = seqs.device
        masks = masks.to(device)
        seqs,_ = self.self_attn(
            x = seqs,
            mask = masks,
        )
        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual
        seqs = seqs*padding_mask.unsqueeze(-1)

        residual = seqs
        seqs = self.ffn_layer_norm(seqs)
        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)
        seqs = seqs + residual
        return seqs, padding_mask
        

    def creat_attention_mask(self,seq_length,actual_lengths):
        masks = []
        for actual_length in actual_lengths:
            mask = torch.full((seq_length, seq_length), float('-inf'))
            mask[:, :actual_length] = 0
            masks.append(mask)
        batch_mask = torch.stack(masks)


        seq_lens = actual_lengths
        batch_seq_len=seq_length
        batch_size = actual_lengths.size(0)

        indices = torch.arange(batch_seq_len, device=seq_lens.device).expand(batch_size, -1)
        lengths = seq_lens.unsqueeze(1).expand(-1, batch_seq_len)
        return batch_mask.unsqueeze(1), indices< lengths
        

if __name__ == "__main__":
    adaptor = TransformerAdaptorLayer(kernel_size=8, stride=8, dropout_p=0.1,model_dim=1280)
    batch_size = 4
    max_length = 412
    dim = 1280
    input, input_mask = generate_batch(batch_size, max_length, dim)
    seqs, padding_mask = adaptor(input,input_mask)
    print(seqs.shape)
