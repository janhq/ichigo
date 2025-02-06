import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


# RoPE
class Rotary(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    Args:
        dim (int): Dimension of the embedding
        base (int, optional): Base for frequency computation. Defaults to 10000.
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        """
        Compute rotary embeddings.

        Args:
            x (Tensor): Input tensor
            seq_dim (int, optional): Sequence dimension. Defaults to 1.

        Returns:
            tuple: Cached cosine and sine embeddings
        """
        seq_len = x.shape[seq_dim]
        if not self.seq_len_cached or seq_len > self.seq_len_cached:
            self.seq_len_cached = 2500
            t = torch.arange(self.seq_len_cached, device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=len(x.shape) - 1)


def rope_rotate(x, positions, cos, sin):
    """Apply rotary position embedding to input tensor."""
    return x * cos[:, positions] + rotate_half(x) * sin[:, positions]


class QueryHead(nn.Linear):
    """Marker class for query head in attention mechanism."""

    pass


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention implementation with optional RoPE and cross-attention support.

    Args:
        n_state (int): Hidden state dimension
        n_head (int): Number of attention heads
        qk_scale (float, optional): Query-Key scaling factor. Defaults to 1.
        rope (bool, optional): Whether to use RoPE. Defaults to False.
        cross (bool, optional): Whether to use cross-attention. Defaults to False.
    """

    def __init__(
        self,
        n_state: int,
        n_head: int,
        qk_scale: float = 1,
        rope: bool = False,
        cross=False,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.sqrt_qk_scale = math.sqrt(qk_scale)
        self.query = QueryHead(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self.cross = cross
        self.query_subsampling = 1
        self.key_subsampling = 1

        self.cached_kvx = None
        self.register_buffer("k_cache", None)
        self.register_buffer("v_cache", None)

        self.rotary = None
        if rope:
            self.rotary = Rotary(n_state // n_head)
        self.qkv = None
        self.kv = None

    def setup_kv_cache(self, max_batch_size, max_seq_len, dtype=torch.float32):
        cache_shape = (
            max_batch_size,
            self.n_head,
            max_seq_len,
            self.n_state // self.n_head,
        )
        self.k_cache = torch.zeros(
            cache_shape, dtype=dtype, device=self.key.weight.device
        )
        self.v_cache = torch.zeros(
            cache_shape, dtype=dtype, device=self.value.weight.device
        )

    def merge_linears(self, layers, mults):
        bias = [x.bias for x in layers if x.bias is not None][0]
        din, dout = layers[0].weight.shape
        new = nn.Linear(din, len(layers) * dout).to(layers[0].weight.device)
        with torch.no_grad():
            new.weight[:] = torch.cat([x.weight * m for x, m in zip(layers, mults)])
            new.bias[:] = torch.cat(
                [
                    torch.zeros_like(bias) if x.bias is None else x.bias * m
                    for x, m in zip(layers, mults)
                ]
            )
        return new

    def convert_for_eval(self):
        if self.qkv or self.kv:
            raise AttributeError("already converted")

        self.odim = self.key.weight.shape[1]
        if self.cross:
            self.q = self.merge_linears([self.query], [self.sqrt_qk_scale])
            self.kv = self.merge_linears(
                [self.key, self.value], [self.sqrt_qk_scale, 1]
            )
        else:
            self.qkv = self.merge_linears(
                [self.query, self.key, self.value],
                [self.sqrt_qk_scale, self.sqrt_qk_scale, 1],
            )

    def split_heads(self, x, x_positions, rope=False, subsampling=1):
        x = x.view(*x.shape[:2], self.n_head, -1)
        if rope:
            x = rope_rotate(x, x_positions * subsampling, *self.rotary(x))
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        qx,
        q_positions,
        kvx,
        kv_positions,
        causal=False,
        mask=None,
    ):
        if self.k_cache is not None:
            assert (
                qx.shape[0] <= self.k_cache.shape[0]
            ), "please pass in a larger max_batch_size to setup_kv_cache"
        if self.qkv:
            q, k, v = self.qkv(qx).split(self.odim, dim=-1)
        elif self.kv:
            q = self.q(qx)
            k, v = self.kv(kvx).split(self.odim, dim=-1)
        else:
            q, k, v = None, None, None

        if q is None:
            q = self.query(qx) * self.sqrt_qk_scale
        q = self.split_heads(
            q, q_positions, rope=self.rotary, subsampling=self.query_subsampling
        )

        if kvx is not self.cached_kvx:
            if k is None:
                k = self.key(kvx) * self.sqrt_qk_scale
            k = self.split_heads(
                k, kv_positions, rope=self.rotary, subsampling=self.key_subsampling
            )
            if v is None:
                v = self.value(kvx)
            v = self.split_heads(v, kv_positions)
            if self.k_cache is not None:
                self.k_cache[: k.shape[0], :, kv_positions] = k
                self.v_cache[: v.shape[0], :, kv_positions] = v

        if self.k_cache is not None:
            k, v = self.k_cache[: k.shape[0]], self.v_cache[: v.shape[0]]

        if mask is not None:
            mask = mask[q_positions, : k.shape[-2]]

        wv = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0, is_causal=causal
        )

        return self.out(wv.permute(0, 2, 1, 3).flatten(start_dim=2))


class ResidualAttentionBlock(nn.Module):
    """
    Residual Attention Block combining self-attention, optional cross-attention, and feed-forward layers.

    Args:
        n_state (int): Hidden state dimension
        n_head (int): Number of attention heads
        cross_attention (bool, optional): Whether to include cross-attention. Defaults to False.
        rope (bool, optional): Whether to use RoPE. Defaults to False.
        qk_scale (float, optional): Query-Key scaling factor. Defaults to 1.
        ffn_mult (int, optional): Multiplier for FFN intermediate dimension. Defaults to 4.
    """

    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        rope: bool = False,
        qk_scale: float = 1,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, qk_scale=qk_scale, rope=rope)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(
                n_state, n_head, qk_scale=qk_scale, rope=rope, cross=True
            )
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * ffn_mult
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def setup_kv_cache(self, max_batch_size, max_seq_len, max_cross_seq_len=None):
        self.attn.setup_kv_cache(max_batch_size, max_seq_len)
        if self.cross_attn:
            self.cross_attn.setup_kv_cache(max_batch_size, max_cross_seq_len)

    def forward(
        self,
        x: Tensor,
        x_positions: Tensor = None,
        xa: Optional[Tensor] = None,
        xa_positions: Optional[Tensor] = None,
        causal=False,
        mask=None,
    ):
        lnx = self.attn_ln(x)
        x = x + self.attn(lnx, x_positions, lnx, x_positions, causal=causal, mask=mask)
        if self.cross_attn:
            lnx = self.cross_attn_ln(x)
            x = x + self.cross_attn(lnx, x_positions, xa, xa_positions)
        x = x + self.mlp(self.mlp_ln(x))
        return x
