import torch
import torch.nn.functional as F
from torch import nn

from donglao_tts.models.rope import RotaryEmbedding, apply_rotary


class SelfAttention(nn.Module):
    """RoPE self-attention. Not inherently causal -- `attn_mask` (bool, True=keep) passed in by
    the caller decides that (causal for AR, padding-only for NAR). Optional KV-cache: pass
    `past_kv=(k,v)` to only compute the new query positions against the full (past+new) keys,
    instead of recomputing the whole sequence every step."""

    def __init__(self, d_model, n_heads, dropout, rope_theta=10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta)

    def forward(self, x, attn_mask, positions, past_kv=None, use_cache=False):
        B, L, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(positions)  # positions: [L] absolute positions of these NEW tokens
        q, k = apply_rotary(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), new_kv


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout, rope_theta=10000.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout, rope_theta=rope_theta)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, positions, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln1(x), attn_mask, positions, past_kv, use_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x, new_kv
