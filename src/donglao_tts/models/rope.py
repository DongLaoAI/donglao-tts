import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """GPT-NeoX/LLaMA-style RoPE: rotates each (even/odd-half) pair of a head's dims by an angle
    proportional to absolute position, so the attention dot product between a rotated q at
    position i and a rotated k at position j only depends on (i-j) -- relative position is
    encoded directly in attention, no learned position table needed (and no hard max_seq_len)."""

    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions):
        # positions: [L] long (absolute positions) -> cos, sin: [L, head_dim]
        freqs = positions.float()[:, None] * self.inv_freq[None, :].to(positions.device)  # [L, Dh/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [L, Dh]
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    # q, k: [B, H, L, Dh]; cos, sin: [L, Dh]
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
