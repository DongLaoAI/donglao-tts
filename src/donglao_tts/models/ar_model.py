import torch
from torch import nn

from donglao_tts.models.embeddings import SplitEmbedding
from donglao_tts.models.transformer_block import TransformerBlock


class ARTransformerLM(nn.Module):
    """Single-stream causal decoder-only Transformer. Input side embeds text/special ids and
    ref-codec ids through two separate tables (`embed`, see SplitEmbedding), but the output
    head only ever classifies into the AR's own small codec-generation space: codec ids
    [0, codebook_size) plus one dedicated EOS class (index == codebook_size). This is deliberately
    decoupled from the text/special SentencePiece vocab -- the AR never predicts a text token, so
    there's no reason for its classification space to include one. Input embedding sharing and
    output classification space are independent design choices (see build_input_embeds).

    `ref_num_quantizers` (<= the codec's true num_quantizers) is how many RVQ layers of ref_codec
    get embedded (summed into one vector per frame -- see build_input_embeds); it sizes `embed`'s
    codec sub-range, decoupled from the codec's real depth since target_codec_in only ever uses
    layer 0 anyway -- no reason to reserve embedding rows for ref layers we never look at.
    Uses RoPE (no learned position table / hard max_seq_len)."""

    def __init__(self, vocab_size, codebook_size, ref_num_quantizers, d_model,
                 n_layers, n_heads, ffn_dim, dropout, rope_theta=10000.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size

        self.embed = SplitEmbedding(vocab_size, codebook_size, ref_num_quantizers, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout, rope_theta=rope_theta)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, codebook_size + 1)  # codec ids + 1 dedicated EOS class

    def forward(self, input_embeds, padding_mask=None, past_key_values=None, use_cache=False):
        """`padding_mask` (True=pad) is only supported for a fresh (non-cached) full-sequence
        forward, i.e. training -- incremental/cached decoding is always a single, unpadded
        sequence (batch=1 generation), so no padding mask is needed there.

        Returns (logits, hidden, new_key_values). `hidden` (post ln_f, pre-head) is exposed so the
        NAR can condition directly on the AR's own representation instead of recomputing context
        from scratch -- this is also what lets NAR gradients flow back into the AR (real joint
        training), not just a shared optimizer/batch."""
        B, L, _ = input_embeds.shape
        device = input_embeds.device
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        x = self.dropout(input_embeds)
        positions = torch.arange(past_len, past_len + L, device=device)

        idx_i = torch.arange(L, device=device).unsqueeze(1)
        idx_j = torch.arange(past_len + L, device=device).unsqueeze(0)
        causal_allowed = (idx_j <= (past_len + idx_i))  # [L, past_len+L] bool, True = keep

        if padding_mask is not None:
            assert past_key_values is None, "padding_mask is not supported with KV-cache decoding"
            attn_mask = causal_allowed[None, None] & (~padding_mask)[:, None, None, :]
        else:
            attn_mask = causal_allowed[None, None]

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = block(x, attn_mask, positions, past_kv, use_cache)
            if use_cache:
                new_caches.append(new_kv)

        x = self.ln_f(x)
        return self.head(x), x, new_caches
