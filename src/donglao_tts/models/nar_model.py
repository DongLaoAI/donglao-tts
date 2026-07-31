import torch
from torch import nn

from donglao_tts.models.embeddings import CodecEmbeddingTable
from donglao_tts.models.transformer_block import TransformerBlock


class NARLayerPredictor(nn.Module):
    """Non-autoregressive-over-time module predicting one target RVQ layer k (1..num_quantizers-1)
    at a time, conditioned on the AR's own per-frame hidden state (already encodes ref voice + text
    + causal context via the AR's attention -- no need to re-derive it here) plus the already-known
    lower layers (0..k-1). Bidirectional (no causal mask) over the T_tgt frames, so a single
    forward pass covers the whole target in parallel -- CPU-friendly, unlike a per-frame AR loop.
    Consuming `ar_hidden` directly (instead of a separate ref-conditioned reconstruction) is also
    what lets NAR's gradient flow back into the AR -- real joint training, not just a shared batch/
    optimizer. Uses RoPE, same as the AR model (no learned position table / hard max_seq_len)."""

    def __init__(self, codebook_size, num_quantizers, d_model,
                 n_layers, n_heads, ffn_dim, dropout, rope_theta=10000.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.codec_table = CodecEmbeddingTable(codebook_size, num_quantizers, d_model)
        self.layer_query_embed = nn.Embedding(num_quantizers, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout, rope_theta=rope_theta)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, codebook_size)  # shared across all layers (same codebook_size)

    def forward(self, ar_hidden, known_target_codec, k, target_padding_mask):
        # ar_hidden [B,T_tgt,d_model]; known_target_codec [B,T_tgt,k] (ground-truth/predicted layers 0..k-1)
        device = ar_hidden.device

        layer_ids = torch.arange(k, device=device)
        known_embed = self.codec_table.embed_frames(known_target_codec, layer_ids)  # [B,T_tgt,d]

        x = self.dropout(ar_hidden + known_embed + self.layer_query_embed.weight[k])
        positions = torch.arange(x.shape[1], device=device)

        attn_mask = (~target_padding_mask)[:, None, None, :]  # bidirectional (no causal restriction)

        for block in self.blocks:
            x, _ = block(x, attn_mask, positions, past_kv=None, use_cache=False)

        x = self.ln_f(x)
        return self.head(x)  # [B, T_tgt, codebook_size]
