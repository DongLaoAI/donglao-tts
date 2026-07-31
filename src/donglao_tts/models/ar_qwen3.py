import torch
from torch import nn
from transformers import Qwen3Config, Qwen3Model

from donglao_tts.models.embeddings import SplitEmbedding


class ARQwen3LM(nn.Module):
    """Drop-in alternative to ARTransformerLM (see ar_model.py) using HuggingFace's Qwen3Model
    (RMSNorm, SwiGLU, GQA, QK-Norm, RoPE) as the backbone -- randomly initialized and trained from
    scratch, no pretrained weights loaded. Built to A/B test whether the missing-EOS behavior seen
    with the custom TransformerBlock backbone is architecture-specific, independent of the
    loss/data-pipeline factors already addressed elsewhere (per-sample loss averaging, dedicated
    EOS class decoupled from the text vocab).

    Same external contract as ARTransformerLM: forward(input_embeds, padding_mask,
    past_key_values, use_cache) -> (logits, hidden, new_caches) -- callers (train.py/generate.py)
    don't need to know which backbone is in use. We never feed `input_ids` to the backbone --
    text/codec ids are embedded by our own `embed` tables (identical to the custom backbone,
    see SplitEmbedding) and passed in as `inputs_embeds`, so Qwen3Model's own token embedding
    table is never indexed and sized to a dummy minimum.

    `ref_num_quantizers` (<= the codec's true num_quantizers) is how many RVQ layers of ref_codec
    get embedded -- see ARTransformerLM for why this is decoupled from the codec's real depth."""

    def __init__(self, vocab_size, codebook_size, ref_num_quantizers, d_model,
                 n_layers, n_heads, ffn_dim, dropout, rope_theta=10000.0,
                 n_kv_heads=None, head_dim=None, max_position_embeddings=4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.embed = SplitEmbedding(vocab_size, codebook_size, ref_num_quantizers, d_model)
        self.dropout = nn.Dropout(dropout)

        qwen_cfg = Qwen3Config(
            vocab_size=8,  # dummy/unused: we always call with inputs_embeds, never input_ids
            hidden_size=d_model,
            intermediate_size=ffn_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads or n_heads,
            head_dim=head_dim or (d_model // n_heads),
            hidden_act="silu",
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=1e-6,
            attention_dropout=dropout,
            rope_parameters={"rope_theta": rope_theta},
            use_cache=True,
            tie_word_embeddings=False,
            attention_bias=False,
        )
        self.backbone = Qwen3Model(qwen_cfg)
        self.head = nn.Linear(d_model, codebook_size + 1)  # codec ids + 1 dedicated EOS class

    def forward(self, input_embeds, padding_mask=None, past_key_values=None, use_cache=False):
        """`padding_mask` (True=pad) is only supported for a fresh (non-cached) full-sequence
        forward, i.e. training -- incremental/cached decoding is always a single, unpadded
        sequence (batch=1 generation), matching ARTransformerLM's contract exactly."""
        B, L, _ = input_embeds.shape
        device = input_embeds.device
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        attention_mask = (~padding_mask).long() if padding_mask is not None else None
        position_ids = torch.arange(past_len, past_len + L, device=device).unsqueeze(0).expand(B, -1)

        outputs = self.backbone(
            inputs_embeds=self.dropout(input_embeds),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden = outputs.last_hidden_state
        new_caches = outputs.past_key_values if use_cache else None
        return self.head(hidden), hidden, new_caches
