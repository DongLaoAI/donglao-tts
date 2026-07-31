"""Export the AR and NAR models to ONNX.

Scope, deliberately: only the Transformer forward passes are exported. Embedding lookups
(SplitEmbedding's text/codec tables, and build_input_embeds' prompt-sequence assembly) stay in
PyTorch and are NOT part of the exported graphs -- they're cheap table gathers/concatenation, not
the compute bottleneck, so folding them into ONNX buys negligible speedup for real added
complexity. See onnx_generate.py's driver for how PyTorch embeddings + ONNX Runtime Transformer
graphs are composed at inference time. (Exception: NARLayerPredictor's per-layer codec embedding
IS folded into its exported graph -- it's small/self-contained and simplifies the driver.)

Autoregressive generation itself (the Python loop with a data-dependent EOS-stop condition, see
generate.py's ar_generate_rvq0) cannot be one ONNX graph -- it's decomposed the same way HF's own
causal-LM ONNX export does it: a "prefill" graph (full prompt -> first logits + KV-cache) and a
"decode-step" graph (one new token + past KV-cache -> next logits + updated KV-cache), looped by
Python/ONNX Runtime driver code, not by the graph itself.

The custom backbone exposes a flat list of (key, value) tensor tuples. Qwen3 instead uses
HuggingFace's `DynamicCache`; the Qwen3 wrappers below convert that Python cache object to/from a
stacked tensor I/O contract so ONNX Runtime never has to understand `DynamicCache` itself.
"""

import torch
from torch import nn
from transformers import DynamicCache

ONNX_OPSET = 18


class _ARPrefillWrapper(nn.Module):
    """input_embeds [B,L,D], padding_mask [B,L] bool (True=pad) -> logits [B,L,V],
    present_keys/present_values [n_layers,B,H,L,Dh] (stacked across layers so the graph has a
    fixed, flat tensor I/O contract instead of one input/output pair per layer)."""

    def __init__(self, ar_model):
        super().__init__()
        self.ar_model = ar_model

    def forward(self, input_embeds, padding_mask):
        logits, _, new_caches = self.ar_model(input_embeds, padding_mask=padding_mask, use_cache=True)
        present_keys = torch.stack([kv[0] for kv in new_caches], dim=0)
        present_values = torch.stack([kv[1] for kv in new_caches], dim=0)
        return logits, present_keys, present_values


class _ARDecodeStepWrapper(nn.Module):
    """input_embeds [B,1,D] (exactly one new frame), past_keys/past_values [n_layers,B,H,past_len,Dh]
    -> logits [B,1,V], present_keys/present_values [n_layers,B,H,past_len+1,Dh]."""

    def __init__(self, ar_model):
        super().__init__()
        self.ar_model = ar_model

    def forward(self, input_embeds, past_keys, past_values):
        past_key_values = [(past_keys[i], past_values[i]) for i in range(past_keys.shape[0])]
        logits, hidden, new_caches = self.ar_model(input_embeds, past_key_values=past_key_values,
                                                    use_cache=True)
        present_keys = torch.stack([kv[0] for kv in new_caches], dim=0)
        present_values = torch.stack([kv[1] for kv in new_caches], dim=0)
        return logits, hidden, present_keys, present_values


class _NARLayerWrapper(nn.Module):
    """Re-expresses NARLayerPredictor.forward with `layer_ids` as an explicit input tensor
    instead of computed via `torch.arange(k)` inside the graph -- `k` is a runtime scalar value
    (not a shape), and ONNX/torch.onnx.export handle a precomputed-by-the-caller index tensor far
    more reliably than data-dependent-length arange. The driver (onnx_generate.py) passes
    `layer_ids = np.arange(k)` alongside `k` itself (still needed for the `layer_query_embed`
    row lookup, a plain Gather)."""

    def __init__(self, nar_model):
        super().__init__()
        self.nar_model = nar_model

    def forward(self, ar_hidden, known_target_codec, layer_ids, k, target_padding_mask):
        nar = self.nar_model
        known_embed = nar.codec_table.embed_frames(known_target_codec, layer_ids)
        x = nar.dropout(ar_hidden + known_embed + nar.layer_query_embed.weight[k])
        positions = torch.arange(x.shape[1], device=x.device)
        attn_mask = (~target_padding_mask)[:, None, None, :]
        for block in nar.blocks:
            x, _ = block(x, attn_mask, positions, past_kv=None, use_cache=False)
        x = nar.ln_f(x)
        return nar.head(x)


def _n_heads_head_dim(ar_model):
    first_block = ar_model.blocks[0]
    return first_block.attn.n_heads, first_block.attn.head_dim


def export_ar_prefill(ar_model, out_path, d_model, opset=ONNX_OPSET):
    """`ar_model` must be an ARTransformerLM (custom backbone) -- see module docstring."""
    ar_model = ar_model.float().eval()
    wrapper = _ARPrefillWrapper(ar_model).eval()
    B, L = 1, 8
    dummy_input_embeds = torch.zeros(B, L, d_model)
    dummy_padding_mask = torch.zeros(B, L, dtype=torch.bool)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_input_embeds, dummy_padding_mask), out_path,
            input_names=["input_embeds", "padding_mask"],
            output_names=["logits", "present_keys", "present_values"],
            dynamic_axes={
                "input_embeds": {0: "batch", 1: "seq_len"},
                "padding_mask": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
                "present_keys": {1: "batch", 3: "seq_len"},
                "present_values": {1: "batch", 3: "seq_len"},
            },
            opset_version=opset,
            dynamo=False,
        )
    return out_path


def export_ar_decode_step(ar_model, out_path, d_model, opset=ONNX_OPSET):
    """`ar_model` must be an ARTransformerLM (custom backbone) -- see module docstring."""
    ar_model = ar_model.float().eval()
    n_heads, head_dim = _n_heads_head_dim(ar_model)
    n_layers = len(ar_model.blocks)
    wrapper = _ARDecodeStepWrapper(ar_model).eval()
    B, past_len = 1, 4
    dummy_input_embeds = torch.zeros(B, 1, d_model)
    dummy_past_keys = torch.zeros(n_layers, B, n_heads, past_len, head_dim)
    dummy_past_values = torch.zeros(n_layers, B, n_heads, past_len, head_dim)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_input_embeds, dummy_past_keys, dummy_past_values), out_path,
            input_names=["input_embeds", "past_keys", "past_values"],
            output_names=["logits", "hidden", "present_keys", "present_values"],
            dynamic_axes={
                "input_embeds": {0: "batch"},
                "past_keys": {1: "batch", 3: "past_len"},
                "past_values": {1: "batch", 3: "past_len"},
                "logits": {0: "batch"},
                "hidden": {0: "batch"},
                "present_keys": {1: "batch", 3: "total_len"},
                "present_values": {1: "batch", 3: "total_len"},
            },
            opset_version=opset,
            dynamo=False,
        )
    return out_path


def export_nar_layer(nar_model, out_path, d_model, opset=ONNX_OPSET):
    nar_model = nar_model.float().eval()
    wrapper = _NARLayerWrapper(nar_model).eval()
    B, T, k = 1, 6, 2
    dummy_ar_hidden = torch.zeros(B, T, d_model)
    dummy_known = torch.zeros(B, T, k, dtype=torch.long)
    dummy_layer_ids = torch.arange(k, dtype=torch.long)
    dummy_k = torch.tensor(k, dtype=torch.long)
    dummy_pad_mask = torch.zeros(B, T, dtype=torch.bool)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_ar_hidden, dummy_known, dummy_layer_ids, dummy_k, dummy_pad_mask),
            out_path,
            input_names=["ar_hidden", "known_target_codec", "layer_ids", "k", "target_padding_mask"],
            output_names=["logits"],
            dynamic_axes={
                "ar_hidden": {0: "batch", 1: "frames"},
                "known_target_codec": {0: "batch", 1: "frames", 2: "k"},
                "layer_ids": {0: "k"},
                "target_padding_mask": {0: "batch", 1: "frames"},
                "logits": {0: "batch", 1: "frames"},
            },
            opset_version=opset,
            dynamo=False,
        )
    return out_path


class _ARQwen3PrefillWrapper(nn.Module):
    """Like _ARPrefillWrapper, but for ARQwen3LM: `new_caches` there is an HF `Cache`
    (`DynamicCache`) object, not a list of (key, value) tuples -- verified against
    transformers==5.13.1's actual `DynamicCache`/`DynamicLayer`: per-layer state lives at
    `cache.layers[i].keys`/`.values`."""

    def __init__(self, ar_model):
        super().__init__()
        self.ar_model = ar_model

    def forward(self, input_embeds, padding_mask):
        logits, _, new_caches = self.ar_model(input_embeds, padding_mask=padding_mask, use_cache=True)
        present_keys = torch.stack([layer.keys for layer in new_caches.layers], dim=0)
        present_values = torch.stack([layer.values for layer in new_caches.layers], dim=0)
        return logits, present_keys, present_values


class _ARQwen3DecodeStepWrapper(nn.Module):
    """Qwen3 decode step with tensor-only ONNX I/O.

    `DynamicCache` stays an implementation detail inside the traced wrapper. `position_ids` is an
    explicit input instead of being derived from a Python-visible cache shape, keeping past_len
    truly dynamic in the exported graph.
    """

    def __init__(self, ar_model):
        super().__init__()
        self.ar_model = ar_model
        self.n_layers = ar_model.backbone.config.num_hidden_layers

    def forward(self, input_embeds, position_ids, past_keys, past_values):
        cache_data = [
            (past_keys[i], past_values[i])
            for i in range(self.n_layers)
        ]
        cache = DynamicCache(cache_data)

        # A one-token decode query has no future token to mask and generation is batch=1 without
        # padding. Passing the already-prepared mapping bypasses Transformers' Python cache-mask
        # construction, which otherwise tends to freeze the traced past length.
        outputs = self.ar_model.backbone(
            inputs_embeds=self.ar_model.dropout(input_embeds),
            attention_mask={"full_attention": None},
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        hidden = outputs.last_hidden_state
        logits = self.ar_model.head(hidden)
        new_cache = outputs.past_key_values
        present_keys = torch.stack([layer.keys for layer in new_cache.layers], dim=0)
        present_values = torch.stack([layer.values for layer in new_cache.layers], dim=0)
        return logits, hidden, present_keys, present_values


def _prepare_qwen3_for_onnx(ar_model):
    # TorchScript's ONNX symbolic does not implement scaled_dot_product_attention with
    # enable_gqa=True. Transformers' eager Qwen3 attention is mathematically equivalent but
    # expands KV heads explicitly and lowers to ordinary ONNX MatMul/Softmax tensor operations.
    ar_model.backbone.config._attn_implementation = "eager"
    return ar_model.float().eval()


def export_ar_qwen3_prefill(ar_model, out_path, d_model, opset=ONNX_OPSET):
    """Export Qwen3 full-prompt prefill to logits and a flat KV cache."""
    ar_model = _prepare_qwen3_for_onnx(ar_model)
    wrapper = _ARQwen3PrefillWrapper(ar_model).eval()
    B, L = 1, 8
    dummy_input_embeds = torch.zeros(B, L, d_model)
    dummy_padding_mask = torch.zeros(B, L, dtype=torch.bool)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_input_embeds, dummy_padding_mask), out_path,
            input_names=["input_embeds", "padding_mask"],
            output_names=["logits", "present_keys", "present_values"],
            dynamic_axes={
                "input_embeds": {0: "batch", 1: "seq_len"},
                "padding_mask": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
                "present_keys": {1: "batch", 3: "seq_len"},
                "present_values": {1: "batch", 3: "seq_len"},
            },
            opset_version=opset,
            dynamo=False,
        )
    return out_path


def export_ar_qwen3_decode_step(ar_model, out_path, d_model, opset=ONNX_OPSET):
    """Export one cached Qwen3 AR step with a dynamic past sequence length."""
    ar_model = _prepare_qwen3_for_onnx(ar_model)
    cfg = ar_model.backbone.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    wrapper = _ARQwen3DecodeStepWrapper(ar_model).eval()

    B, past_len = 1, 4
    dummy_input_embeds = torch.zeros(B, 1, d_model)
    dummy_position_ids = torch.tensor([[past_len]], dtype=torch.long)
    dummy_past_keys = torch.zeros(n_layers, B, n_kv_heads, past_len, head_dim)
    dummy_past_values = torch.zeros(n_layers, B, n_kv_heads, past_len, head_dim)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_embeds, dummy_position_ids, dummy_past_keys, dummy_past_values),
            out_path,
            input_names=["input_embeds", "position_ids", "past_keys", "past_values"],
            output_names=["logits", "hidden", "present_keys", "present_values"],
            dynamic_axes={
                "input_embeds": {0: "batch"},
                "position_ids": {0: "batch"},
                "past_keys": {1: "batch", 3: "past_len"},
                "past_values": {1: "batch", 3: "past_len"},
                "logits": {0: "batch"},
                "hidden": {0: "batch"},
                "present_keys": {1: "batch", 3: "total_len"},
                "present_values": {1: "batch", 3: "total_len"},
            },
            opset_version=opset,
            dynamo=False,
        )
    return out_path
