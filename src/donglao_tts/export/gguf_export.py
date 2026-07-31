"""EXPERIMENTAL: export the AR qwen3 backbone's weights to a GGUF file.

Read this before using: **a stock llama.cpp binary (llama-cli/llama-server) CANNOT run the
result.** GGUF/llama.cpp fundamentally assume a single input vocab (token ids -> one embedding
table) and the same vocab for output logits. This model doesn't fit that:

- Input is always `inputs_embeds` (SplitEmbedding's text_table + codec_table, concatenated by
  build_input_embeds), never token ids from a single vocab.
- The output head classifies into [0, codebook_size) + one EOS class -- a completely different,
  much smaller space than any text vocab.
- The NAR branch (bidirectional, non-autoregressive) and the MOSS RVQ codec have no equivalent in
  llama.cpp's computational model at all; they always need ONNX/PyTorch/OpenVINO regardless of
  what happens here.

What this DOES do, and why it's still useful: exports the qwen3 backbone's actual Transformer
weights (attention + FFN + norms) under the exact tensor names/metadata keys llama.cpp's own
Qwen3 architecture expects (`blk.{i}.attn_q.weight`, etc. -- taken directly from the installed
`gguf` package's `MODEL_TENSORS[MODEL_ARCH.QWEN3]`, not guessed), so the weights are at least
byte-correct and directly reusable by a custom driver that:
  1. computes input_embeds the normal way (SplitEmbedding, in PyTorch -- see onnx_generate.py's
     scoping rationale for why embeddings aren't part of any export here either),
  2. feeds them into the backbone via a Qwen3-graph-compatible low-level API that accepts
     embeddings instead of token ids (e.g. llama.cpp's C API with `llama_batch.embd` set, via
     llama-cpp-python's low-level bindings -- NOT implemented here, left as a documented next
     step since it's a substantial separate effort in its own right),
  3. applies this model's own head (exported here under a non-standard key, see below) to the
     final hidden state instead of a standard vocab-projection `output.weight`.

Our own SplitEmbedding/head tensors ARE included in the file (for bundling convenience), but
under explicitly non-standard keys (`donglao.*`), never as `token_embd.weight`/`output.weight` --
naming them as if they were a real vocab embedding/projection would be actively misleading, since
they aren't loadable into any stock llama.cpp inference path either way. No tokenizer.ggml.* KV
section is written at all: there's no meaningful single vocab to describe.
"""

import gguf


def export_ar_qwen3_gguf(ar_model, out_path, name="donglao-tts-ar-qwen3"):
    """`ar_model` must be an ARQwen3LM (model.ar.backbone: qwen3). Writes the backbone under
    standard Qwen3 GGUF tensor names + architecture metadata, and this model's own
    SplitEmbedding/head under non-standard `donglao.*` keys (see module docstring)."""
    cfg = ar_model.backbone.config
    n_layers = cfg.num_hidden_layers

    writer = gguf.GGUFWriter(out_path, "qwen3")
    writer.add_name(name)
    writer.add_description(
        "EXPERIMENTAL, NOT runnable via stock llama.cpp -- see this file's own KV metadata "
        "'donglao.runnable_via_stock_llama_cpp' (false) and the exporting code's module "
        "docstring (src/donglao_tts/export/gguf_export.py) for why."
    )
    writer.add_architecture()
    writer.add_context_length(cfg.max_position_embeddings)
    writer.add_embedding_length(cfg.hidden_size)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(cfg.intermediate_size)
    writer.add_head_count(cfg.num_attention_heads)
    writer.add_head_count_kv(cfg.num_key_value_heads)
    writer.add_key_length(cfg.head_dim)
    writer.add_value_length(cfg.head_dim)
    writer.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    writer.add_rope_freq_base(cfg.rope_parameters["rope_theta"])
    writer.add_bool("donglao.runnable_via_stock_llama_cpp", False)

    sd = {k: v.detach().cpu().float().numpy() for k, v in ar_model.state_dict().items()}

    writer.add_tensor("output_norm.weight", sd["backbone.norm.weight"])
    for i in range(n_layers):
        prefix = f"backbone.layers.{i}."
        blk = f"blk.{i}."
        writer.add_tensor(blk + "attn_norm.weight", sd[prefix + "input_layernorm.weight"])
        writer.add_tensor(blk + "attn_q.weight", sd[prefix + "self_attn.q_proj.weight"])
        writer.add_tensor(blk + "attn_q_norm.weight", sd[prefix + "self_attn.q_norm.weight"])
        writer.add_tensor(blk + "attn_k.weight", sd[prefix + "self_attn.k_proj.weight"])
        writer.add_tensor(blk + "attn_k_norm.weight", sd[prefix + "self_attn.k_norm.weight"])
        writer.add_tensor(blk + "attn_v.weight", sd[prefix + "self_attn.v_proj.weight"])
        writer.add_tensor(blk + "attn_output.weight", sd[prefix + "self_attn.o_proj.weight"])
        writer.add_tensor(blk + "ffn_norm.weight", sd[prefix + "post_attention_layernorm.weight"])
        writer.add_tensor(blk + "ffn_gate.weight", sd[prefix + "mlp.gate_proj.weight"])
        writer.add_tensor(blk + "ffn_up.weight", sd[prefix + "mlp.up_proj.weight"])
        writer.add_tensor(blk + "ffn_down.weight", sd[prefix + "mlp.down_proj.weight"])

    # Non-standard keys, deliberately -- see module docstring.
    writer.add_tensor("donglao.text_embd.weight", sd["embed.text_table.weight"])
    writer.add_tensor("donglao.codec_embd.weight", sd["embed.codec_table.audio_embed.weight"])
    writer.add_tensor("donglao.output_head.weight", sd["head.weight"])
    writer.add_tensor("donglao.output_head.bias", sd["head.bias"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return out_path
