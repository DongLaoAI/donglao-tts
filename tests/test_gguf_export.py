import numpy as np
import pytest
import torch

pytest.importorskip("gguf")
import gguf  # noqa: E402

from donglao_tts.export.gguf_export import export_ar_qwen3_gguf  # noqa: E402
from donglao_tts.models.ar_qwen3 import ARQwen3LM  # noqa: E402

D_MODEL, N_LAYERS, N_HEADS, N_KV_HEADS, FFN_DIM = 32, 2, 4, 2, 64
CODEBOOK_SIZE, VOCAB_SIZE = 16, 20


def test_export_ar_qwen3_gguf_writes_correct_tensors_and_metadata(tmp_path):
    torch.manual_seed(0)
    ar_model = ARQwen3LM(
        vocab_size=VOCAB_SIZE, codebook_size=CODEBOOK_SIZE, ref_num_quantizers=1,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
        n_kv_heads=N_KV_HEADS,
    ).eval()

    out_path = str(tmp_path / "ar_qwen3.gguf")
    export_ar_qwen3_gguf(ar_model, out_path)

    reader = gguf.GGUFReader(out_path)

    fields = reader.fields
    assert fields["general.architecture"].parts[-1].tobytes().decode("utf-8") == "qwen3"
    assert int(fields["qwen3.block_count"].parts[-1][0]) == N_LAYERS
    assert int(fields["qwen3.embedding_length"].parts[-1][0]) == D_MODEL
    assert int(fields["qwen3.attention.head_count"].parts[-1][0]) == N_HEADS
    assert int(fields["qwen3.attention.head_count_kv"].parts[-1][0]) == N_KV_HEADS
    assert bool(fields["donglao.runnable_via_stock_llama_cpp"].parts[-1][0]) is False

    tensor_names = {t.name for t in reader.tensors}
    expected_per_layer = ["attn_norm", "attn_q", "attn_q_norm", "attn_k", "attn_k_norm",
                           "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"]
    for i in range(N_LAYERS):
        for name in expected_per_layer:
            assert f"blk.{i}.{name}.weight" in tensor_names
    assert "output_norm.weight" in tensor_names
    assert "donglao.text_embd.weight" in tensor_names
    assert "donglao.codec_embd.weight" in tensor_names
    assert "donglao.output_head.weight" in tensor_names
    assert "donglao.output_head.bias" in tensor_names
    # deliberately absent -- see module docstring (no real single-vocab embedding/output exists)
    assert "token_embd.weight" not in tensor_names
    assert "output.weight" not in tensor_names

    sd = ar_model.state_dict()
    by_name = {t.name: t for t in reader.tensors}

    def as_np(tensor_name):
        return by_name[tensor_name].data.reshape(by_name[tensor_name].shape[::-1])

    assert np.allclose(as_np("blk.0.attn_q.weight"),
                        sd["backbone.layers.0.self_attn.q_proj.weight"].numpy(), atol=1e-5)
    assert np.allclose(as_np("output_norm.weight"), sd["backbone.norm.weight"].numpy(), atol=1e-5)
    assert np.allclose(as_np("donglao.output_head.weight"), sd["head.weight"].numpy(), atol=1e-5)
    assert np.allclose(as_np("donglao.text_embd.weight"),
                        sd["embed.text_table.weight"].numpy(), atol=1e-5)
