import os

import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

from donglao_tts.export.onnx_export import (  # noqa: E402
    export_ar_decode_step,
    export_ar_prefill,
    export_ar_qwen3_decode_step,
    export_ar_qwen3_prefill,
    export_nar_layer,
)
from donglao_tts.models.ar_model import ARTransformerLM  # noqa: E402
from donglao_tts.models.ar_qwen3 import ARQwen3LM  # noqa: E402
from donglao_tts.models.nar_model import NARLayerPredictor  # noqa: E402

D_MODEL, N_LAYERS, N_HEADS, FFN_DIM = 32, 2, 4, 64
CODEBOOK_SIZE, NUM_QUANTIZERS, VOCAB_SIZE = 16, 4, 20


def _make_ar_model():
    torch.manual_seed(0)
    return ARTransformerLM(
        vocab_size=VOCAB_SIZE, codebook_size=CODEBOOK_SIZE, ref_num_quantizers=1,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    ).eval()


def _make_nar_model():
    torch.manual_seed(1)
    return NARLayerPredictor(
        codebook_size=CODEBOOK_SIZE, num_quantizers=NUM_QUANTIZERS, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    ).eval()


def _make_qwen3_model():
    torch.manual_seed(5)
    return ARQwen3LM(
        vocab_size=VOCAB_SIZE, codebook_size=CODEBOOK_SIZE, ref_num_quantizers=1,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
        n_kv_heads=2,
    ).eval()


def test_ar_prefill_onnx_matches_pytorch(tmp_path):
    ar_model = _make_ar_model()
    onnx_path = str(tmp_path / "ar_prefill.onnx")
    export_ar_prefill(ar_model, onnx_path, D_MODEL)

    torch.manual_seed(2)
    B, L = 1, 6
    input_embeds = torch.randn(B, L, D_MODEL)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)

    with torch.no_grad():
        torch_logits, _, torch_caches = ar_model(input_embeds, padding_mask=padding_mask,
                                                  use_cache=True)

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits, onnx_keys, onnx_values = session.run(
        None, {"input_embeds": input_embeds.numpy(), "padding_mask": padding_mask.numpy()})

    assert np.allclose(torch_logits.numpy(), onnx_logits, atol=1e-4)
    torch_keys = torch.stack([kv[0] for kv in torch_caches], dim=0).numpy()
    torch_values = torch.stack([kv[1] for kv in torch_caches], dim=0).numpy()
    assert np.allclose(torch_keys, onnx_keys, atol=1e-4)
    assert np.allclose(torch_values, onnx_values, atol=1e-4)


def test_ar_decode_step_onnx_matches_pytorch(tmp_path):
    ar_model = _make_ar_model()
    prefill_path = str(tmp_path / "ar_prefill.onnx")
    decode_path = str(tmp_path / "ar_decode_step.onnx")
    export_ar_prefill(ar_model, prefill_path, D_MODEL)
    export_ar_decode_step(ar_model, decode_path, D_MODEL)

    torch.manual_seed(3)
    B, L = 1, 5
    input_embeds = torch.randn(B, L, D_MODEL)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)

    with torch.no_grad():
        _, _, torch_caches = ar_model(input_embeds, padding_mask=padding_mask, use_cache=True)
        next_embed = torch.randn(B, 1, D_MODEL)
        torch_logits, torch_hidden, torch_new_caches = ar_model(
            next_embed, past_key_values=torch_caches, use_cache=True
        )

    prefill_session = onnxruntime.InferenceSession(prefill_path, providers=["CPUExecutionProvider"])
    _, onnx_keys, onnx_values = prefill_session.run(
        None, {"input_embeds": input_embeds.numpy(), "padding_mask": padding_mask.numpy()})

    decode_session = onnxruntime.InferenceSession(decode_path, providers=["CPUExecutionProvider"])
    onnx_logits, onnx_hidden, onnx_new_keys, onnx_new_values = decode_session.run(
        None, {"input_embeds": next_embed.numpy(), "past_keys": onnx_keys, "past_values": onnx_values})

    assert np.allclose(torch_logits.numpy(), onnx_logits, atol=1e-4)
    assert np.allclose(torch_hidden.numpy(), onnx_hidden, atol=1e-4)
    torch_new_keys = torch.stack([kv[0] for kv in torch_new_caches], dim=0).numpy()
    torch_new_values = torch.stack([kv[1] for kv in torch_new_caches], dim=0).numpy()
    assert np.allclose(torch_new_keys, onnx_new_keys, atol=1e-4)
    assert np.allclose(torch_new_values, onnx_new_values, atol=1e-4)


def test_nar_layer_onnx_matches_pytorch(tmp_path):
    nar_model = _make_nar_model()
    onnx_path = str(tmp_path / "nar_layer.onnx")
    export_nar_layer(nar_model, onnx_path, D_MODEL)

    torch.manual_seed(4)
    B, T, k = 1, 7, 2
    ar_hidden = torch.randn(B, T, D_MODEL)
    known_target_codec = torch.randint(0, CODEBOOK_SIZE, (B, T, k))
    target_padding_mask = torch.zeros(B, T, dtype=torch.bool)

    with torch.no_grad():
        torch_logits = nar_model(ar_hidden, known_target_codec, k, target_padding_mask)

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    layer_ids = np.arange(k, dtype=np.int64)
    (onnx_logits,) = session.run(None, {
        "ar_hidden": ar_hidden.numpy(), "known_target_codec": known_target_codec.numpy(),
        "layer_ids": layer_ids, "k": np.array(k, dtype=np.int64),
        "target_padding_mask": target_padding_mask.numpy(),
    })

    assert np.allclose(torch_logits.numpy(), onnx_logits, atol=1e-4)


def test_ar_qwen3_prefill_onnx_exports_and_runs(tmp_path):
    ar_model = _make_qwen3_model()
    onnx_path = str(tmp_path / "ar_qwen3_prefill.onnx")
    export_ar_qwen3_prefill(ar_model, onnx_path, D_MODEL)
    assert os.path.exists(onnx_path)

    B, L = 1, 6
    input_embeds = torch.randn(B, L, D_MODEL)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits, onnx_keys, onnx_values = session.run(
        None, {"input_embeds": input_embeds.numpy(), "padding_mask": padding_mask.numpy()})
    assert np.isfinite(onnx_logits).all()

    with torch.no_grad():
        torch_logits, _, torch_cache = ar_model(
            input_embeds, padding_mask=padding_mask, use_cache=True
        )
    torch_keys = torch.stack([layer.keys for layer in torch_cache.layers]).numpy()
    torch_values = torch.stack([layer.values for layer in torch_cache.layers]).numpy()
    assert np.allclose(torch_logits.numpy(), onnx_logits, atol=1e-4)
    assert np.allclose(torch_keys, onnx_keys, atol=1e-4)
    assert np.allclose(torch_values, onnx_values, atol=1e-4)


def test_ar_qwen3_decode_step_onnx_matches_pytorch_for_multiple_steps(tmp_path):
    ar_model = _make_qwen3_model()
    prefill_path = str(tmp_path / "ar_qwen3_prefill.onnx")
    decode_path = str(tmp_path / "ar_qwen3_decode_step.onnx")
    export_ar_qwen3_prefill(ar_model, prefill_path, D_MODEL)
    export_ar_qwen3_decode_step(ar_model, decode_path, D_MODEL)

    torch.manual_seed(6)
    B, L = 1, 5
    input_embeds = torch.randn(B, L, D_MODEL)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    with torch.no_grad():
        _, _, torch_cache = ar_model(input_embeds, padding_mask=padding_mask, use_cache=True)

    prefill_session = onnxruntime.InferenceSession(prefill_path, providers=["CPUExecutionProvider"])
    _, onnx_keys, onnx_values = prefill_session.run(
        None, {"input_embeds": input_embeds.numpy(), "padding_mask": padding_mask.numpy()}
    )
    decode_session = onnxruntime.InferenceSession(decode_path, providers=["CPUExecutionProvider"])

    for step in range(3):
        next_embed = torch.randn(B, 1, D_MODEL)
        position_ids = torch.tensor([[L + step]], dtype=torch.long)
        with torch.no_grad():
            torch_logits, torch_hidden, torch_cache = ar_model(
                next_embed, past_key_values=torch_cache, use_cache=True
            )

        onnx_logits, onnx_hidden, onnx_keys, onnx_values = decode_session.run(None, {
            "input_embeds": next_embed.numpy(),
            "position_ids": position_ids.numpy(),
            "past_keys": onnx_keys,
            "past_values": onnx_values,
        })
        torch_keys = torch.stack([layer.keys for layer in torch_cache.layers]).numpy()
        torch_values = torch.stack([layer.values for layer in torch_cache.layers]).numpy()
        assert np.allclose(torch_logits.numpy(), onnx_logits, atol=1e-4)
        assert np.allclose(torch_hidden.numpy(), onnx_hidden, atol=1e-4)
        assert np.allclose(torch_keys, onnx_keys, atol=1e-4)
        assert np.allclose(torch_values, onnx_values, atol=1e-4)
