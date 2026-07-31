import os

import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

from donglao_tts.export.onnx_export import export_ar_prefill, export_nar_layer  # noqa: E402
from donglao_tts.export.quantize import quantize_onnx_dynamic  # noqa: E402
from donglao_tts.models.ar_model import ARTransformerLM  # noqa: E402
from donglao_tts.models.nar_model import NARLayerPredictor  # noqa: E402

D_MODEL, N_LAYERS, N_HEADS, FFN_DIM = 32, 2, 4, 64
CODEBOOK_SIZE, NUM_QUANTIZERS, VOCAB_SIZE = 16, 4, 20


def test_quantize_ar_prefill_shrinks_file_and_stays_close(tmp_path):
    torch.manual_seed(0)
    ar_model = ARTransformerLM(
        vocab_size=VOCAB_SIZE, codebook_size=CODEBOOK_SIZE, ref_num_quantizers=1,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    ).eval()

    onnx_path = str(tmp_path / "ar_prefill.onnx")
    quant_path = str(tmp_path / "ar_prefill.quant.onnx")
    export_ar_prefill(ar_model, onnx_path, D_MODEL)
    quantize_onnx_dynamic(onnx_path, quant_path)

    assert os.path.exists(quant_path)
    # int8 weights should meaningfully shrink the on-disk graph (embeddings/activations stay
    # full precision, but Linear weights -- most of the parameters here -- become int8).
    assert os.path.getsize(quant_path) < os.path.getsize(onnx_path)

    torch.manual_seed(1)
    B, L = 1, 6
    input_embeds = torch.randn(B, L, D_MODEL)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    with torch.no_grad():
        torch_logits, _, _ = ar_model(input_embeds, padding_mask=padding_mask, use_cache=True)

    session = onnxruntime.InferenceSession(quant_path, providers=["CPUExecutionProvider"])
    quant_logits, _, _ = session.run(
        None, {"input_embeds": input_embeds.numpy(), "padding_mask": padding_mask.numpy()})

    assert np.isfinite(quant_logits).all()
    # int8 weight quantization is lossy -- not a numeric-parity check like ONNX/OpenVINO export,
    # just "the argmax prediction usually agrees" as a sanity bar that nothing is badly broken.
    agreement = (torch_logits.numpy().argmax(-1) == quant_logits.argmax(-1)).mean()
    assert agreement > 0.5


def test_quantize_nar_layer_runs(tmp_path):
    torch.manual_seed(2)
    nar_model = NARLayerPredictor(
        codebook_size=CODEBOOK_SIZE, num_quantizers=NUM_QUANTIZERS, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    ).eval()

    onnx_path = str(tmp_path / "nar_layer.onnx")
    quant_path = str(tmp_path / "nar_layer.quant.onnx")
    export_nar_layer(nar_model, onnx_path, D_MODEL)
    quantize_onnx_dynamic(onnx_path, quant_path)

    B, T, k = 1, 7, 2
    ar_hidden = torch.randn(B, T, D_MODEL).numpy()
    known_target_codec = torch.randint(0, CODEBOOK_SIZE, (B, T, k)).numpy()
    layer_ids = np.arange(k, dtype=np.int64)
    target_padding_mask = np.zeros((B, T), dtype=bool)

    session = onnxruntime.InferenceSession(quant_path, providers=["CPUExecutionProvider"])
    (logits,) = session.run(None, {
        "ar_hidden": ar_hidden, "known_target_codec": known_target_codec,
        "layer_ids": layer_ids, "k": np.array(k, dtype=np.int64),
        "target_padding_mask": target_padding_mask,
    })
    assert np.isfinite(logits).all()
