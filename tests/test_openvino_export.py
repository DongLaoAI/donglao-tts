import numpy as np
import pytest
import torch

pytest.importorskip("openvino")

from donglao_tts.export.onnx_export import export_ar_prefill, export_nar_layer  # noqa: E402
from donglao_tts.export.openvino_export import (  # noqa: E402
    compile_openvino_model,
    convert_onnx_to_openvino,
)
from donglao_tts.models.ar_model import ARTransformerLM  # noqa: E402
from donglao_tts.models.nar_model import NARLayerPredictor  # noqa: E402

D_MODEL, N_LAYERS, N_HEADS, FFN_DIM = 32, 2, 4, 64
CODEBOOK_SIZE, NUM_QUANTIZERS, VOCAB_SIZE = 16, 4, 20


def test_ar_prefill_openvino_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    ar_model = ARTransformerLM(
        vocab_size=VOCAB_SIZE, codebook_size=CODEBOOK_SIZE, ref_num_quantizers=1,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    ).eval()

    onnx_path = str(tmp_path / "ar_prefill.onnx")
    xml_path = str(tmp_path / "ar_prefill.xml")
    export_ar_prefill(ar_model, onnx_path, D_MODEL)
    convert_onnx_to_openvino(onnx_path, xml_path)

    torch.manual_seed(1)
    B, L = 1, 6
    input_embeds = torch.randn(B, L, D_MODEL)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    with torch.no_grad():
        torch_logits, _, _ = ar_model(input_embeds, padding_mask=padding_mask, use_cache=True)

    compiled = compile_openvino_model(xml_path)
    result = compiled({"input_embeds": input_embeds.numpy(), "padding_mask": padding_mask.numpy()})
    ov_logits = result[compiled.output("logits")]

    assert np.allclose(torch_logits.numpy(), ov_logits, atol=1e-4)


def test_nar_layer_openvino_matches_pytorch(tmp_path):
    torch.manual_seed(2)
    nar_model = NARLayerPredictor(
        codebook_size=CODEBOOK_SIZE, num_quantizers=NUM_QUANTIZERS, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    ).eval()

    onnx_path = str(tmp_path / "nar_layer.onnx")
    xml_path = str(tmp_path / "nar_layer.xml")
    export_nar_layer(nar_model, onnx_path, D_MODEL)
    convert_onnx_to_openvino(onnx_path, xml_path)

    torch.manual_seed(3)
    B, T, k = 1, 7, 2
    ar_hidden = torch.randn(B, T, D_MODEL)
    known_target_codec = torch.randint(0, CODEBOOK_SIZE, (B, T, k))
    target_padding_mask = torch.zeros(B, T, dtype=torch.bool)
    with torch.no_grad():
        torch_logits = nar_model(ar_hidden, known_target_codec, k, target_padding_mask)

    compiled = compile_openvino_model(xml_path)
    layer_ids = np.arange(k, dtype=np.int64)
    result = compiled({
        "ar_hidden": ar_hidden.numpy(), "known_target_codec": known_target_codec.numpy(),
        "layer_ids": layer_ids, "k": np.array(k, dtype=np.int64),
        "target_padding_mask": target_padding_mask.numpy(),
    })
    ov_logits = result[compiled.output("logits")]

    assert np.allclose(torch_logits.numpy(), ov_logits, atol=1e-4)
