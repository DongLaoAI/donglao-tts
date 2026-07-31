import numpy as np
import onnxruntime
import torch
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch import nn

from donglao_tts.export.onnx_export import export_nar_layer
from donglao_tts.models.ar_model import ARTransformerLM
from donglao_tts.models.nar_model import NARLayerPredictor
from donglao_tts.quantization import extract_plain_state_dict, prepare_model_qat

D_MODEL, N_LAYERS, N_HEADS, FFN_DIM = 32, 2, 4, 64
CODEBOOK_SIZE, NUM_QUANTIZERS, VOCAB_SIZE = 16, 4, 20


def _count_linear_types(model):
    n_qat, n_plain = 0, 0
    for m in model.modules():
        if isinstance(m, nnqat.Linear):
            n_qat += 1
        elif isinstance(m, nn.Linear):
            n_plain += 1
    return n_qat, n_plain


def test_prepare_model_qat_swaps_linear_layers():
    torch.manual_seed(0)
    ar_model = ARTransformerLM(
        vocab_size=VOCAB_SIZE, codebook_size=CODEBOOK_SIZE, ref_num_quantizers=1,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    )
    n_qat_before, n_plain_before = _count_linear_types(ar_model)
    assert n_qat_before == 0
    assert n_plain_before > 0  # qkv_proj, out_proj, mlp x2, head, per layer

    prepare_model_qat(ar_model)
    n_qat_after, n_plain_after = _count_linear_types(ar_model)
    assert n_plain_after == 0
    assert n_qat_after == n_plain_before  # every Linear got swapped, none dropped/duplicated


def test_qat_ar_model_trains_end_to_end():
    """Forward + backward through a QAT-prepared model must work unmodified (fake-quant is a
    straight-through estimator, differentiable) -- this is the actual "training-loop usage"
    contract, not just a module-swap count."""
    torch.manual_seed(1)
    ar_model = ARTransformerLM(
        vocab_size=VOCAB_SIZE, codebook_size=CODEBOOK_SIZE, ref_num_quantizers=1,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    )
    prepare_model_qat(ar_model)
    ar_model.train()

    B, L = 2, 6
    input_embeds = torch.randn(B, L, D_MODEL)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    labels = torch.randint(0, CODEBOOK_SIZE, (B, L))

    logits, _, _ = ar_model(input_embeds, padding_mask=padding_mask)
    loss = F.cross_entropy(logits.transpose(1, 2), labels)
    loss.backward()

    grad_found = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in ar_model.parameters() if p.requires_grad)
    assert grad_found


def test_extract_plain_state_dict_then_export_onnx(tmp_path):
    """The actual supported deployment path (see quantization.py's module docstring): QAT-train,
    extract plain-compatible weights into a fresh un-wrapped model, export that to ONNX exactly
    like a normally-trained model -- verifying the full loop works end to end, not just the
    state_dict projection in isolation."""
    torch.manual_seed(2)
    nar_model = NARLayerPredictor(
        codebook_size=CODEBOOK_SIZE, num_quantizers=NUM_QUANTIZERS, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    )
    prepare_model_qat(nar_model)
    nar_model.train()

    B, T, k = 2, 5, 2
    for _ in range(3):
        ar_hidden = torch.randn(B, T, D_MODEL)
        known = torch.randint(0, CODEBOOK_SIZE, (B, T, k))
        pad_mask = torch.zeros(B, T, dtype=torch.bool)
        logits = nar_model(ar_hidden, known, k, pad_mask)
        logits.sum().backward()

    plain_model = NARLayerPredictor(
        codebook_size=CODEBOOK_SIZE, num_quantizers=NUM_QUANTIZERS, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.0,
    )
    extract_plain_state_dict(nar_model, plain_model)
    plain_model.eval()

    onnx_path = str(tmp_path / "nar_layer.onnx")
    export_nar_layer(plain_model, onnx_path, D_MODEL)

    with torch.no_grad():
        torch_logits = plain_model(ar_hidden, known, k, pad_mask)

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    layer_ids = np.arange(k, dtype=np.int64)
    (onnx_logits,) = session.run(None, {
        "ar_hidden": ar_hidden.numpy(), "known_target_codec": known.numpy(),
        "layer_ids": layer_ids, "k": np.array(k, dtype=np.int64),
        "target_padding_mask": pad_mask.numpy(),
    })
    assert np.allclose(torch_logits.numpy(), onnx_logits, atol=1e-4)
