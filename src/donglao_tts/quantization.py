"""Quantization-Aware Training (QAT) support.

Uses PyTorch's eager-mode `torch.ao.quantization` API (module-swap based on a `.qconfig`
attribute set per-module), not FX graph-mode quantization -- eager mode's module-swap is a purely
structural `named_modules()` walk, independent of `forward()`'s control flow, so it works
unmodified on the AR/NAR models' custom container classes (TransformerBlock, SelfAttention, etc.)
without needing to touch their forward() methods. Only `nn.Linear` layers are wrapped (the vast
majority of parameters); `nn.Embedding`/`nn.LayerNorm` are excluded (see prepare_model_qat's
docstring) and left at full precision, matching the same scoping as the PTQ path (see quantize.py).

Training-loop usage (see train.qat/train.qat_start_step in configs/base.yaml): call
`prepare_model_qat` on ar_model/nar_model right after `build_models(...)`, then train exactly as
before -- the fake-quant modules are differentiable (straight-through estimator), so
loss.backward() needs no changes.

Deployment path after QAT training: `extract_plain_state_dict` (NOT PyTorch's own eager-mode
`convert()`-to-native-int8 -- confirmed that raises `Could not run 'quantized::linear' ... CPU
backend` here, because it needs explicit QuantStub/DeQuantStub markers inserted into forward() to
know where float/int8 tensors hand off, which would mean invasively modifying every model file).
Instead: load the QAT model's plain-compatible weights (confirmed a strict superset match, no
missing/unexpected keys) back into a fresh, un-wrapped model instance, then export that to ONNX
(see export/onnx_export.py) and apply PTQ (see quantize.py) as usual -- the exported int8 graph
should retain accuracy better than PTQ alone, since the weights were trained to tolerate
quantization noise.
"""

from torch import nn
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat


def prepare_model_qat(model, backend="fbgemm"):
    """Wraps `model`'s nn.Linear layers with fake-quant observers for QAT, in place. `backend`
    picks the target int8 kernel set ('fbgemm' for x86 server CPUs, 'qnnpack' for ARM/mobile) --
    only matters if you later go through PyTorch's own convert()/quantized inference, which this
    module's supported path (extract_plain_state_dict + ONNX PTQ) does not.

    Setting `.qconfig` on the top-level model cascades to every submodule, including
    nn.Embedding and nn.LayerNorm -- but embedding quantization needs a different qconfig
    (float_qparams_weight_only_qconfig, not the Linear-oriented default one), and LayerNorm has
    no quantized CPU kernel wired up in this eager-mode flow without explicit QuantStub/
    DeQuantStub boundaries. Explicitly opt both out (`.qconfig = None`) to keep them full
    precision, matching quantize.py's PTQ scoping (Gather ops untouched there too) -- only
    nn.Linear ends up wrapped, which is where the vast majority of parameters live anyway."""
    model.qconfig = get_default_qat_qconfig(backend)
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.LayerNorm)):
            m.qconfig = None
    return prepare_qat(model, inplace=True)


def extract_plain_state_dict(qat_model, plain_model):
    """After QAT training, project `qat_model`'s state_dict down to just the keys a fresh,
    un-wrapped `plain_model` (same architecture, not QAT-prepared) actually has -- dropping every
    `*_fake_quant.*`/`*.activation_post_process.*` observer buffer prepare_qat added -- and load
    it into `plain_model` in place. Confirmed these are a strict superset (no missing/unexpected
    keys on load): the underlying nn.Linear weight/bias tensors are untouched by fake-quant
    (which only observes/simulates in forward(), never rewrites the stored parameter), so this is
    a lossless extraction of the QAT-trained weights, ready for normal ONNX export."""
    plain_keys = set(plain_model.state_dict().keys())
    filtered = {k: v for k, v in qat_model.state_dict().items() if k in plain_keys}
    plain_model.load_state_dict(filtered)
    return plain_model
