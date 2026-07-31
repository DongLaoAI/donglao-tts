import warnings

import pytest
import torch

from donglao_tts.utils.precision import resolve_dtype


def test_fp16_on_cpu_falls_back_to_float32():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dtype = resolve_dtype("fp16", torch.device("cpu"))
        assert dtype == torch.float32
        assert any("fp16" in str(warning.message) for warning in w)


def test_float32_on_cpu():
    assert resolve_dtype("float32", torch.device("cpu")) == torch.float32


def test_bf16_on_cpu():
    assert resolve_dtype("bf16", torch.device("cpu")) == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_on_cuda():
    assert resolve_dtype("bf16", torch.device("cuda")) == torch.bfloat16


def test_unknown_precision_raises():
    with pytest.raises(ValueError):
        resolve_dtype("int8", torch.device("cpu"))
