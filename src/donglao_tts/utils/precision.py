import warnings

import torch

_DTYPES = {
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def resolve_dtype(name, device):
    name = name.lower()
    if name not in _DTYPES:
        raise ValueError(f"unknown precision {name!r}, expected one of {list(_DTYPES)}")
    if name == "fp16" and torch.device(device).type == "cpu":
        warnings.warn("fp16 has poor CPU kernel support; falling back to float32")
        return torch.float32
    return _DTYPES[name]
