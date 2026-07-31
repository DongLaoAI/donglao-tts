"""Safe loading helpers for local training checkpoints."""

import torch


def load_checkpoint(path, map_location=None, *, mmap=False):
    """Load the tensor/primitive checkpoint format written by ``donglao-train``.

    ``weights_only=True`` deliberately rejects arbitrary Python objects and their pickle reduce
    hooks. Use ``safetensors`` for distributed model bundles (see :mod:`donglao_tts.hub`).
    """
    return torch.load(
        path,
        map_location=map_location,
        weights_only=True,
        mmap=mmap,
    )
