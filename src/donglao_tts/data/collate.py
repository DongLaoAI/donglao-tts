import torch
from torch.nn.utils.rnn import pad_sequence


def _pad_codec(tensors, pad_value=0):
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    T_max = int(lengths.max())
    K = tensors[0].shape[1]
    out = torch.full((len(tensors), T_max, K), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out, lengths


def collate_fn(batch, pad_id=0):
    ref_text_ids = [b["ref_text_ids"] for b in batch]
    target_text_ids = [b["target_text_ids"] for b in batch]

    ref_text_len = torch.tensor([t.shape[0] for t in ref_text_ids], dtype=torch.long)
    target_text_len = torch.tensor([t.shape[0] for t in target_text_ids], dtype=torch.long)

    ref_text_padded = pad_sequence(
        ref_text_ids, batch_first=True, padding_value=pad_id
    ).long()
    target_text_padded = pad_sequence(
        target_text_ids, batch_first=True, padding_value=pad_id
    ).long()

    ref_codec, ref_codec_len = _pad_codec([b["ref_codec"] for b in batch])
    target_codec, target_codec_len = _pad_codec([b["target_codec"] for b in batch])
    # Compiled shards store codes as uint16. Cast once after padding instead of allocating an
    # int64 tensor for every individual sample.
    ref_codec = ref_codec.long()
    target_codec = target_codec.long()

    return {
        "ref_text_ids": ref_text_padded, "ref_text_len": ref_text_len,
        "ref_codec": ref_codec, "ref_codec_len": ref_codec_len,
        "target_text_ids": target_text_padded, "target_text_len": target_text_len,
        "target_codec": target_codec, "target_codec_len": target_codec_len,
    }
