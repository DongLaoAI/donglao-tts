import random

import torch
from torch import nn

SPECIAL_NAMES = ["bos", "eos", "text_ref", "code_ref", "text_target", "code_target"]


class SpecialTokens:
    def __init__(self, sp):
        for name in SPECIAL_NAMES:
            tag = f"[{name.upper()}]"
            token_id = sp.piece_to_id(tag)
            if token_id == sp.unk_id():
                raise ValueError(f"special token {tag} not found in SentencePiece vocab")
            setattr(self, name, token_id)


class CodecEmbeddingTable(nn.Module):
    """OmniVoice-style: one embedding table of size num_quantizers*codebook_size, one independent
    sub-range (offset by layer_idx*codebook_size) per RVQ layer -- each (layer, code) pair gets
    its own freely-learned vector (no forced sharing across layers), since a given raw code index
    means something different depending on which residual layer it belongs to. Reused by ref-codec
    (all layers, summed per frame) and target-codec (single layer at a time)."""

    def __init__(self, codebook_size, num_quantizers, d_model):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.audio_embed = nn.Embedding(num_quantizers * codebook_size, d_model)
        self.register_buffer(
            "layer_offsets", torch.arange(num_quantizers) * codebook_size, persistent=False
        )

    def embed_layer(self, codes, layer_idx):
        # codes [...] -> [..., d_model]
        return self.audio_embed(codes + self.layer_offsets[layer_idx])

    def embed_frames(self, codes, layer_ids=None):
        # codes [..., K] -> [..., d_model], sum over K quantizer layers
        K = codes.shape[-1]
        if layer_ids is None:
            layer_ids = torch.arange(K, device=codes.device)
        shifted = codes + self.layer_offsets[layer_ids]  # [..., K]
        return self.audio_embed(shifted).sum(dim=-2)  # [..., K, d_model] -> [..., d_model]

    def pool_frames(self, codes, lengths, layer_ids=None):
        # codes [B, T, K], lengths [B] -> [B, d_model], mean over valid frames only
        frame_embed = self.embed_frames(codes, layer_ids)  # [B, T, d_model]
        time_idx = torch.arange(codes.shape[1], device=codes.device)
        mask = (time_idx[None, :] < lengths[:, None]).unsqueeze(-1).to(frame_embed.dtype)  # [B, T, 1]
        summed = (frame_embed * mask).sum(dim=1)
        return summed / lengths.clamp(min=1).unsqueeze(-1).to(frame_embed.dtype)


class SplitEmbedding(nn.Module):
    """Separate embedding tables for text/special tokens (`text_table`) and codec ids
    (`codec_table`) -- used by the AR model only (NAR keeps its own independent
    `CodecEmbeddingTable`, unaffected by this class). Previously these lived in one shared table
    with offset sub-ranges (see git history); splitting them means a text id and a codec id can
    never collide by construction (there's no shared id space to reason about at all), at the
    cost of two separate embedding matrices instead of one -- text ids index `text_table` (size
    vocab_size) directly with no offset, codec ids go through `codec_table` (a `CodecEmbeddingTable`,
    same one-sub-range-per-RVQ-layer design NAR uses), entirely independent of `text_table`."""

    def __init__(self, vocab_size, codebook_size, num_quantizers, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.text_table = nn.Embedding(vocab_size, d_model)
        self.codec_table = CodecEmbeddingTable(codebook_size, num_quantizers, d_model)

    def embed_text(self, ids):
        return self.text_table(ids)

    def embed_codec_layer(self, codes, layer_idx):
        return self.codec_table.embed_layer(codes, layer_idx)

    def embed_codec_sum(self, codes):
        """codes [T, K] -> [T, d_model], one summed vector per frame (see
        CodecEmbeddingTable.embed_frames) -- used for ref_codec: K RVQ layers collapse into a
        single attention position per frame instead of K separate positions."""
        return self.codec_table.embed_frames(codes)


def migrate_legacy_ar_state_dict(state_dict, vocab_size):
    """Backward-compat for AR checkpoints saved before the UnifiedEmbedding -> SplitEmbedding
    split: the old `embed.table.weight` [vocab_size + num_quantizers*codebook_size, d_model] had
    text rows first and codec rows after (see the old class's layer_offsets, which started at
    vocab_size) -- exactly the same order/layout SplitEmbedding's two tables use now, just as one
    tensor instead of two. So this is a lossless 1:1 split, not an approximation. Returns the
    state_dict unchanged if it's already new-style (no `embed.table.weight` key)."""
    if "embed.table.weight" not in state_dict:
        return state_dict
    state_dict = dict(state_dict)
    full = state_dict.pop("embed.table.weight")
    state_dict["embed.text_table.weight"] = full[:vocab_size].clone()
    state_dict["embed.codec_table.audio_embed.weight"] = full[vocab_size:].clone()
    return state_dict


def build_input_embeds(embed, special, batch, target_prompt_aug_prob=0.0):
    """Assemble the single-stream AR sequence:

        [BOS] [TEXT_REF] ref_text [CODE_REF] ref_codec(summed across first embed.num_quantizers
              layers per frame, T_ref pos)
              [TEXT_TARGET] target_text [CODE_TARGET] target_rvq0(T_tgt pos)

    `labels` is pre-shifted so labels[:, i] is exactly what logits[:, i] should predict (no extra
    shift needed at loss time): every position up to and including [CODE_TARGET] predicts the
    next target-codec frame's RVQ0 id, the last target-codec frame predicts EOS, everything else
    (both ref/target text and the ref-codec segment) is masked with -100 since it's conditioning,
    never a prediction target.

    `target_prompt_aug_prob` (AR-branch-only augmentation; NAR's own loss in run_step supervises
    the full target_codec_len regardless): with this probability, per sample, a random cut point
    `c` in [1, Tt-1] is drawn and labels for target frames [0, c) are ALSO masked with -100 --
    those frames are still fed as input exactly as before (this only changes which positions
    count as a loss target, not what the model conditions on), so they act as extra, in-domain
    "prompt continuation" context the model isn't tested on, and only frames [c, Tt) + EOS remain
    real prediction targets. Without this (prob 0, or Tt<=1), behavior is unchanged: the full
    target_codec is a loss target, matching plain teacher forcing.

    Label ids live entirely in the AR head's own small classification space -- [0, codebook_size)
    for a codec id, `codebook_size` for EOS -- decoupled from the text/special SentencePiece vocab
    (`special.eos` is not used here at all; the AR never needs to predict a text token). This is
    independent of `embed`'s own layout (see SplitEmbedding) -- input representation and output
    classification space are two separate design choices.
    """
    device = embed.text_table.weight.device
    d_model = embed.text_table.weight.shape[1]
    B = batch["ref_text_ids"].shape[0]

    def marker(token_id):
        return embed.embed_text(torch.tensor(token_id, device=device)).unsqueeze(0)

    seqs = []
    label_seqs = []
    target_start_idx = []
    for b in range(B):
        Lr = int(batch["ref_text_len"][b])
        Tr = int(batch["ref_codec_len"][b])
        Lt = int(batch["target_text_len"][b])
        Tt = int(batch["target_codec_len"][b])

        ref_text = embed.embed_text(batch["ref_text_ids"][b, :Lr])
        # only the first `embed.num_quantizers` RVQ layers of ref_codec get embedded (usually a
        # subset of the codec's true depth -- see SplitEmbedding/ARTransformerLM), summed into one
        # vector per frame (T_ref positions total, not T_ref*embed.num_quantizers)
        ref_codec = embed.embed_codec_sum(batch["ref_codec"][b, :Tr, :embed.num_quantizers])
        target_text = embed.embed_text(batch["target_text_ids"][b, :Lt])
        target_rvq0 = batch["target_codec"][b, :Tt, 0]
        target_codec_in = embed.embed_codec_layer(target_rvq0, 0)

        seq = torch.cat([
            marker(special.bos),
            marker(special.text_ref),
            ref_text,
            marker(special.code_ref),
            ref_codec,
            marker(special.text_target),
            target_text,
            marker(special.code_target),
            target_codec_in,
        ], dim=0)
        seqs.append(seq)

        L_b = seq.shape[0]
        pos_start = L_b - Tt - 1  # index of the [CODE_TARGET] marker
        cut = 0
        if target_prompt_aug_prob > 0 and Tt > 1 and random.random() < target_prompt_aug_prob:
            cut = random.randint(1, Tt // 2)  # frames [0, cut) become unlabeled prompt context
        labels = torch.full((L_b,), -100, dtype=torch.long, device=device)
        labels[pos_start + cut:pos_start + Tt] = target_rvq0[cut:]
        labels[pos_start + Tt] = embed.codebook_size  # dedicated EOS class
        label_seqs.append(labels)
        target_start_idx.append(pos_start + 1)  # where target_codec_in[0] (frame 0) begins

    lengths = torch.tensor([s.shape[0] for s in seqs], device=device)
    L_max = int(lengths.max())

    input_embeds = torch.zeros(B, L_max, d_model, device=device, dtype=seqs[0].dtype)
    labels_out = torch.full((B, L_max), -100, dtype=torch.long, device=device)
    padding_mask = torch.ones(B, L_max, dtype=torch.bool, device=device)

    for b in range(B):
        L_b = seqs[b].shape[0]
        input_embeds[b, :L_b] = seqs[b]
        labels_out[b, :L_b] = label_seqs[b]
        padding_mask[b, :L_b] = False

    target_start_idx = torch.tensor(target_start_idx, device=device)
    return input_embeds, labels_out, padding_mask, target_start_idx


def extract_target_hidden(hidden, target_codec_len, target_start_idx):
    """hidden [B, L, d_model] (AR's post-ln_f output) -> [B, Tt_max, d_model], the per-frame hidden
    state at each target-codec position (the state produced right after that frame's own RVQ0 code
    was fed in as input) -- this is what the NAR conditions on instead of recomputing ref/text
    context from scratch."""
    B, _, d_model = hidden.shape
    Tt_max = int(target_codec_len.max())
    out = torch.zeros(B, Tt_max, d_model, device=hidden.device, dtype=hidden.dtype)
    for b in range(B):
        start = int(target_start_idx[b])
        Tt = int(target_codec_len[b])
        out[b, :Tt] = hidden[b, start:start + Tt]
    return out
