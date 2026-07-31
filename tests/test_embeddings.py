import torch

from donglao_tts.models.embeddings import (
    CodecEmbeddingTable,
    SpecialTokens,
    SplitEmbedding,
    build_input_embeds,
    extract_target_hidden,
)


class FakeSP:
    _IDS = {"[BOS]": 3, "[EOS]": 4, "[TEXT_REF]": 5, "[CODE_REF]": 6,
            "[TEXT_TARGET]": 7, "[CODE_TARGET]": 8}

    def piece_to_id(self, tag):
        return self._IDS.get(tag, 0)

    def unk_id(self):
        return 0


NUM_QUANTIZERS = 8       # codec's true RVQ depth -- how many layers ref_codec/target_codec store
REF_NUM_QUANTIZERS = 4   # how many of those layers the AR's embed table actually embeds for ref_codec


def _make_batch(B=2):
    torch.manual_seed(0)
    return {
        "ref_text_ids": torch.randint(9, 20, (B, 5)),
        "ref_text_len": torch.tensor([5, 3]),
        "ref_codec": torch.randint(0, 32, (B, 6, NUM_QUANTIZERS)),
        "ref_codec_len": torch.tensor([6, 4]),
        "target_text_ids": torch.randint(9, 20, (B, 4)),
        "target_text_len": torch.tensor([4, 2]),
        "target_codec": torch.randint(0, 32, (B, 7, 8)),
        "target_codec_len": torch.tensor([7, 5]),
    }


def test_special_tokens_resolved():
    special = SpecialTokens(FakeSP())
    assert special.bos == 3
    assert special.eos == 4
    assert special.text_ref == 5
    assert special.code_ref == 6
    assert special.text_target == 7
    assert special.code_target == 8


def test_codec_embedding_table_shapes():
    table = CodecEmbeddingTable(codebook_size=32, num_quantizers=8, d_model=16)
    codes = torch.randint(0, 32, (2, 6, 8))
    frame_embed = table.embed_frames(codes)
    assert frame_embed.shape == (2, 6, 16)

    lengths = torch.tensor([6, 4])
    pooled = table.pool_frames(codes, lengths)
    assert pooled.shape == (2, 16)


def test_split_embedding_shapes_and_separate_tables():
    vocab_size, codebook_size, num_quantizers, d_model = 20, 32, 8, 16
    embed = SplitEmbedding(vocab_size, codebook_size, num_quantizers, d_model)

    # text and codec now live in two entirely independent tables, not offset sub-ranges of one.
    assert embed.text_table.num_embeddings == vocab_size
    assert embed.codec_table.audio_embed.num_embeddings == num_quantizers * codebook_size

    text_ids = torch.tensor([0, 5, 19])
    text_vecs = embed.embed_text(text_ids)
    assert text_vecs.shape == (3, d_model)

    single_layer = embed.embed_codec_layer(torch.randint(0, codebook_size, (2, 6)), 0)
    assert single_layer.shape == (2, 6, d_model)

    T = 6
    codes_2d = torch.randint(0, codebook_size, (T, num_quantizers))
    summed = embed.embed_codec_sum(codes_2d)
    assert summed.shape == (T, d_model)
    # one summed vector per frame -- must equal manually summing each frame's per-layer embeddings.
    for t in range(T):
        expected = sum(embed.embed_codec_layer(codes_2d[t, k], k) for k in range(num_quantizers))
        assert torch.allclose(summed[t], expected, atol=1e-6)

    # text id 5 and codec layer-0 code 5 are now looked up in different modules/parameters
    # entirely -- there's no shared id space left to collide in.
    text_row = embed.embed_text(torch.tensor(5))
    codec_row = embed.embed_codec_layer(torch.tensor(5), 0)
    assert not torch.equal(text_row, codec_row)
    assert embed.text_table.weight is not embed.codec_table.audio_embed.weight


def test_build_input_embeds_shapes_and_labels():
    B, V_p, d_model = 2, 20, 16
    special = SpecialTokens(FakeSP())
    batch = _make_batch(B)
    embed = SplitEmbedding(vocab_size=V_p, codebook_size=32,
                           num_quantizers=REF_NUM_QUANTIZERS, d_model=d_model)

    input_embeds, labels, padding_mask, target_start_idx = build_input_embeds(
        embed, special, batch)

    assert input_embeds.shape[0] == B
    assert input_embeds.shape[2] == d_model
    assert labels.shape == padding_mask.shape == input_embeds.shape[:2]

    for b in range(B):
        Lr, Tr = int(batch["ref_text_len"][b]), int(batch["ref_codec_len"][b])
        Lt, Tt = int(batch["target_text_len"][b]), int(batch["target_codec_len"][b])
        # ref_codec is summed per frame (one position per frame, not per RVQ layer) -- only the
        # first REF_NUM_QUANTIZERS of the codec's NUM_QUANTIZERS stored layers feed the sum
        expected_L = 5 + Lr + Tr + Lt + Tt

        real_L = int((~padding_mask[b]).sum())
        assert real_L == expected_L
        assert int(target_start_idx[b]) == expected_L - Tt

        sample_labels = labels[b][labels[b] != -100]
        assert sample_labels.shape[0] == Tt + 1
        assert sample_labels[-1].item() == embed.codebook_size  # dedicated EOS class
        assert torch.equal(sample_labels[:-1], batch["target_codec"][b, :Tt, 0])


def test_build_input_embeds_ignores_ref_codec_layers_beyond_ref_num_quantizers():
    B, V_p, d_model = 2, 20, 16
    special = SpecialTokens(FakeSP())
    batch = _make_batch(B)
    embed = SplitEmbedding(vocab_size=V_p, codebook_size=32,
                           num_quantizers=REF_NUM_QUANTIZERS, d_model=d_model)

    input_embeds, _, _, _ = build_input_embeds(embed, special, batch)

    # perturbing ref_codec layers >= REF_NUM_QUANTIZERS must not change the assembled sequence at
    # all -- those layers are never read, matching SplitEmbedding's smaller codec sub-range.
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["ref_codec"][:, :, REF_NUM_QUANTIZERS:] = torch.randint(
        0, 32, batch2["ref_codec"][:, :, REF_NUM_QUANTIZERS:].shape)
    input_embeds2, _, _, _ = build_input_embeds(embed, special, batch2)

    assert torch.equal(input_embeds, input_embeds2)


def test_build_input_embeds_gradient_flows_to_both_tables():
    B, V_p, d_model = 2, 20, 16
    special = SpecialTokens(FakeSP())
    batch = _make_batch(B)
    embed = SplitEmbedding(vocab_size=V_p, codebook_size=32,
                           num_quantizers=REF_NUM_QUANTIZERS, d_model=d_model)

    input_embeds, _, _, _ = build_input_embeds(embed, special, batch)
    input_embeds.sum().backward()

    # gradient should reach both tables independently now -- there's no single shared table left.
    assert embed.text_table.weight.grad is not None
    assert embed.codec_table.audio_embed.weight.grad is not None
    assert embed.text_table.weight.grad.abs().sum() > 0
    assert embed.codec_table.audio_embed.weight.grad.abs().sum() > 0


def test_build_input_embeds_target_prompt_aug_default_off_matches_full_target_loss():
    B, V_p, d_model = 2, 20, 16
    special = SpecialTokens(FakeSP())
    batch = _make_batch(B)
    embed = SplitEmbedding(vocab_size=V_p, codebook_size=32,
                           num_quantizers=REF_NUM_QUANTIZERS, d_model=d_model)

    # prob=0.0 (the default) must behave exactly like no augmentation at all: every target frame
    # remains a loss target.
    _, labels, _, _ = build_input_embeds(embed, special, batch, target_prompt_aug_prob=0.0)
    for b in range(B):
        Tt = int(batch["target_codec_len"][b])
        sample_labels = labels[b][labels[b] != -100]
        assert sample_labels.shape[0] == Tt + 1


def test_build_input_embeds_target_prompt_aug_masks_a_prefix_not_the_input():
    B, V_p, d_model = 2, 20, 16
    special = SpecialTokens(FakeSP())
    batch = _make_batch(B)
    embed = SplitEmbedding(vocab_size=V_p, codebook_size=32,
                           num_quantizers=REF_NUM_QUANTIZERS, d_model=d_model)

    input_embeds_noaug, _, _, _ = build_input_embeds(embed, special, batch,
                                                       target_prompt_aug_prob=0.0)

    # prob=1.0 forces the cut on every sample with Tt > 1 (both samples here have Tt=7 and Tt=5).
    torch.manual_seed(123)
    input_embeds_aug, labels_aug, _, target_start_idx = build_input_embeds(
        embed, special, batch, target_prompt_aug_prob=1.0)

    # the input sequence (what the model actually sees/conditions on) must be byte-for-byte
    # identical whether or not the augmentation fires -- only which positions count as a loss
    # target changes, never the fed-in content.
    assert torch.equal(input_embeds_noaug, input_embeds_aug)

    for b in range(B):
        Tt = int(batch["target_codec_len"][b])
        target_rvq0 = batch["target_codec"][b, :Tt, 0]
        pos_start = int(target_start_idx[b]) - 1

        sample_target_labels = labels_aug[b, pos_start:pos_start + Tt]
        n_valid = int((sample_target_labels != -100).sum())
        # at least one frame must remain a real target (cut in [1, Tt-1]), and at least one frame
        # (the cut prefix) must have been masked out, since prob=1.0 forces a cut for Tt>1.
        assert 1 <= n_valid <= Tt - 1

        cut = Tt - n_valid
        assert torch.all(sample_target_labels[:cut] == -100)
        assert torch.equal(sample_target_labels[cut:], target_rvq0[cut:])
        assert labels_aug[b, pos_start + Tt].item() == embed.codebook_size  # EOS unaffected


def test_extract_target_hidden_picks_correct_span():
    B, V_p, d_model = 2, 20, 16
    special = SpecialTokens(FakeSP())
    batch = _make_batch(B)
    embed = SplitEmbedding(vocab_size=V_p, codebook_size=32,
                           num_quantizers=REF_NUM_QUANTIZERS, d_model=d_model)

    _, _, padding_mask, target_start_idx = build_input_embeds(embed, special, batch)
    L_max = padding_mask.shape[1]

    # fake hidden state: value at position i == i (for every batch/feature), so extraction is
    # directly verifiable against the expected index range.
    hidden = torch.arange(L_max).float().view(1, L_max, 1).expand(B, L_max, d_model).clone()

    target_hidden = extract_target_hidden(hidden, batch["target_codec_len"], target_start_idx)
    Tt_max = int(batch["target_codec_len"].max())
    assert target_hidden.shape == (B, Tt_max, d_model)

    for b in range(B):
        Tt = int(batch["target_codec_len"][b])
        start = int(target_start_idx[b])
        expected = torch.arange(start, start + Tt).float()
        assert torch.equal(target_hidden[b, :Tt, 0], expected)
