import pytest
import torch
from types import SimpleNamespace

from donglao_tts.cli.train import extend_optimizer_state, get_aux_weight, run_step
from donglao_tts.models.ar_model import ARTransformerLM
from donglao_tts.models.nar_model import NARLayerPredictor
from donglao_tts.models.training_aux import CTCAuxiliaryHead, eos_auxiliary_loss


def test_ctc_auxiliary_head_upsamples_by_two_and_backpropagates():
    torch.manual_seed(0)
    head = CTCAuxiliaryHead(d_model=8, vocab_size=12, upsample_factor=2)
    hidden = torch.randn(2, 5, 8, requires_grad=True)

    logits = head(hidden)
    assert logits.shape == (2, 10, 13)
    assert head.blank_id == 12

    targets = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    loss = head.loss(
        hidden,
        input_lengths=torch.tensor([5, 4]),
        targets=targets,
        target_lengths=torch.tensor([3, 2]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None
    assert hidden.grad.abs().sum() > 0
    assert head.upsample.weight.grad is not None


def test_ctc_auxiliary_head_rejects_unsupported_factor():
    with pytest.raises(ValueError, match="upsample_factor=2"):
        CTCAuxiliaryHead(d_model=8, vocab_size=12, upsample_factor=4)


def test_eos_auxiliary_loss_rewards_codec_then_eos_without_a_new_head():
    codebook_size = 3
    # target_start=2 means prediction positions are 1,2,3: two codec frames then EOS.
    target_start = torch.tensor([2])
    target_lengths = torch.tensor([2])
    good = torch.zeros(1, 5, codebook_size + 1)
    good[0, 1:3, 0] = 6.0
    good[0, 3, codebook_size] = 6.0

    bad = good.clone()
    bad[0, 1, codebook_size] = 8.0  # early EOS
    bad[0, 3, codebook_size] = -8.0  # missing final EOS

    good_loss = eos_auxiliary_loss(good, target_lengths, target_start, codebook_size)
    bad_loss = eos_auxiliary_loss(bad, target_lengths, target_start, codebook_size)
    assert good_loss < bad_loss


def test_eos_auxiliary_loss_balances_short_and_long_samples():
    codebook_size = 2
    logits = torch.zeros(2, 10, codebook_size + 1, requires_grad=True)
    lengths = torch.tensor([2, 7])
    starts = torch.tensor([1, 1])
    loss = eos_auxiliary_loss(logits, lengths, starts, codebook_size)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_auxiliary_weight_ramps_linearly():
    assert get_aux_weight(step=0, warmup_steps=100, base_weight=0.1) == 0.0
    assert get_aux_weight(step=50, warmup_steps=100, base_weight=0.1) == pytest.approx(0.05)
    assert get_aux_weight(step=200, warmup_steps=100, base_weight=0.1) == 0.1
    assert get_aux_weight(step=0, warmup_steps=0, base_weight=0.1) == 0.1


def test_legacy_optimizer_state_can_add_fresh_ctc_parameters():
    main = torch.nn.Linear(3, 2)
    old_optimizer = torch.optim.AdamW(main.parameters(), lr=1e-3)
    main(torch.ones(1, 3)).sum().backward()
    old_optimizer.step()

    auxiliary = torch.nn.Linear(2, 4)
    combined_params = list(main.parameters()) + list(auxiliary.parameters())
    resumed_optimizer = torch.optim.AdamW(combined_params, lr=1e-3)
    migrated = extend_optimizer_state(
        old_optimizer.state_dict(), extra_param_count=len(list(auxiliary.parameters()))
    )
    resumed_optimizer.load_state_dict(migrated)

    assert len(resumed_optimizer.param_groups[0]["params"]) == len(combined_params)
    assert len(resumed_optimizer.state) == len(list(main.parameters()))
    assert all(parameter in resumed_optimizer.state for parameter in main.parameters())
    assert all(parameter not in resumed_optimizer.state for parameter in auxiliary.parameters())


def test_run_step_combines_ctc_eos_ar_and_nar_losses():
    torch.manual_seed(3)
    batch_size, quantizers = 2, 3
    codebook_size, vocab_size, d_model = 16, 20, 16
    ar_model = ARTransformerLM(
        vocab_size, codebook_size, quantizers, d_model,
        n_layers=1, n_heads=4, ffn_dim=32, dropout=0.0,
    )
    nar_model = NARLayerPredictor(
        codebook_size, quantizers, d_model,
        n_layers=1, n_heads=4, ffn_dim=32, dropout=0.0,
    )
    ctc_model = CTCAuxiliaryHead(d_model, vocab_size)
    special = SimpleNamespace(
        bos=0, eos=1, text_ref=2, code_ref=3, text_target=4, code_target=5
    )
    batch = {
        "ref_text_ids": torch.randint(6, vocab_size, (batch_size, 3)),
        "ref_text_len": torch.tensor([3, 2]),
        "ref_codec": torch.randint(0, codebook_size, (batch_size, 4, quantizers)),
        "ref_codec_len": torch.tensor([4, 3]),
        "target_text_ids": torch.randint(6, vocab_size, (batch_size, 3)),
        "target_text_len": torch.tensor([3, 2]),
        "target_codec": torch.randint(0, codebook_size, (batch_size, 5, quantizers)),
        "target_codec_len": torch.tensor([5, 4]),
    }

    loss, ar_loss, nar_loss, ctc_loss, eos_loss, nar_by_layer = run_step(
        ar_model, nar_model, ctc_model, special, quantizers, batch,
        torch.device("cpu"), torch.bfloat16, 1.0, 1.0,
        ctc_loss_weight=0.1, eos_aux_loss_weight=0.1,
    )
    expected = ar_loss + nar_loss + 0.1 * ctc_loss + 0.1 * eos_loss
    torch.testing.assert_close(loss, expected)
    assert set(nar_by_layer) == {1, 2}
    assert all(torch.isfinite(value) for value in (loss, ar_loss, nar_loss, ctc_loss, eos_loss))

    loss.backward()
    assert ctc_model.upsample.weight.grad is not None
    assert ar_model.head.weight.grad is not None
