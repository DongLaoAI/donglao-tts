import argparse
import copy
import math
import os
import random
from functools import partial

import torch
import torch.nn.functional as F
from donglao_g2p import Pipeline
from torch.utils.data import DataLoader

from donglao_tts.checkpoint import load_checkpoint
from donglao_tts.config import load_config
from donglao_tts.data.collate import collate_fn
from donglao_tts.data.dataset import MultiCorpusTTSDataset, TTSDataset
from donglao_tts.data.sampler import LengthBucketBatchSampler
from donglao_tts.generate import generate_sample
from donglao_tts.models.build import build_models
from donglao_tts.models.codec.moss_codec import MossCodec
from donglao_tts.models.embeddings import (
    SpecialTokens,
    build_input_embeds,
    extract_target_hidden,
    migrate_legacy_ar_state_dict,
)
from donglao_tts.models.training_aux import CTCAuxiliaryHead, eos_auxiliary_loss
from donglao_tts.quantization import prepare_model_qat
from donglao_tts.utils.precision import resolve_dtype


def make_time_pad_mask(lengths, T_max, device):
    idx = torch.arange(T_max, device=device)
    return idx[None, :] >= lengths[:, None].to(device)


def get_lr(step, warmup_steps, max_steps, base_lr):
    """`step` must be a real optimizer-update count (not a micro-batch/accumulation-substep
    count) -- warmup_steps/max_steps in config are authored in that unit."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def get_aux_weight(step, warmup_steps, base_weight):
    """Linearly introduce a training-only objective, then hold its configured weight."""
    if base_weight <= 0:
        return 0.0
    if warmup_steps <= 0:
        return base_weight
    return base_weight * min(1.0, step / warmup_steps)


def extend_optimizer_state(saved_state, extra_param_count):
    """Add fresh parameters to a legacy single-group Adam state without losing old momentum."""
    if extra_param_count <= 0:
        return saved_state
    state = copy.deepcopy(saved_state)
    if len(state["param_groups"]) != 1:
        raise ValueError("cannot extend a legacy optimizer with more than one parameter group")
    parameter_ids = state["param_groups"][0]["params"]
    next_id = max(parameter_ids, default=-1) + 1
    parameter_ids.extend(range(next_id, next_id + extra_param_count))
    return state


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("step_") and f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(checkpoint_dir, ckpts[-1])


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def print_run_info(cfg, device, dtype, vocab_size, codebook_size, num_quantizers,
                    ar_model, nar_model, ctc_model, train_ds, val_ds):
    model_cfg, train_cfg = cfg["model"], cfg["train"]

    print("=" * 60)
    print("Model")
    print(f"  device={device} precision={model_cfg['precision']} (dtype={dtype})")
    print(f"  vocab_size={vocab_size} codebook_size={codebook_size} num_quantizers={num_quantizers}")
    print(f"  AR : backbone={model_cfg['ar'].get('backbone', 'custom')} d_model={model_cfg['d_model']} "
          f"n_layers={model_cfg['ar']['n_layers']} n_heads={model_cfg['ar']['n_heads']} "
          f"ffn_dim={model_cfg['ar']['ffn_dim']} params={count_params(ar_model):,}")
    print(f"  NAR: d_model={model_cfg['d_model']} n_layers={model_cfg['nar']['n_layers']} "
          f"n_heads={model_cfg['nar']['n_heads']} ffn_dim={model_cfg['nar']['ffn_dim']} "
          f"params={count_params(nar_model):,}")
    print(f"  total params={count_params(ar_model) + count_params(nar_model):,}")
    if ctc_model is not None:
        print(
            f"  CTC: training-only upsample={ctc_model.upsample_factor}x "
            f"weight={train_cfg['ctc_loss_weight']} params={count_params(ctc_model):,}"
        )
    if train_cfg.get("eos_aux_loss_weight", 0.0) > 0:
        print(
            f"  EOS auxiliary: shared-logit balanced loss "
            f"weight={train_cfg['eos_aux_loss_weight']} (no inference head)"
        )

    print("Dataset")
    if "datasets" in train_cfg:
        for dataset_path in train_cfg["datasets"]:
            print(f"  compiled={dataset_path}")
        for corpus, count in sorted(train_ds.corpus_counts.items()):
            print(f"  corpus {corpus}: {count} train entries")
    else:
        for i, manifest in enumerate(train_cfg["manifests"]):
            print(
                f"  corpus {i} ({os.path.basename(manifest)}): "
                f"{train_ds.corpus_counts.get(i, 0)} train entries"
            )
    print(
        f"  train={len(train_ds)} val={len(val_ds)} "
        f"train_speakers={train_ds.num_speakers}"
    )
    print(
        f"  max_batch_size={train_cfg['batch_size']} "
        f"grad_accum_steps={train_cfg['grad_accum_steps']}"
    )
    if train_cfg.get("max_frames_per_batch", 0):
        print(
            f"  max_frames_per_batch={train_cfg['max_frames_per_batch']:,} "
            f"bucket_size={train_cfg.get('bucket_size', 256)}"
        )
    print(f"  steps_per_epoch~={len(train_ds) // train_cfg['batch_size']}")
    nar_start = train_cfg["nar_start_step"]
    print(f"  nar_start_step={nar_start} "
          f"({'joint AR+NAR from step 0' if nar_start == 0 else f'AR-only until step {nar_start}'})")
    print("=" * 60)


def run_step(ar_model, nar_model, ctc_model, special, num_quantizers, batch, device, dtype,
             ar_loss_weight, nar_loss_weight, ctc_loss_weight=0.0, eos_aux_loss_weight=0.0,
             train_nar=True, target_prompt_aug_prob=0.0):
    """NAR now conditions on the AR's own per-frame hidden state (see extract_target_hidden),
    so its gradient flows back into the AR -- real joint training, not just a shared batch. All
    `num_quantizers-1` remaining RVQ layers are trained every step (not one random layer), each via
    its own NAR forward pass reusing the same `ar_hidden_at_target`; `nar_loss` is their mean so
    `nar_loss_weight`'s scale stays comparable to a single-layer loss.

    `target_prompt_aug_prob` only affects the AR branch's labels (see build_input_embeds) --
    NAR's own loss below still supervises the full target_codec_len regardless, since it isn't
    passed through at all here."""
    batch = to_device(batch, device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        input_embeds, labels, pad_mask, target_start_idx = build_input_embeds(
            ar_model.embed, special, batch, target_prompt_aug_prob=target_prompt_aug_prob)
        ar_logits, ar_hidden, _ = ar_model(input_embeds, padding_mask=pad_mask)
        # Per-sample average (not a flattened batch-wide mean): target_codec_len varies a lot
        # across a batch (ref-codec length especially), so a flattened mean would let long
        # samples dominate the gradient and dilute the already-rare EOS signal even further
        # (every sample has exactly one EOS among target_codec_len+1 AR-loss positions).
        ar_per_token = F.cross_entropy(ar_logits.transpose(1, 2), labels, ignore_index=-100,
                                        reduction="none")  # [B, L]
        ar_valid = labels != -100
        ar_loss = (ar_per_token.sum(dim=1) / ar_valid.sum(dim=1).clamp(min=1)).mean()

        eos_loss = eos_auxiliary_loss(
            ar_logits, batch["target_codec_len"], target_start_idx, ar_model.codebook_size)
        loss = ar_loss_weight * ar_loss + eos_aux_loss_weight * eos_loss

        if ctc_model is not None and ctc_loss_weight > 0:
            ar_hidden_at_target = extract_target_hidden(
                ar_hidden, batch["target_codec_len"], target_start_idx)
            ctc_loss = ctc_model.loss(
                ar_hidden_at_target,
                batch["target_codec_len"],
                batch["target_text_ids"],
                batch["target_text_len"],
            )
            loss = loss + ctc_loss_weight * ctc_loss
        else:
            ctc_loss = torch.zeros((), device=device, dtype=ar_loss.dtype)

        nar_losses_by_layer = {}
        if not train_nar:
            # NAR training hasn't started yet (see train.nar_start_step) -- skip every layer's
            # forward/loss entirely to save compute during the AR-only warmup phase.
            nar_loss = torch.zeros((), device=device, dtype=ar_loss.dtype)
            return loss, ar_loss, nar_loss, ctc_loss, eos_loss, nar_losses_by_layer

        if ctc_model is None or ctc_loss_weight <= 0:
            ar_hidden_at_target = extract_target_hidden(
                ar_hidden, batch["target_codec_len"], target_start_idx)

        T_tgt_max = batch["target_codec"].shape[1]
        target_pad_mask = make_time_pad_mask(batch["target_codec_len"], T_tgt_max, device)

        nar_loss_sum = torch.zeros((), device=device, dtype=ar_loss.dtype)
        for k in range(1, num_quantizers):
            nar_logits = nar_model(ar_hidden_at_target, batch["target_codec"][..., :k], k,
                                    target_pad_mask)
            labels_k = batch["target_codec"][..., k].clone()
            labels_k[target_pad_mask] = -100
            loss_k = F.cross_entropy(nar_logits.transpose(1, 2), labels_k, ignore_index=-100)
            nar_losses_by_layer[k] = loss_k
            nar_loss_sum = nar_loss_sum + loss_k
        nar_loss = nar_loss_sum / (num_quantizers - 1)

    loss = loss + nar_loss_weight * nar_loss
    return loss, ar_loss, nar_loss, ctc_loss, eos_loss, nar_losses_by_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                         help="path to a training config YAML, resolved relative to the current "
                              "working directory (paths inside it -- datasets, checkpoint_dir, "
                              "tokenizer.model_path, sample.* -- are resolved the same way)")
    parser.add_argument("--resume", default="auto",
                         help="checkpoint path to resume from, 'auto' to pick the latest one in "
                              "train.checkpoint_dir, or 'none' to start fresh")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg, model_cfg = cfg["train"], cfg["model"]
    qat_enabled = train_cfg.get("qat", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if qat_enabled:
        # torch.ao.quantization's fake-quant/observer ops (torch.fused_moving_avg_obs_fake_quant)
        # only support float32 -- confirmed directly: running the usual bf16 autocast under QAT
        # raises "expected scalar type Float but found BFloat16" deep inside the wrapped Linear
        # layers' observer hooks. QAT fine-tuning runs are typically short, so paying full fp32
        # compute cost here (instead of model.precision's usual bf16/fp16) is an acceptable trade.
        dtype = torch.float32
        print(f"QAT enabled: overriding precision {model_cfg['precision']!r} -> float32 "
              f"(fake-quant observers don't support bf16/fp16)")
    else:
        dtype = resolve_dtype(model_cfg["precision"], device)

    random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])

    spm_path = cfg["tokenizer"]["model_path"]
    if "datasets" in train_cfg:
        reference_percentile = train_cfg.get("reference_percentile", 90)
        train_ds = MultiCorpusTTSDataset(
            train_cfg["datasets"],
            spm_path,
            split="train",
            reference_percentile=reference_percentile,
        )
        val_ds = MultiCorpusTTSDataset(
            train_cfg["datasets"],
            spm_path,
            split="val",
            reference_percentile=reference_percentile,
        )
    else:
        manifests = list(train_cfg["manifests"])
        train_ds = TTSDataset(
            manifests,
            spm_path,
            split="train",
            val_ratio=train_cfg["val_split_ratio"],
            seed=train_cfg["seed"],
        )
        val_ds = TTSDataset(
            manifests,
            spm_path,
            split="val",
            val_ratio=train_cfg["val_split_ratio"],
            seed=train_cfg["seed"],
        )

    collate = partial(collate_fn, pad_id=0)
    train_batch_sampler = LengthBucketBatchSampler(
        train_ds,
        train_cfg["batch_size"],
        max_frames_per_batch=train_cfg.get("max_frames_per_batch", 0),
        bucket_size=train_cfg.get("bucket_size", 256),
        shuffle=True,
        drop_last=True,
        seed=train_cfg["seed"],
    )
    train_dl = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate,
        persistent_workers=train_cfg["num_workers"] > 0,
        pin_memory=device.type == "cuda",
        prefetch_factor=train_cfg.get("prefetch_factor", 2)
        if train_cfg["num_workers"] > 0
        else None,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    sp = train_ds.sp
    special = SpecialTokens(sp)
    vocab_size = sp.get_piece_size()

    ar_model, nar_model, codebook_size, num_quantizers = build_models(cfg, vocab_size, device)
    ctc_loss_weight = float(train_cfg.get("ctc_loss_weight", 0.0))
    ctc_model = None
    if ctc_loss_weight > 0:
        ctc_model = CTCAuxiliaryHead(
            d_model=model_cfg["d_model"],
            vocab_size=vocab_size,
            upsample_factor=int(train_cfg.get("ctc_upsample_factor", 2)),
        ).to(device)
    print_run_info(cfg, device, dtype, vocab_size, codebook_size, num_quantizers,
                    ar_model, nar_model, ctc_model, train_ds, val_ds)

    checkpoint_dir = train_cfg["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    resume_arg = args.resume
    resume_path = find_latest_checkpoint(checkpoint_dir) if resume_arg == "auto" else (
        None if resume_arg == "none" else resume_arg)

    init_checkpoint = train_cfg.get("init_checkpoint")
    if resume_path is None and init_checkpoint:
        # Weight-only warm start (see train.init_checkpoint's comment in the config): applied
        # before QAT wrapping/optimizer construction, so it's always a plain, strict state_dict
        # load -- a real `--resume` of *this* run's own checkpoint_dir always takes priority and
        # skips this entirely, since that already carries forward whatever this run was
        # originally initialized from.
        print(f"initializing weights from {init_checkpoint} (fresh optimizer, step=0 -- "
              "this is a one-time init, not a resume)")
        init_ckpt = load_checkpoint(init_checkpoint, map_location=device)
        init_ar_state = init_ckpt["ar_model"]
        if "embed.table.weight" in init_ar_state:
            init_ar_state = migrate_legacy_ar_state_dict(init_ar_state, vocab_size)
        ar_model.load_state_dict(init_ar_state)
        nar_model.load_state_dict(init_ckpt["nar_model"])

    if qat_enabled:
        # Must happen before the optimizer is built: prepare_model_qat swaps nn.Linear for
        # fake-quant-wrapped modules with their own new parameter tensors in place, so building
        # `params`/`optimizer` from the *already-QAT-wrapped* models is what makes the optimizer
        # actually track them (see quantization.py's module docstring for the full workflow --
        # typically: train fp32 to convergence first, then a separate --resume run with
        # train.qat: true for a short fake-quant fine-tune).
        prepare_model_qat(ar_model, backend=train_cfg.get("qat_backend", "fbgemm"))
        prepare_model_qat(nar_model, backend=train_cfg.get("qat_backend", "fbgemm"))
        print(f"QAT enabled: nn.Linear layers wrapped with fake-quant "
              f"(backend={train_cfg.get('qat_backend', 'fbgemm')})")

    main_params = list(ar_model.parameters()) + list(nar_model.parameters())
    ctc_params = list(ctc_model.parameters()) if ctc_model is not None else []
    params = main_params + ctc_params
    optimizer = torch.optim.AdamW(params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    sample_codec = MossCodec.from_config(args.config)
    sample_pipeline = Pipeline()
    gen_output_path = cfg["sample"]["output_path"]
    ref_output_path = os.path.join(os.path.dirname(gen_output_path), "ref.wav")

    step = 0  # real optimizer-update count -- see get_lr's docstring for why this must not be a
              # micro-batch/accumulation-substep count
    # Auxiliary warmup counts start when each objective is introduced, not at the run's global
    # step. Thus adding CTC to a mature legacy checkpoint still ramps its freshly initialized head.
    ctc_step = 0
    eos_aux_step = 0
    if resume_path is not None:
        print(f"resuming from checkpoint {resume_path}")
        ckpt = load_checkpoint(resume_path, map_location=device)
        ar_state = ckpt["ar_model"]
        is_legacy_embedding = "embed.table.weight" in ar_state
        if is_legacy_embedding:
            ar_state = migrate_legacy_ar_state_dict(ar_state, vocab_size)
            print("  checkpoint used the old single-table embedding (pre-SplitEmbedding); "
                  "migrated its weights into text_table/codec_table. This changes the flat "
                  "parameter list's shape, so optimizer momentum can't be reloaded as-is -- "
                  "starting Adam state fresh (model weights are still fully restored).")
        # strict=False under QAT: a checkpoint saved before QAT was enabled won't have the
        # fake-quant/observer buffer keys prepare_model_qat added (they'll just start fresh and
        # calibrate from data as fine-tuning proceeds) -- the actual learned weights still load.
        ar_model.load_state_dict(ar_state, strict=not qat_enabled)
        nar_model.load_state_dict(ckpt["nar_model"], strict=not qat_enabled)
        checkpoint_has_ctc = ctc_model is not None and "ctc_model" in ckpt
        if checkpoint_has_ctc:
            ctc_model.load_state_dict(ckpt["ctc_model"])
        elif ctc_model is not None:
            print("  checkpoint predates the training-only CTC head; initialized CTC weights "
                  "from scratch while preserving AR/NAR weights")
        if not is_legacy_embedding and not qat_enabled:
            # optimizer state references the pre-QAT parameter tensors -- doesn't line up with
            # the freshly QAT-wrapped ones, same reasoning as the legacy-embedding case above.
            optimizer_state = ckpt["optimizer"]
            if ctc_model is not None and not checkpoint_has_ctc:
                optimizer_state = extend_optimizer_state(optimizer_state, len(ctc_params))
            optimizer.load_state_dict(optimizer_state)
        if ckpt.get("step_unit") == "optimizer":
            step = ckpt["step"]
        else:
            # checkpoint predates the optimizer-step/micro-step fix -- its "step" was a
            # micro-batch count, so divide by grad_accum_steps to get the real number of
            # optimizer updates already baked into these weights.
            step = ckpt["step"] // train_cfg["grad_accum_steps"]
            print(f"  checkpoint used legacy micro-step counting (raw step={ckpt['step']}); "
                  f"converted to {step} real optimizer steps")
        ctc_step = int(ckpt.get("ctc_step", 0)) if checkpoint_has_ctc else 0
        eos_aux_step = int(ckpt.get("eos_aux_step", 0))
        print(f"  resumed at optimizer step {step}")

    optimizer.zero_grad()
    data_iter = iter(train_dl)
    micro_step = 0
    while step < train_cfg["max_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)

        train_nar = step >= train_cfg["nar_start_step"]
        current_ctc_weight = get_aux_weight(
            ctc_step, train_cfg.get("ctc_warmup_steps", 0), ctc_loss_weight)
        current_eos_weight = get_aux_weight(
            eos_aux_step, train_cfg.get("eos_aux_warmup_steps", 0),
            float(train_cfg.get("eos_aux_loss_weight", 0.0)))
        loss, ar_loss, nar_loss, ctc_loss, eos_loss, nar_losses_by_layer = run_step(
            ar_model, nar_model, ctc_model, special, num_quantizers, batch, device, dtype,
            train_cfg["ar_loss_weight"], train_cfg["nar_loss_weight"], train_nar=train_nar,
            ctc_loss_weight=current_ctc_weight, eos_aux_loss_weight=current_eos_weight,
            target_prompt_aug_prob=train_cfg.get("target_prompt_aug_prob", 0.0))
        (loss / train_cfg["grad_accum_steps"]).backward()
        micro_step += 1

        if micro_step % train_cfg["grad_accum_steps"] != 0:
            continue  # still accumulating -- nothing below runs until a real optimizer update

        current_lr = get_lr(step, train_cfg["warmup_steps"], train_cfg["max_steps"], train_cfg["lr"])
        torch.nn.utils.clip_grad_norm_(params, train_cfg["grad_clip"])
        for g in optimizer.param_groups:
            g["lr"] = current_lr
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        if ctc_model is not None:
            ctc_step += 1
        if train_cfg.get("eos_aux_loss_weight", 0.0) > 0:
            eos_aux_step += 1

        if step % train_cfg["log_every_steps"] == 0:
            if nar_losses_by_layer:
                nar_str = " ".join(f"rvq{i}={v.item():.4f}" for i, v in nar_losses_by_layer.items())
            else:
                nar_str = f"not started (starts at step {train_cfg['nar_start_step']})"
            print(f"step {step} lr {current_lr:.2e} loss {loss.item():.4f} "
                  f"ar_rvq0 {ar_loss.item():.4f} nar_avg {nar_loss.item():.4f} "
                  f"ctc {ctc_loss.item():.4f}@{current_ctc_weight:.3f} "
                  f"eos {eos_loss.item():.4f}@{current_eos_weight:.3f} | nar {nar_str}")

        if step % train_cfg["eval_every_steps"] == 0:
            ar_model.eval()
            nar_model.eval()
            if ctc_model is not None:
                ctc_model.eval()
            with torch.no_grad():
                val_losses = []
                for i, vbatch in enumerate(val_dl):
                    if i >= 20:
                        break
                    vloss, _, _, _, _, _ = run_step(
                        ar_model, nar_model, ctc_model, special, num_quantizers, vbatch, device,
                        dtype, train_cfg["ar_loss_weight"], train_cfg["nar_loss_weight"],
                        ctc_loss_weight=current_ctc_weight,
                        eos_aux_loss_weight=current_eos_weight, train_nar=train_nar)
                    val_losses.append(vloss.item())
                if val_losses:
                    print(f"step {step} val_loss {sum(val_losses) / len(val_losses):.4f}")
            ar_model.train()
            nar_model.train()
            if ctc_model is not None:
                ctc_model.train()

        if step % train_cfg["save_every_steps"] == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
            checkpoint = {
                "ar_model": ar_model.state_dict(), "nar_model": nar_model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "step_unit": "optimizer",
                "ctc_step": ctc_step, "eos_aux_step": eos_aux_step, "config": cfg,
            }
            if ctc_model is not None:
                checkpoint["ctc_model"] = ctc_model.state_dict()
            torch.save(checkpoint, ckpt_path)
            existing = sorted(
                (f for f in os.listdir(checkpoint_dir) if f.startswith("step_")),
                key=lambda f: int(f.split("_")[1].split(".")[0]))
            for old in existing[:-train_cfg["keep_last_checkpoints"]]:
                os.remove(os.path.join(checkpoint_dir, old))

            ar_model.eval()
            nar_model.eval()
            gen_wav, ref_wav = generate_sample(cfg, ar_model, nar_model, sample_codec, sp, special,
                                                codebook_size, num_quantizers, device,
                                                dtype, sample_pipeline,
                                                max_frames=cfg["sample"]["max_frames"])
            ar_model.train()
            nar_model.train()
            sample_codec.save_audio(ref_wav, ref_output_path)  # fixed path, overwritten every save
            if gen_wav is not None:
                sample_codec.save_audio(gen_wav, gen_output_path)  # fixed path, overwritten every save
                print(f"step {step} saved sample audio to {gen_output_path} and {ref_output_path}")
            else:
                print(f"step {step} sample generation produced 0 frames, "
                      f"only saved {ref_output_path}")


if __name__ == "__main__":
    main()
