"""Export the AR/NAR models to ONNX and cross-check ONNX Runtime output against PyTorch eager on
a real checkpoint + a real config sample (``sample:`` in the config) -- unlike
``tests/test_onnx_export.py`` (small random-init models), this exercises the actual trained
weights ``donglao-infer`` would load, end to end, and writes out a real decoded .wav.

Both ``custom`` and ``qwen3`` AR backbones export prefill + cached decode-step graphs. The Qwen3
decode wrapper converts its internal Transformers ``DynamicCache`` to/from plain ONNX tensors.
"""

import argparse
import os
import time

import numpy as np
import sentencepiece as spm
import torch
from donglao_g2p import Pipeline

from donglao_tts.checkpoint import load_checkpoint
from donglao_tts.config import load_config
from donglao_tts.export.onnx_export import (
    export_ar_decode_step,
    export_ar_prefill,
    export_ar_qwen3_decode_step,
    export_ar_qwen3_prefill,
    export_nar_layer,
)
from donglao_tts.generate import ar_generate_rvq0, build_sample_from_config, make_batch, nar_fill_layers
from donglao_tts.models.build import build_models
from donglao_tts.models.codec.moss_codec import MossCodec
from donglao_tts.models.embeddings import SpecialTokens, build_input_embeds, migrate_legacy_ar_state_dict

CPU = torch.device("cpu")


def _load_real_model(cfg, device):
    spm_path = cfg["tokenizer"]["model_path"]
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    special = SpecialTokens(sp)
    vocab_size = sp.get_piece_size()

    ar_model, nar_model, codebook_size, num_quantizers = build_models(cfg, vocab_size, device)

    checkpoint_dir = cfg["train"]["checkpoint_dir"]
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("step_")] \
        if os.path.isdir(checkpoint_dir) else []
    if not checkpoints:
        raise RuntimeError(
            f"no step_*.pt checkpoint found in {checkpoint_dir!r} -- ONNX parity checks need "
            "real trained weights, not random init (donglao-train first)"
        )
    latest = sorted(checkpoints, key=lambda f: int(f.split("_")[1].split(".")[0]))[-1]
    ckpt = load_checkpoint(os.path.join(checkpoint_dir, latest), map_location=device)
    ar_model.load_state_dict(migrate_legacy_ar_state_dict(ckpt["ar_model"], vocab_size))
    nar_model.load_state_dict(ckpt["nar_model"])
    ar_model.eval()
    nar_model.eval()
    print(f"loaded checkpoint {latest}")
    return ar_model, nar_model, sp, special, codebook_size, num_quantizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                         help="path to a config YAML, resolved relative to the current working "
                              "directory (same convention as donglao-train/donglao-infer)")
    parser.add_argument("--out-dir", default=None,
                         help="where to write exported .onnx files + gen_onnx.wav "
                              "(default: <train.checkpoint_dir>/onnx)")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
        from donglao_tts.export.onnx_generate import OnnxARGenerator, OnnxNARGenerator
    except ImportError as e:
        raise ImportError(
            "donglao-onnx-infer needs the 'export' extra: "
            'pip install "donglao-tts[export]"'
        ) from e

    cfg = load_config(args.config)
    d_model = cfg["model"]["d_model"]
    backbone = cfg["model"]["ar"].get("backbone", "custom")
    out_dir = args.out_dir or os.path.join(cfg["train"]["checkpoint_dir"], "onnx")
    os.makedirs(out_dir, exist_ok=True)

    # ONNX export needs CPU fp32 tensors (the export helpers build their dummy inputs on CPU) --
    # loading straight onto CPU also makes the ONNX Runtime (CPUExecutionProvider) vs PyTorch
    # eager comparison below a true apples-to-apples fp32/CPU one, not conflated with any
    # separate CUDA-vs-CPU numerical drift.
    ar_model, nar_model, sp, special, codebook_size, num_quantizers = _load_real_model(cfg, CPU)
    ar_model = ar_model.float()
    nar_model = nar_model.float()

    print(f"backbone={backbone!r} -- exporting ONNX graphs to {out_dir}")
    nar_path = export_nar_layer(nar_model, os.path.join(out_dir, "nar_layer.onnx"), d_model)
    decode_path = None
    if backbone == "custom":
        prefill_path = export_ar_prefill(ar_model, os.path.join(out_dir, "ar_prefill.onnx"), d_model)
        decode_path = export_ar_decode_step(ar_model, os.path.join(out_dir, "ar_decode_step.onnx"), d_model)
    elif backbone == "qwen3":
        prefill_path = export_ar_qwen3_prefill(ar_model, os.path.join(out_dir, "ar_qwen3_prefill.onnx"), d_model)
        decode_path = export_ar_qwen3_decode_step(
            ar_model, os.path.join(out_dir, "ar_qwen3_decode_step.onnx"), d_model
        )
    else:
        raise ValueError(f"unknown model.ar.backbone: {backbone!r}")

    codec = MossCodec.from_config(args.config)
    pipeline = Pipeline()
    ref_text_ids, ref_codec, target_text_ids = build_sample_from_config(cfg["sample"], codec, sp, pipeline)

    # 1) Deterministic prefill logits parity: PyTorch eager vs ONNX Runtime, same real prompt.
    empty_target_codec = torch.zeros(0, ref_codec.shape[1], dtype=torch.long)
    batch = make_batch(ref_text_ids, ref_codec, target_text_ids, empty_target_codec, CPU)
    with torch.no_grad():
        input_embeds, _, pad_mask, _ = build_input_embeds(ar_model.embed, special, batch)
        logits_pt, _, _ = ar_model(input_embeds, padding_mask=pad_mask, use_cache=True)

    sess = ort.InferenceSession(prefill_path, providers=["CPUExecutionProvider"])
    logits_onnx, _, _ = sess.run(None, {
        "input_embeds": input_embeds.numpy().astype(np.float32),
        "padding_mask": pad_mask.numpy(),
    })
    max_diff = float(np.abs(logits_pt.detach().numpy() - logits_onnx).max())
    print(f"AR prefill logits max abs diff (PyTorch vs ONNX Runtime): {max_diff:.3e}")
    assert max_diff < 1e-3, "AR prefill ONNX export does not match PyTorch eager output"

    # 2) Real PyTorch reference generation (realistic sampling per config) -> rvq0 codes + AR hidden.
    # Forced fp32 here regardless of model.precision: the ONNX graphs are always exported/run in
    # fp32 (see onnx_export.py), so comparing against a bf16-autocast PyTorch reference would
    # measure precision drift, not export correctness (confirmed directly: a bf16 PyTorch NAR
    # reference disagreed with the fp32 ONNX NAR on ~15% of argmax picks, while an fp32 PyTorch
    # reference matched it exactly, 0 mismatches).
    sample_cfg = cfg["sample"]
    dtype = torch.float32
    t0 = time.perf_counter()
    rvq0_codes, ar_hidden = ar_generate_rvq0(
        ar_model, special, codebook_size, ref_text_ids, ref_codec, target_text_ids, CPU, dtype,
        max_frames=sample_cfg["max_frames"], temperature=sample_cfg.get("temperature", 0.8),
        top_k=sample_cfg.get("top_k", 10))
    ar_pt_time = time.perf_counter() - t0
    if len(rvq0_codes) == 0:
        raise RuntimeError("AR generated 0 frames -- nothing to feed the NAR/ONNX check")
    print(f"PyTorch AR generated {len(rvq0_codes)} RVQ0 frames in {ar_pt_time:.3f}s (CPU)")

    t0 = time.perf_counter()
    full_codes_pt = nar_fill_layers(nar_model, ar_hidden, rvq0_codes, num_quantizers, CPU, dtype)
    nar_pt_time = time.perf_counter() - t0

    # 3) Same rvq0_codes/ar_hidden through the ONNX NAR.
    onnx_nar = OnnxNARGenerator(nar_path, num_quantizers)
    t0 = time.perf_counter()
    full_codes_onnx = torch.from_numpy(onnx_nar.fill_layers(ar_hidden, rvq0_codes))
    nar_onnx_time = time.perf_counter() - t0
    n_mismatched = int((full_codes_pt[:, 1:] != full_codes_onnx[:, 1:]).sum())
    total = full_codes_pt[:, 1:].numel()
    print(f"NAR argmax codes mismatched vs PyTorch: {n_mismatched}/{total}")
    print(f"NAR timing (CPU, {len(rvq0_codes)} frames): PyTorch eager {nar_pt_time:.3f}s, "
          f"ONNX Runtime {nar_onnx_time:.3f}s")

    # 4) Decode the ONNX-produced codes to real audio.
    t0 = time.perf_counter()
    gen_wav_onnx = codec.decode(full_codes_onnx.transpose(0, 1))  # [T,n_q] -> [n_q,T]
    decode_time = time.perf_counter() - t0
    out_wav = os.path.join(out_dir, "gen_onnx.wav")
    codec.save_audio(gen_wav_onnx, out_wav)
    print(f"saved ONNX-generated audio to {out_wav} "
          f"(peak amplitude {gen_wav_onnx.abs().max().item():.4f})")

    # RTF = wall-clock processing time / generated audio duration (<1 means faster than real
    # time). ONNX Runtime here only has CPUExecutionProvider available in this environment (see
    # `ort.get_available_providers()`) -- both the "onnx" and "pytorch reference" RTF below are
    # CPU numbers using the same AR-in-PyTorch generation, so they isolate the NAR's ONNX-vs-eager
    # cost specifically; they are NOT representative of the GPU RTF donglao-infer gets in
    # production (model.precision/device in this config target cuda/bf16, not cpu/fp32).
    audio_duration = gen_wav_onnx.shape[-1] / codec.sampling_rate
    onnx_pipeline_time = ar_pt_time + nar_onnx_time + decode_time
    pt_pipeline_time = ar_pt_time + nar_pt_time + decode_time
    print(f"\naudio duration: {audio_duration:.3f}s")
    print(f"RTF (AR PyTorch + NAR ONNX + decode, CPU fp32):    {onnx_pipeline_time / audio_duration:.3f}  "
          f"(ar={ar_pt_time:.3f}s nar={nar_onnx_time:.3f}s decode={decode_time:.3f}s)")
    print(f"RTF (AR PyTorch + NAR PyTorch + decode, CPU fp32): {pt_pipeline_time / audio_duration:.3f}  "
          f"(ar={ar_pt_time:.3f}s nar={nar_pt_time:.3f}s decode={decode_time:.3f}s)")

    # 5) Fully ONNX AR + NAR generation for either backbone. The lightweight embedding lookup and
    # sampling loop remain in PyTorch/numpy; all Transformer compute runs in ONNX Runtime.
    onnx_ar = OnnxARGenerator(prefill_path, decode_path, ar_model.embed, codebook_size)
    t0 = time.perf_counter()
    onnx_rvq0, onnx_ar_hidden = onnx_ar.generate_rvq0(
        input_embeds,
        pad_mask,
        max_frames=sample_cfg["max_frames"],
        temperature=0,
        seed=0,
        return_hidden=True,
    )
    ar_onnx_time = time.perf_counter() - t0
    print(f"full ONNX AR generation (greedy) produced {len(onnx_rvq0)} frames "
          f"in {ar_onnx_time:.3f}s (CPU)")
    if len(onnx_rvq0) > 0:
        t0 = time.perf_counter()
        onnx_only_codes = torch.from_numpy(onnx_nar.fill_layers(onnx_ar_hidden, onnx_rvq0))
        onnx_only_nar_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        onnx_only_wav = codec.decode(onnx_only_codes.transpose(0, 1))
        onnx_only_decode_time = time.perf_counter() - t0
        onnx_only_path = os.path.join(out_dir, "gen_full_onnx.wav")
        codec.save_audio(onnx_only_wav, onnx_only_path)
        onnx_only_duration = onnx_only_wav.shape[-1] / codec.sampling_rate
        full_onnx_only_time = ar_onnx_time + onnx_only_nar_time + onnx_only_decode_time
        print(f"saved full ONNX-generated audio to {onnx_only_path}")
        print(f"RTF (ONNX AR + ONNX NAR + decode, CPU fp32): "
              f"{full_onnx_only_time / onnx_only_duration:.3f}  "
              f"(ar={ar_onnx_time:.3f}s nar={onnx_only_nar_time:.3f}s "
              f"decode={onnx_only_decode_time:.3f}s)")


if __name__ == "__main__":
    main()
