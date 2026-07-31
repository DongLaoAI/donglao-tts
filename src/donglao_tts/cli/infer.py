import argparse
import os
import time

import sentencepiece as spm
import torch
from donglao_g2p import Pipeline

from donglao_tts.checkpoint import load_checkpoint
from donglao_tts.config import load_config
from donglao_tts.generate import generate_sample
from donglao_tts.models.build import build_models
from donglao_tts.models.codec.moss_codec import MossCodec
from donglao_tts.models.embeddings import SpecialTokens, migrate_legacy_ar_state_dict
from donglao_tts.utils.precision import resolve_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                         help="path to a config YAML, resolved relative to the current working "
                              "directory (same convention as donglao-train)")
    parser.add_argument("--device", default=None,
                         help="'cuda' or 'cpu'; default auto-detects (cuda if available)")
    parser.add_argument("--benchmark", type=int, default=1,
                         help="run generation this many times and print per-run timing "
                              "(1 = no benchmarking, just generate once)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg["model"]["precision"], device)

    spm_path = cfg["tokenizer"]["model_path"]
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    special = SpecialTokens(sp)
    vocab_size = sp.get_piece_size()

    ar_model, nar_model, codebook_size, num_quantizers = build_models(cfg, vocab_size, device)

    checkpoint_dir = cfg["train"]["checkpoint_dir"]
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("step_")] \
        if os.path.isdir(checkpoint_dir) else []
    if checkpoints:
        latest = sorted(checkpoints, key=lambda f: int(f.split("_")[1].split(".")[0]))[-1]
        ckpt = load_checkpoint(
            os.path.join(checkpoint_dir, latest),
            map_location=device,
        )
        ar_state = migrate_legacy_ar_state_dict(ckpt["ar_model"], vocab_size)
        ar_model.load_state_dict(ar_state)
        nar_model.load_state_dict(ckpt["nar_model"])
        print(f"loaded checkpoint {latest}")
    else:
        print("no checkpoint found -- generating with randomly initialized weights "
              "(verifies the generate/decode plumbing only, output will not be meaningful speech)")

    codec = MossCodec.from_config(args.config)
    pipeline = Pipeline()

    ar_model.eval()
    nar_model.eval()
    for i in range(args.benchmark):
        start = time.time()
        gen_wav, ref_wav = generate_sample(cfg, ar_model, nar_model, codec, sp, special,
                                            codebook_size, num_quantizers, device, dtype, pipeline,
                                            max_frames=cfg["sample"]["max_frames"])
        if args.benchmark > 1:
            print(f"run {i}: {time.time() - start:.3f}s")

    gen_output_path = cfg["sample"]["output_path"]
    ref_output_path = os.path.join(os.path.dirname(gen_output_path), "ref.wav")
    codec.save_audio(ref_wav, ref_output_path)
    print(f"saved ref audio to {ref_output_path}")

    if gen_wav is None:
        raise RuntimeError(
            "AR generated 0 frames -- nothing to decode. With randomly initialized weights "
            "(no checkpoint found) this is expected: the model has no reason to predict a codec "
            "id on its first step. Train first via `donglao-train`, or point "
            "train.checkpoint_dir at a real checkpoint."
        )

    peak = gen_wav.abs().max().item()
    print(f"decoded audio shape {tuple(gen_wav.shape)}, peak amplitude {peak:.4f}")
    assert peak > 1e-3, "decoded audio is silent"

    codec.save_audio(gen_wav, gen_output_path)
    print(f"saved gen audio to {gen_output_path}")


if __name__ == "__main__":
    main()
