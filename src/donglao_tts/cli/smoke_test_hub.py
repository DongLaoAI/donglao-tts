"""Smoke-test an installed donglao-tts package against a Hugging Face model bundle."""

import argparse
import json
import os
import time
from importlib.metadata import PackageNotFoundError, version

import torch
from huggingface_hub import HfApi, hf_hub_download

from donglao_tts import DongLaoTTS


def _package_version():
    try:
        return version("donglao-tts")
    except PackageNotFoundError:
        return "source checkout"


def _read_hub_json(repo_id, filename, revision):
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _validate_bundle(repo_id, revision):
    manifest = _read_hub_json(repo_id, "bundle_manifest.json", revision)
    config = _read_hub_json(repo_id, "config.json", revision)

    if manifest.get("library_name") != "donglao-tts":
        raise RuntimeError("Hub repository is not a donglao-tts bundle")
    if manifest.get("format_version") != 2:
        raise RuntimeError(
            f"unsupported bundle format {manifest.get('format_version')!r}; expected 2"
        )
    native_files = set(manifest.get("native", {}).get("files", []))
    required = {"ar_model.safetensors", "nar_model.safetensors"}
    if not required.issubset(native_files):
        raise RuntimeError(f"bundle manifest is missing native weights: {sorted(required)}")
    if not manifest.get("codec", {}).get("bundled"):
        print("warning: MOSS is not bundled; it will be downloaded from its upstream repo")
    return manifest, config


def _parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Download a DongLao TTS bundle from Hugging Face and verify that the installed "
            "Python package can load it. Supply all reference/text arguments to also synthesize."
        )
    )
    parser.add_argument("--repo-id", default="DongLao/DongLao-TTS")
    parser.add_argument(
        "--revision",
        default=None,
        help="branch, tag, or commit; default resolves the current repository commit",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="default: CUDA when available, otherwise CPU",
    )
    parser.add_argument("--ref-audio", help="reference WAV/FLAC for an end-to-end test")
    parser.add_argument("--ref-text", help="exact transcript of --ref-audio")
    parser.add_argument("--target-text", help="text to synthesize")
    parser.add_argument("--output", default="hub-smoke-test.wav")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    requested_inference = any((args.ref_audio, args.ref_text, args.target_text))
    if requested_inference and not all((args.ref_audio, args.ref_text, args.target_text)):
        raise SystemExit(
            "--ref-audio, --ref-text, and --target-text must be supplied together"
        )
    if args.ref_audio and not os.path.isfile(args.ref_audio):
        raise SystemExit(f"reference audio does not exist: {args.ref_audio}")
    if args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested, but CUDA is not available")

    info = HfApi().model_info(args.repo_id, revision=args.revision)
    revision = info.sha
    print(f"package: donglao-tts {_package_version()}")
    print(f"model:   {args.repo_id}@{revision}")
    print(f"device:  {device}")
    manifest, _ = _validate_bundle(args.repo_id, revision)
    print(
        "bundle:  format=2, native=safetensors, "
        f"codec_bundled={manifest['codec'].get('bundled', False)}"
    )

    started = time.perf_counter()
    tts = DongLaoTTS.from_pretrained(
        args.repo_id,
        revision=revision,
        device=device,
    )
    print(
        f"loaded:  AR={_parameter_count(tts.ar_model):,} params, "
        f"NAR={_parameter_count(tts.nar_model):,} params, "
        f"codec={tts.codec.__class__.__name__}, tokenizer={tts.tokenizer.get_piece_size()} pieces "
        f"({time.perf_counter() - started:.1f}s)"
    )

    if not requested_inference:
        print("PASS: installed package loaded all native weights and the bundled MOSS codec")
        return

    started = time.perf_counter()
    gen_wav = tts.generate(
        args.target_text,
        reference_audio=args.ref_audio,
        reference_text=args.ref_text,
        output_path=args.output,
        max_frames=args.max_frames,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    output = os.path.abspath(args.output)
    peak = gen_wav.abs().max().item()
    print(
        f"audio:   shape={tuple(gen_wav.shape)}, peak={peak:.4f}, "
        f"saved={output} ({time.perf_counter() - started:.1f}s)"
    )
    print("PASS: Hub download, native model inference, NAR fill, and MOSS decode succeeded")


if __name__ == "__main__":
    main()
