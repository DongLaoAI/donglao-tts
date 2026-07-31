"""Save/load/push a trained donglao-tts bundle to/from the Hugging Face Hub.

A complete release bundle contains the native PyTorch/safetensors model, ONNX exports, the
SentencePiece model, and a pinned copy of the MOSS codec (weights, config, custom code and
license). Older bundles that only contain the native model and refer to MOSS by repo id remain
loadable.
"""

import argparse
import json
import os
import shutil
import tempfile

import sentencepiece as spm
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file

from donglao_tts.checkpoint import load_checkpoint
from donglao_tts.config import load_config
from donglao_tts.models.build import build_models
from donglao_tts.models.codec.moss_codec import MossCodec
from donglao_tts.models.embeddings import SpecialTokens, migrate_legacy_ar_state_dict

_BUNDLE_SPM_NAME = "spm.model"
_BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
_BUNDLED_CODEC_DIR = "moss_codec"
_ONNX_DIR = "onnx"


def _export_onnx_bundle(ar_model, nar_model, config, out_dir):
    """Export every ONNX graph currently supported by the configured AR backbone."""
    try:
        from donglao_tts.export.onnx_export import (
            export_ar_decode_step,
            export_ar_prefill,
            export_ar_qwen3_decode_step,
            export_ar_qwen3_prefill,
            export_nar_layer,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional runtime packages
        raise ImportError(
            "ONNX bundle export requires: pip install 'donglao-tts[export]'"
        ) from exc

    onnx_dir = os.path.join(out_dir, _ONNX_DIR)
    os.makedirs(onnx_dir, exist_ok=True)
    d_model = config["model"]["d_model"]
    backbone = config["model"]["ar"].get("backbone", "custom")

    files = ["nar_layer.onnx"]
    export_nar_layer(nar_model, os.path.join(onnx_dir, files[-1]), d_model)
    if backbone == "custom":
        files.extend(["ar_prefill.onnx", "ar_decode_step.onnx"])
        export_ar_prefill(ar_model, os.path.join(onnx_dir, files[-2]), d_model)
        export_ar_decode_step(ar_model, os.path.join(onnx_dir, files[-1]), d_model)
        complete_ar = True
    elif backbone == "qwen3":
        files.extend(["ar_qwen3_prefill.onnx", "ar_qwen3_decode_step.onnx"])
        export_ar_qwen3_prefill(ar_model, os.path.join(onnx_dir, files[-2]), d_model)
        export_ar_qwen3_decode_step(ar_model, os.path.join(onnx_dir, files[-1]), d_model)
        complete_ar = True
    else:
        raise ValueError(f"unknown model.ar.backbone: {backbone!r}")

    return {
        "directory": _ONNX_DIR,
        "backbone": backbone,
        "files": files,
        "complete_ar_generation": complete_ar,
        "opset": 18,
    }


def _bundle_moss_codec(config, out_dir):
    """Copy the exact pinned MOSS snapshot into the release without executing its code."""
    codec_cfg = config["codec"]
    source_repo_id = codec_cfg["repo_id"]
    source_revision = codec_cfg.get("revision")
    source_dir = snapshot_download(source_repo_id, revision=source_revision)
    codec_dir = os.path.join(out_dir, _BUNDLED_CODEC_DIR)
    if os.path.exists(codec_dir):
        shutil.rmtree(codec_dir)
    shutil.copytree(source_dir, codec_dir)
    return {
        "bundled": True,
        "directory": _BUNDLED_CODEC_DIR,
        "source_repo_id": source_repo_id,
        "source_revision": source_revision,
    }


def _write_default_model_card(out_dir, manifest):
    """Create a minimal editable model card without claiming a license for trained weights."""
    path = os.path.join(out_dir, "README.md")
    if os.path.exists(path):
        return
    onnx = manifest.get("onnx")
    onnx_note = "ONNX was not included."
    if onnx:
        onnx_note = (
            f"ONNX graphs are in `{onnx['directory']}/`. Complete AR generation: "
            f"`{str(onnx['complete_ar_generation']).lower()}`."
        )
    codec = manifest.get("codec", {})
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "---\nlibrary_name: donglao-tts\ntags:\n- text-to-speech\n- audio\n- onnx\n"
            "---\n\n# DongLao TTS\n\n"
            "This repository contains a donglao-tts release bundle. Edit this model card before "
            "publishing to document training data, evaluation, intended use, limitations, and "
            "the license of the trained weights.\n\n"
            f"{onnx_note}\n\n"
            f"The bundled MOSS codec comes from `{codec.get('source_repo_id')}` at revision "
            f"`{codec.get('source_revision')}` and retains its upstream files and license.\n"
        )


def save_pretrained_bundle(
    ar_model,
    nar_model,
    config,
    out_dir,
    *,
    include_onnx=False,
    include_codec=False,
):
    """Write a self-contained inference bundle.

    Only the codec, tokenizer, and model sections are published. Training paths, sample text,
    and other local-only configuration must not leak into a shared model bundle.
    """
    os.makedirs(out_dir, exist_ok=True)

    save_file(ar_model.state_dict(), os.path.join(out_dir, "ar_model.safetensors"))
    save_file(nar_model.state_dict(), os.path.join(out_dir, "nar_model.safetensors"))

    shutil.copyfile(config["tokenizer"]["model_path"], os.path.join(out_dir, _BUNDLE_SPM_NAME))

    bundle_config = {
        section: json.loads(json.dumps(config[section]))
        for section in ("codec", "tokenizer", "model")
    }
    bundle_config["tokenizer"]["model_path"] = _BUNDLE_SPM_NAME
    manifest = {
        "format_version": 2,
        "library_name": "donglao-tts",
        "native": {
            "format": "safetensors",
            "files": ["ar_model.safetensors", "nar_model.safetensors"],
        },
        "tokenizer": _BUNDLE_SPM_NAME,
    }

    if include_codec:
        manifest["codec"] = _bundle_moss_codec(config, out_dir)
        bundle_config["codec"]["bundled_path"] = _BUNDLED_CODEC_DIR
    else:
        manifest["codec"] = {
            "bundled": False,
            "source_repo_id": config["codec"]["repo_id"],
            "source_revision": config["codec"].get("revision"),
        }

    if include_onnx:
        manifest["onnx"] = _export_onnx_bundle(ar_model, nar_model, config, out_dir)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(bundle_config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, _BUNDLE_MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    _write_default_model_card(out_dir, manifest)

    return out_dir


def push_to_hub(repo_id, local_dir, private=False, commit_message=None):
    """Create (if needed) and upload `local_dir` (see save_pretrained_bundle) to `repo_id`.
    Requires an authenticated huggingface_hub session (HF_TOKEN env var, or a prior
    `huggingface-cli login`) -- this is intentionally left to the caller's own credentials."""
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=local_dir,
                       commit_message=commit_message or "Upload donglao-tts model bundle")
    return f"https://huggingface.co/{repo_id}"


def load_from_hub(repo_id, revision=None, device=None):
    """Download a bundle (see save_pretrained_bundle) and reconstruct everything needed to
    generate audio. Returns (ar_model, nar_model, codec, sp, special, codebook_size,
    num_quantizers) -- ar_model/nar_model are already .eval()'d."""
    local_dir = snapshot_download(repo_id, revision=revision)

    with open(os.path.join(local_dir, "config.json")) as f:
        cfg = json.load(f)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    spm_path = os.path.join(local_dir, os.path.basename(cfg["tokenizer"]["model_path"]))
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    special = SpecialTokens(sp)
    vocab_size = sp.get_piece_size()

    # Prefer the bundled, pinned codec. Keep repo_id/revision as the fallback for legacy bundles.
    runtime_cfg = json.loads(json.dumps(cfg))
    bundled_codec_path = cfg["codec"].get("bundled_path")
    if bundled_codec_path:
        local_codec_dir = os.path.join(local_dir, bundled_codec_path)
        if not os.path.isdir(local_codec_dir):
            raise FileNotFoundError(
                f"bundle declares codec.bundled_path={bundled_codec_path!r}, but it is missing"
            )
        runtime_cfg["codec"]["repo_id"] = local_codec_dir
        runtime_cfg["codec"]["revision"] = None

    ar_model, nar_model, codebook_size, num_quantizers = build_models(
        runtime_cfg, vocab_size, device
    )

    ar_state = load_file(os.path.join(local_dir, "ar_model.safetensors"), device=str(device))
    ar_state = migrate_legacy_ar_state_dict(ar_state, vocab_size)
    ar_model.load_state_dict(ar_state)
    nar_state = load_file(os.path.join(local_dir, "nar_model.safetensors"), device=str(device))
    nar_model.load_state_dict(nar_state)
    ar_model.eval()
    nar_model.eval()

    codec = MossCodec(
        repo_id=runtime_cfg["codec"]["repo_id"],
        num_quantizers=runtime_cfg["codec"]["num_quantizers"],
        device=str(device),
        revision=runtime_cfg["codec"].get("revision"),
    )

    return ar_model, nar_model, codec, sp, special, codebook_size, num_quantizers


def _push_to_hub_cli():
    """Entry point for the `donglao-push-to-hub` console script: bundles a local train.py
    checkpoint and pushes it. Requires your own HF_TOKEN (or a prior `huggingface-cli login`) --
    this is never run automatically, only when you invoke it yourself."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                         help="training config the checkpoint was produced with")
    parser.add_argument("--checkpoint", required=True,
                         help="path to a donglao-train-saved .pt checkpoint, e.g. run/step_30000.pt")
    parser.add_argument("--repo-id", required=True, help="e.g. myorg/donglao-tts")
    parser.add_argument("--out-dir", default=None,
                         help="local staging directory for the bundle; defaults to a temp dir")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sp = spm.SentencePieceProcessor(model_file=cfg["tokenizer"]["model_path"])
    vocab_size = sp.get_piece_size()
    device = torch.device("cpu")
    ar_model, nar_model, _, _ = build_models(cfg, vocab_size, device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    ar_state = migrate_legacy_ar_state_dict(ckpt["ar_model"], vocab_size)
    ar_model.load_state_dict(ar_state)
    nar_model.load_state_dict(ckpt["nar_model"])

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="donglao_tts_bundle_")
    print("bundling native safetensors, ONNX graphs, and the pinned MOSS codec snapshot")
    save_pretrained_bundle(
        ar_model,
        nar_model,
        cfg,
        out_dir,
        include_onnx=True,
        include_codec=True,
    )
    print(f"bundle staged at {out_dir}")

    url = push_to_hub(args.repo_id, out_dir, private=args.private)
    print(f"pushed to {url}")


if __name__ == "__main__":
    _push_to_hub_cli()
