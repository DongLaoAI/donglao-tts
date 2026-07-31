import json
import os

import torch
from safetensors.torch import save_file

import donglao_tts.hub as hub


def _minimal_config(spm_path):
    return {
        "tokenizer": {"model_path": str(spm_path)},
        "codec": {
            "repo_id": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
            "revision": "abc123",
            "num_quantizers": 8,
            "device": "cuda",
        },
        "model": {"d_model": 8, "ar": {"backbone": "qwen3"}},
    }


def test_complete_bundle_manifest_and_layout(tmp_path, monkeypatch):
    spm_path = tmp_path / "source.model"
    spm_path.write_bytes(b"sentencepiece")
    out_dir = tmp_path / "bundle"
    cfg = _minimal_config(spm_path)
    cfg["train"] = {"datasets": ["/private/training-data"]}
    cfg["sample"] = {"ref_text": "private reference transcript"}

    def fake_codec_bundle(config, destination):
        codec_dir = os.path.join(destination, "moss_codec")
        os.makedirs(codec_dir)
        with open(os.path.join(codec_dir, "LICENSE"), "w", encoding="utf-8") as f:
            f.write("Apache-2.0")
        return {
            "bundled": True,
            "directory": "moss_codec",
            "source_repo_id": config["codec"]["repo_id"],
            "source_revision": config["codec"]["revision"],
        }

    def fake_onnx_bundle(ar_model, nar_model, config, destination):
        onnx_dir = os.path.join(destination, "onnx")
        os.makedirs(onnx_dir)
        for filename in ("nar_layer.onnx", "ar_qwen3_prefill.onnx", "ar_qwen3_decode_step.onnx"):
            with open(os.path.join(onnx_dir, filename), "wb") as f:
                f.write(b"onnx")
        return {
            "directory": "onnx",
            "backbone": "qwen3",
            "files": [
                "nar_layer.onnx",
                "ar_qwen3_prefill.onnx",
                "ar_qwen3_decode_step.onnx",
            ],
            "complete_ar_generation": True,
            "opset": 18,
        }

    monkeypatch.setattr(hub, "_bundle_moss_codec", fake_codec_bundle)
    monkeypatch.setattr(hub, "_export_onnx_bundle", fake_onnx_bundle)
    hub.save_pretrained_bundle(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2),
        cfg,
        out_dir,
        include_onnx=True,
        include_codec=True,
    )

    with open(out_dir / "config.json", encoding="utf-8") as f:
        bundled_cfg = json.load(f)
    with open(out_dir / "bundle_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    assert bundled_cfg["tokenizer"]["model_path"] == "spm.model"
    assert bundled_cfg["codec"]["bundled_path"] == "moss_codec"
    assert "train" not in bundled_cfg
    assert "sample" not in bundled_cfg
    assert manifest["codec"]["source_revision"] == "abc123"
    assert manifest["onnx"]["complete_ar_generation"] is True
    assert (out_dir / "ar_model.safetensors").is_file()
    assert (out_dir / "nar_model.safetensors").is_file()
    assert (out_dir / "moss_codec" / "LICENSE").is_file()
    assert (out_dir / "onnx" / "nar_layer.onnx").is_file()
    assert (out_dir / "onnx" / "ar_qwen3_decode_step.onnx").is_file()
    assert (out_dir / "README.md").is_file()


def test_load_from_hub_prefers_bundled_codec_and_requested_device(tmp_path, monkeypatch):
    bundle = tmp_path / "downloaded"
    codec_dir = bundle / "moss_codec"
    codec_dir.mkdir(parents=True)
    (bundle / "spm.model").write_bytes(b"fake")
    cfg = _minimal_config(bundle / "spm.model")
    cfg["tokenizer"]["model_path"] = "spm.model"
    cfg["codec"]["bundled_path"] = "moss_codec"
    with open(bundle / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    ar_model = torch.nn.Linear(2, 2)
    nar_model = torch.nn.Linear(2, 2)
    save_file(ar_model.state_dict(), bundle / "ar_model.safetensors")
    save_file(nar_model.state_dict(), bundle / "nar_model.safetensors")

    class FakeSentencePiece:
        def __init__(self, model_file):
            self.model_file = model_file

        def get_piece_size(self):
            return 10

    codec_call = {}

    def fake_codec(**kwargs):
        codec_call.update(kwargs)
        return kwargs

    def fake_build_models(runtime_cfg, vocab_size, device):
        assert runtime_cfg["codec"]["repo_id"] == str(codec_dir)
        assert runtime_cfg["codec"]["revision"] is None
        return ar_model, nar_model, 1024, 8

    monkeypatch.setattr(hub, "snapshot_download", lambda *args, **kwargs: str(bundle))
    monkeypatch.setattr(hub.spm, "SentencePieceProcessor", FakeSentencePiece)
    monkeypatch.setattr(hub, "SpecialTokens", lambda sp: object())
    monkeypatch.setattr(hub, "build_models", fake_build_models)
    monkeypatch.setattr(hub, "MossCodec", fake_codec)

    result = hub.load_from_hub("DongLao/DongLao-TTS", device="cpu")

    assert result[2]["repo_id"] == str(codec_dir)
    assert codec_call["revision"] is None
    assert codec_call["device"] == "cpu"
