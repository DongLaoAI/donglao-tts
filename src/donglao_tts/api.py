"""High-level inference API for pretrained DongLao TTS bundles."""

import json
import os

import torch
from donglao_g2p import Pipeline
from huggingface_hub import HfApi, hf_hub_download

from donglao_tts.generate import generate_sample
from donglao_tts.hub import load_from_hub
from donglao_tts.utils.precision import resolve_dtype


class DongLaoTTS:
    """A loaded DongLao TTS model ready for repeated reference-voice synthesis."""

    def __init__(
        self,
        *,
        repo_id,
        revision,
        device,
        dtype,
        ar_model,
        nar_model,
        codec,
        tokenizer,
        special_tokens,
        codebook_size,
        num_quantizers,
        pipeline,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.device = torch.device(device)
        self.dtype = dtype
        self.ar_model = ar_model
        self.nar_model = nar_model
        self.codec = codec
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(cls, repo_id="DongLao/DongLao-TTS", *, revision=None, device=None):
        """Download a Hub bundle and return a reusable inference object.

        ``revision`` may be a branch, tag, or commit. It is always resolved to an immutable
        commit before config and weights are downloaded, preventing files from different commits
        from being mixed if a branch changes during startup.
        """
        resolved_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available")

        resolved_revision = HfApi().model_info(repo_id, revision=revision).sha
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=resolved_revision,
        )
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        loaded = load_from_hub(
            repo_id,
            revision=resolved_revision,
            device=resolved_device,
        )
        (
            ar_model,
            nar_model,
            codec,
            tokenizer,
            special_tokens,
            codebook_size,
            num_quantizers,
        ) = loaded
        dtype = resolve_dtype(config["model"].get("precision", "float32"), resolved_device)
        return cls(
            repo_id=repo_id,
            revision=resolved_revision,
            device=resolved_device,
            dtype=dtype,
            ar_model=ar_model,
            nar_model=nar_model,
            codec=codec,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            pipeline=Pipeline(),
        )

    @property
    def sample_rate(self):
        """Output sample rate reported by the bundled audio codec."""
        return self.codec.sampling_rate

    @torch.no_grad()
    def generate(
        self,
        text,
        *,
        reference_audio,
        reference_text,
        output_path=None,
        max_frames=200,
        temperature=1.0,
        top_k=0,
    ):
        """Synthesize ``text`` in the reference voice and return a CPU waveform tensor.

        ``reference_text`` must be the exact transcript of ``reference_audio``. When
        ``output_path`` is provided, the same waveform is also written as an audio file.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError("reference_text must be a non-empty string")
        reference_audio = os.fspath(reference_audio)
        if not os.path.isfile(reference_audio):
            raise FileNotFoundError(f"reference audio does not exist: {reference_audio}")
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        config = {
            "sample": {
                "ref_audio": os.path.abspath(reference_audio),
                "ref_text": reference_text,
                "target_text": text,
                "temperature": temperature,
                "top_k": top_k,
            }
        }
        waveform, _ = generate_sample(
            config,
            self.ar_model,
            self.nar_model,
            self.codec,
            self.tokenizer,
            self.special_tokens,
            self.codebook_size,
            self.num_quantizers,
            self.device,
            self.dtype,
            self.pipeline,
            max_frames=max_frames,
        )
        if waveform is None or waveform.numel() == 0:
            raise RuntimeError("the AR model generated zero audio frames")
        if not torch.isfinite(waveform).all():
            raise RuntimeError("generated waveform contains NaN or infinity")

        if output_path is not None:
            output_path = os.path.abspath(os.fspath(output_path))
            self.codec.save_audio(waveform, output_path)
        return waveform
