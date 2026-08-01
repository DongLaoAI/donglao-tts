"""High-level inference API for pretrained DongLao TTS bundles."""

import json
import os

import torch
from donglao_g2p import Pipeline
from huggingface_hub import HfApi, hf_hub_download

from donglao_tts.generate import generate_batch_samples, generate_sample, generate_sample_stream
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

    @staticmethod
    def _validate_options(reference_audio, reference_text, max_frames, temperature, top_k):
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
        return os.path.abspath(reference_audio)

    def _generation_args(self):
        return (
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
        )

    @staticmethod
    def _validate_waveform(waveform, message="the AR model generated zero audio frames"):
        if waveform is None or waveform.numel() == 0:
            raise RuntimeError(message)
        if not torch.isfinite(waveform).all():
            raise RuntimeError("generated waveform contains NaN or infinity")

    @torch.no_grad()
    def generate(
        self,
        text,
        *,
        reference_audio,
        reference_text,
        output_path=None,
        max_frames=200,
        temperature=0.8,
        top_k=10,
    ):
        """Synthesize ``text`` in the reference voice and return a CPU waveform tensor.

        ``reference_text`` must be the exact transcript of ``reference_audio``. The target is
        phonemized, split on periods, synthesized sentence by sentence, and concatenated. When
        ``output_path`` is provided, the same waveform is also written as an audio file.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        reference_audio = self._validate_options(
            reference_audio, reference_text, max_frames, temperature, top_k
        )

        config = {
            "sample": {
                "ref_audio": reference_audio,
                "ref_text": reference_text,
                "target_text": text,
                "temperature": temperature,
                "top_k": top_k,
            }
        }
        waveform, _ = generate_sample(
            config,
            *self._generation_args(),
            max_frames=max_frames,
        )
        self._validate_waveform(waveform)

        if output_path is not None:
            output_path = os.path.abspath(os.fspath(output_path))
            self.codec.save_audio(waveform, output_path)
        return waveform

    @torch.no_grad()
    def generate_batch(
        self,
        texts,
        *,
        reference_audio,
        reference_text,
        output_paths=None,
        max_frames=200,
        temperature=0.8,
        top_k=10,
    ):
        """Synthesize multiple texts with one shared reference and return waveform tensors.

        G2P and reference-audio encoding are shared across the call. Each target is still decoded
        independently so a target may contain any number of period-delimited sentences.
        """
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of strings, not a single string")
        texts = list(texts)
        if not texts:
            raise ValueError("texts must contain at least one item")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("every item in texts must be a non-empty string")
        reference_audio = self._validate_options(
            reference_audio, reference_text, max_frames, temperature, top_k
        )

        if output_paths is None:
            output_paths = [None] * len(texts)
        else:
            if isinstance(output_paths, (str, bytes, os.PathLike)):
                raise TypeError("output_paths must be an iterable of paths")
            output_paths = list(output_paths)
            if len(output_paths) != len(texts):
                raise ValueError("output_paths must have the same length as texts")

        config = {
            "sample": {
                "ref_audio": reference_audio,
                "ref_text": reference_text,
                "temperature": temperature,
                "top_k": top_k,
            }
        }
        waveforms, _ = generate_batch_samples(
            config,
            texts,
            *self._generation_args(),
            max_frames=max_frames,
        )
        for index, (waveform, output_path) in enumerate(zip(waveforms, output_paths)):
            self._validate_waveform(
                waveform, f"the AR model generated zero audio frames for batch item {index}"
            )
            if output_path is not None:
                self.codec.save_audio(waveform, os.path.abspath(os.fspath(output_path)))
        return waveforms

    def generate_stream(
        self,
        text,
        *,
        reference_audio,
        reference_text,
        max_frames=200,
        temperature=0.8,
        top_k=10,
        chunk_frames=5,
    ):
        """Stream audio decoded from consecutive groups of generated RVQ frames.

        The AR KV-cache is preserved while each group of ``chunk_frames`` RVQ0 tokens is completed
        by the NAR model and decoded. The final group may contain fewer frames.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(chunk_frames, int) or isinstance(chunk_frames, bool) or chunk_frames < 1:
            raise ValueError("chunk_frames must be a positive integer")
        reference_audio = self._validate_options(
            reference_audio, reference_text, max_frames, temperature, top_k
        )
        config = {
            "sample": {
                "ref_audio": reference_audio,
                "ref_text": reference_text,
                "target_text": text,
                "temperature": temperature,
                "top_k": top_k,
            }
        }
        stream = generate_sample_stream(
            config,
            *self._generation_args(),
            max_frames=max_frames,
            chunk_frames=chunk_frames,
        )

        def validated_stream():
            while True:
                try:
                    with torch.no_grad():
                        waveform = next(stream)
                except StopIteration:
                    return
                self._validate_waveform(waveform)
                yield waveform

        return validated_stream()
