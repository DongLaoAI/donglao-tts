<div align="center">
  <img
    src="https://raw.githubusercontent.com/DongLaoAI/donglao-tts/main/assets/donglao-tts-logo.png"
    alt="donglao-tts — angular horizontal singing crocodile logo"
    width="720"
  >

  <h1>donglao-tts</h1>
</div>

An AR + NAR text-to-speech toolkit built on residual vector quantization.

`donglao-tts` supports model training, reference-voice speech synthesis, Hugging Face model
bundles, ONNX/OpenVINO export, and quantization experiments. The sample configuration targets
Vietnamese and uses
[MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)
as its audio codec.

> **Project status:** research/pre-1.0. APIs may change between releases. This package does not
> include a pretrained checkpoint, dataset, tokenizer, or production HTTP service.

The current release supports Python 3.10 on Linux x86-64. Model weights are distributed
separately through Hugging Face Hub. FFmpeg 4-8 shared libraries are required for audio I/O.

## Install

```bash
python -m pip install donglao-tts
```

Install an optional runtime:

```bash
python -m pip install "donglao-tts[export]"
python -m pip install "donglao-tts[openvino]"
python -m pip install "donglao-tts[gguf]"
```

With uv:

```bash
uv add donglao-tts
uv add "donglao-tts[export]"
```

Verify that the installed package can download and load the published weights:

```bash
donglao-smoke-test-hub --repo-id DongLao/DongLao-TTS --device cpu
```

## Python API

```python
from donglao_tts import DongLaoTTS

tts = DongLaoTTS.from_pretrained("DongLao/DongLao-TTS")
waveform = tts.generate(
    "Text to synthesize.",
    reference_audio="reference.wav",
    reference_text="Exact transcript of the reference recording.",
    output_path="output.wav",
)
```

Reuse the loaded `tts` object for additional synthesis calls. The method returns the generated
waveform as a CPU PyTorch tensor, whether or not `output_path` is provided.

## Commands

Every model command requires an explicit YAML configuration:

```bash
donglao-init-config configs/local.yaml
donglao-train --config configs/local.yaml --resume none
donglao-infer --config configs/local.yaml --device cuda
donglao-push-to-hub \
  --config configs/local.yaml \
  --checkpoint run/step_120000.pt \
  --repo-id YOUR_ORG/donglao-tts \
  --private
```

Data-preparation commands are included in the package:

```bash
donglao-prepare-dataset --help
donglao-phonemize-manifest --help
donglao-build-phoneme-corpus --help
donglao-build-tokenizer --help
donglao-init-config --help
```

## Security and responsible use

The MOSS codec requires Hugging Face remote code. The sample configuration pins it to an
immutable commit; review upstream changes before updating that revision. PyTorch checkpoints are
loaded with `weights_only=True`, while shared model bundles use `safetensors`.

Use reference voices only with appropriate consent. Protect audio, transcripts, embeddings, and
checkpoints as sensitive data, and do not use the project for impersonation, fraud, harassment,
or bypassing voice authentication.

The source repository contains complete English and Vietnamese documentation, architecture
notes, development instructions, security boundaries, and the project roadmap.
