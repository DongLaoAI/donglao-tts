<div align="center">
  <img src="https://raw.githubusercontent.com/DongLaoAI/donglao-tts/main/assets/donglao-tts-logo.png" alt="donglao-tts — angular horizontal singing crocodile logo" width="432" />

  <h1>donglao-tts</h1>

  <p><strong>Reference-voice text-to-speech with a simple Python API.</strong></p>
</div>

> Research/pre-1.0 release. APIs may change between versions. Use reference voices only with the
> speaker's permission.

## Recommended Python API

```bash
uv add donglao-tts

# Or with pip
python -m pip install donglao-tts
```

```python
from donglao_tts import DongLaoTTS

tts = DongLaoTTS.from_pretrained("DongLao/DongLao-TTS")

waveform = tts.generate(
    "Xin chào mọi người, đây là Đông Lào TTS.",
    reference_audio="reference.wav",
    reference_text="Exact transcript of the speech in reference.wav.",
    output_path="output.wav",
)

print(tts.revision)
print(tts.sample_rate)
print(tuple(waveform.shape))
```

The first call downloads the model, tokenizer, and bundled MOSS codec. CUDA is selected when
available; otherwise the model runs on CPU. Reuse the loaded `tts` object for subsequent calls.

For reproducible use, pin a tested model commit:

```python
tts = DongLaoTTS.from_pretrained(
    "DongLao/DongLao-TTS",
    revision="6ba3003ccb8d938c2a725a4117084492909c9419",
    device="cuda",
)
```

Generation controls are available when needed:

```python
waveform = tts.generate(
    "Text to synthesize.",
    reference_audio="reference.wav",
    reference_text="Exact reference transcript.",
    output_path="output.wav",  # optional
    max_frames=200,
    temperature=0.8,
    top_k=10,
)
```

After G2P conversion, the target is split on periods, synthesized sentence by sentence, and the
resulting audio is concatenated in order. The returned waveform is a CPU PyTorch tensor with shape
`[channels, samples]`.

Multiple targets sharing the same reference can reuse its preprocessing with `generate_batch()`:

```python
waveforms = tts.generate_batch(
    ["First target.", "Second target."],
    reference_audio="reference.wav",
    reference_text="Exact reference transcript.",
)
```

`generate_stream()` yields audio decoded from consecutive RVQ-token groups:

```python
for audio_chunk in tts.generate_stream(
    "First sentence. Second sentence.",
    reference_audio="reference.wav",
    reference_text="Exact reference transcript.",
    chunk_frames=5,
):
    consume(audio_chunk)
```

The AR KV-cache is preserved between groups. NAR completes the remaining RVQ layers and MOSS
decodes each group while AR generation continues. At a 25 Hz codec rate, five frames represent
approximately 200 ms of audio.

## Runtime requirements

- Python `>=3.10,<3.11`
- Linux x86-64
- PyTorch and TorchAudio `>=2.8.0,<3`
- CUDA-capable GPU recommended; CPU inference is supported

## Reference audio

Use a clean, consented recording containing one speaker. `reference_text` must match the spoken
content exactly. Avoid music, overlapping speakers, clipping, long silence, and heavy reverb.

## Verify the installation

```bash
donglao-smoke-test-hub \
  --repo-id DongLao/DongLao-TTS \
  --device cpu
```

For an end-to-end test, also pass `--ref-audio`, `--ref-text`, `--target-text`, and `--output`.

## Responsible use

Obtain consent before using a voice, disclose synthetic audio when appropriate, and protect
reference recordings and transcripts. Do not use the package for impersonation, fraud,
harassment, or bypassing voice authentication.

## Links

- [Source and full documentation](https://github.com/DongLaoAI/donglao-tts)
- [Published model](https://huggingface.co/DongLao/DongLao-TTS)
- [Issue tracker](https://github.com/DongLaoAI/donglao-tts/issues)
- [Apache License 2.0](https://github.com/DongLaoAI/donglao-tts/blob/main/LICENSE)
