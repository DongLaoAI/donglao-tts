import torch

import donglao_tts.generate as generate


class _Pipeline:
    def __init__(self):
        self.inputs = None

    def phonemize_batch(self, texts):
        self.inputs = texts
        return "ref phones.", "first phones. second phones.."


class _Tokenizer:
    def __init__(self):
        self.encoded = []

    def encode(self, text, out_type):
        assert out_type is int
        self.encoded.append(text)
        return [len(self.encoded)]


class _Codec:
    def __init__(self):
        self.decode_calls = 0
        self.encode_calls = 0

    def encode_file(self, path):
        assert path == "reference.wav"
        self.encode_calls += 1
        return torch.zeros(2, 3, dtype=torch.long)

    def decode(self, codes):
        self.decode_calls += 1
        waveforms = (
            torch.tensor([[0.0]]),
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 4.0]]),
        )
        return waveforms[self.decode_calls - 1]


def test_generate_sample_splits_phonemes_and_concatenates_audio(monkeypatch):
    pipeline = _Pipeline()
    tokenizer = _Tokenizer()
    codec = _Codec()
    ar_calls = []

    def fake_ar(*args, **kwargs):
        ar_calls.append((args[5].tolist(), kwargs))
        return [1], torch.zeros(1, 1, 2)

    def fake_nar(*args, **kwargs):
        return torch.zeros(1, 2, dtype=torch.long)

    monkeypatch.setattr(generate, "ar_generate_rvq0", fake_ar)
    monkeypatch.setattr(generate, "nar_fill_layers", fake_nar)

    waveform, reference = generate.generate_sample(
        {
            "sample": {
                "ref_audio": "reference.wav",
                "ref_text": "reference text",
                "target_text": "first. second.",
            }
        },
        object(),
        object(),
        codec,
        tokenizer,
        object(),
        10,
        2,
        torch.device("cpu"),
        torch.float32,
        pipeline,
    )

    assert pipeline.inputs == ["reference text", "first. second."]
    assert tokenizer.encoded == ["ref phones.", "first phones", "second phones"]
    assert [call[0] for call in ar_calls] == [[2], [3]]
    assert all(call[1]["temperature"] == 0.8 for call in ar_calls)
    assert all(call[1]["top_k"] == 10 for call in ar_calls)
    assert torch.equal(waveform, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.equal(reference, torch.tensor([[0.0]]))


def test_generate_batch_encodes_reference_once(monkeypatch):
    class BatchPipeline:
        def phonemize_batch(self, texts):
            assert texts == ["reference", "first", "second"]
            return ["ref phones.", "first phones.", "second a. second b."]

    class BatchCodec(_Codec):
        def decode(self, codes):
            self.decode_calls += 1
            return torch.tensor([[float(self.decode_calls)]])

    codec = BatchCodec()
    tokenizer = _Tokenizer()

    monkeypatch.setattr(
        generate,
        "ar_generate_rvq0",
        lambda *args, **kwargs: ([1], torch.zeros(1, 1, 2)),
    )
    monkeypatch.setattr(
        generate,
        "nar_fill_layers",
        lambda *args, **kwargs: torch.zeros(1, 2, dtype=torch.long),
    )

    waveforms, reference = generate.generate_batch_samples(
        {
            "sample": {
                "ref_audio": "reference.wav",
                "ref_text": "reference",
                "temperature": 0.8,
                "top_k": 10,
            }
        },
        ["first", "second"],
        object(),
        object(),
        codec,
        tokenizer,
        object(),
        10,
        2,
        torch.device("cpu"),
        torch.float32,
        BatchPipeline(),
    )

    assert codec.encode_calls == 1
    assert torch.equal(reference, torch.tensor([[1.0]]))
    assert torch.equal(waveforms[0], torch.tensor([[2.0]]))
    assert torch.equal(waveforms[1], torch.tensor([[3.0, 4.0]]))


def test_ar_stream_preserves_cache_and_yields_frame_groups(monkeypatch):
    class Embed:
        text_table = torch.nn.Embedding(2, 3)

        def embed_codec_layer(self, code, layer):
            assert layer == 0
            return torch.ones(code.shape[0], 3)

    class Model:
        embed = Embed()

        def __init__(self):
            self.caches = []

        def __call__(self, input_embeds, padding_mask=None, past_key_values=None, use_cache=False):
            assert use_cache
            self.caches.append(past_key_values)
            cache = len(self.caches)
            logits = torch.zeros(1, input_embeds.shape[1], 8)
            hidden = torch.full((1, input_embeds.shape[1], 3), float(cache))
            return logits, hidden, cache

    sampled = iter([1, 2, 3, 4, 5, 6, 7])
    monkeypatch.setattr(generate, "sample_from_logits", lambda *args: next(sampled))
    monkeypatch.setattr(
        generate,
        "build_input_embeds",
        lambda *args: (
            torch.zeros(1, 2, 3),
            None,
            torch.zeros(1, 2, dtype=torch.bool),
            None,
        ),
    )
    model = Model()

    chunks = list(generate.ar_generate_rvq0_stream(
        model,
        object(),
        7,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(2, 2, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.device("cpu"),
        torch.float32,
        max_frames=10,
        chunk_frames=5,
    ))

    assert [codes for codes, _ in chunks] == [[1, 2, 3, 4, 5], [6]]
    assert chunks[0][1].shape == (1, 5, 3)
    assert chunks[1][1].shape == (1, 1, 3)
    assert model.caches == [None, 1, 2, 3, 4, 5, 6]
