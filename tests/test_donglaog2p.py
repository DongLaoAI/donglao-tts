from donglao_g2p import Pipeline


def test_pipeline_returns_nonempty_phonemes():
    phonemes = Pipeline().phonemize("Xin chào! PyTorch")

    assert isinstance(phonemes, str)
    assert phonemes.strip()
