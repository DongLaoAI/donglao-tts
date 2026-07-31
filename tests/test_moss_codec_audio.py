import soundfile as sf
import torch

from donglao_tts.models.codec.moss_codec import MossCodec


def _audio_only_codec(*, sampling_rate=16_000, num_channels=1):
    codec = object.__new__(MossCodec)
    codec.sampling_rate = sampling_rate
    codec.num_channels = num_channels
    return codec


def test_save_and_load_audio_without_torchcodec(tmp_path):
    codec = _audio_only_codec()
    waveform = torch.linspace(-0.5, 0.5, 320).unsqueeze(0)
    output_path = tmp_path / "roundtrip.wav"

    codec.save_audio(waveform, output_path)
    loaded = codec.load_audio(output_path)

    assert codec.input_channels == 1
    assert loaded.shape == waveform.shape
    torch.testing.assert_close(loaded, waveform, atol=4e-5, rtol=0)


def test_load_audio_downmixes_stereo(tmp_path):
    codec = _audio_only_codec()
    stereo = torch.stack((torch.full((160,), 0.25), torch.full((160,), -0.25)))
    input_path = tmp_path / "stereo.wav"
    sf.write(input_path, stereo.transpose(0, 1).numpy(), codec.sampling_rate)

    loaded = codec.load_audio(input_path)

    assert codec.input_channels == 2
    assert loaded.shape == (1, 160)
    torch.testing.assert_close(loaded, torch.zeros_like(loaded), atol=4e-5, rtol=0)
