import os

import torch
import torchaudio
import yaml
from transformers import AutoModel


class MossCodec:
    def __init__(self, repo_id, num_quantizers=8, device=None, revision=None):
        self.device = device if device and torch.cuda.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = AutoModel.from_pretrained(
            repo_id,
            revision=revision,
            trust_remote_code=True,
        ).to(self.device).eval()
        self.num_quantizers = num_quantizers
        self.sampling_rate = self.model.config.sampling_rate
        self.num_channels = self.model.config.number_channels

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        codec_cfg = cfg["codec"]
        return cls(
            repo_id=codec_cfg["repo_id"],
            num_quantizers=codec_cfg.get("num_quantizers", 8),
            device=codec_cfg.get("device"),
            revision=codec_cfg.get("revision"),
        )

    def load_audio(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        self.input_channels = wav.shape[0]
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)  # downmix to mono
        if wav.shape[0] != self.num_channels:
            wav = wav.repeat(self.num_channels, 1)
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        return wav

    @torch.no_grad()
    def encode(self, wav):
        wav = wav.unsqueeze(0).to(self.device)  # [C,T] -> [1,C,T]
        out = self.model.encode(wav, num_quantizers=self.num_quantizers)
        length = int(out.audio_codes_lengths[0].item())
        return out.audio_codes[:, 0, :length]  # [n_q,T]

    def encode_file(self, audio_path):
        return self.encode(self.load_audio(audio_path))

    @torch.no_grad()
    def decode(self, codes):
        codes = codes.unsqueeze(1).to(self.device)  # [n_q,T] -> [n_q,1,T]
        out = self.model.decode(codes, padding_mask=None)
        return out.audio[0].cpu()  # [C,T_samples]

    def save_audio(self, wav, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchaudio.save(path, wav, self.sampling_rate)
