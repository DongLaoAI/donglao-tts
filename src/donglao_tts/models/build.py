from transformers import AutoConfig

from donglao_tts.models.ar_model import ARTransformerLM
from donglao_tts.models.ar_qwen3 import ARQwen3LM
from donglao_tts.models.nar_model import NARLayerPredictor


def build_models(cfg, vocab_size, device):
    """Construct (ar_model, nar_model) from a loaded config dict. Shared by the training CLI,
    the inference CLI, and hub.py's load_from_hub -- the one place that knows how config.model/
    config.codec map onto real module instances."""
    model_cfg = cfg["model"]
    codec_cfg = cfg["codec"]
    codec_hf_cfg = AutoConfig.from_pretrained(
        codec_cfg["repo_id"],
        revision=codec_cfg.get("revision"),
        trust_remote_code=True,
    )
    codebook_size = codec_hf_cfg.quantizer_kwargs["codebook_size"]
    num_quantizers = cfg["codec"]["num_quantizers"]

    ar_cfg = model_cfg["ar"]
    ref_num_quantizers = ar_cfg.get("ref_num_quantizers", num_quantizers)
    backbone = ar_cfg.get("backbone", "custom")
    if backbone == "qwen3":
        ar_model = ARQwen3LM(
            vocab_size=vocab_size, codebook_size=codebook_size,
            ref_num_quantizers=ref_num_quantizers,
            d_model=model_cfg["d_model"], n_layers=ar_cfg["n_layers"], n_heads=ar_cfg["n_heads"],
            ffn_dim=ar_cfg["ffn_dim"], dropout=ar_cfg["dropout"], rope_theta=model_cfg["rope_theta"],
            n_kv_heads=ar_cfg.get("n_kv_heads"),
        ).to(device)
    elif backbone == "custom":
        ar_model = ARTransformerLM(
            vocab_size=vocab_size, codebook_size=codebook_size,
            ref_num_quantizers=ref_num_quantizers,
            d_model=model_cfg["d_model"], n_layers=ar_cfg["n_layers"],
            n_heads=ar_cfg["n_heads"], ffn_dim=ar_cfg["ffn_dim"],
            dropout=ar_cfg["dropout"], rope_theta=model_cfg["rope_theta"],
        ).to(device)
    else:
        raise ValueError(f"unknown model.ar.backbone: {backbone!r} (expected 'custom' or 'qwen3')")

    nar_model = NARLayerPredictor(
        codebook_size=codebook_size, num_quantizers=num_quantizers, d_model=model_cfg["d_model"],
        n_layers=model_cfg["nar"]["n_layers"], n_heads=model_cfg["nar"]["n_heads"],
        ffn_dim=model_cfg["nar"]["ffn_dim"], dropout=model_cfg["nar"]["dropout"],
        rope_theta=model_cfg["rope_theta"],
    ).to(device)

    return ar_model, nar_model, codebook_size, num_quantizers
