import torch
import torch.nn.functional as F

from donglao_tts.models.embeddings import build_input_embeds


def sample_from_logits(logits, temperature, top_k):
    """logits [V]. temperature<=0 means pure greedy (argmax), matching the old deterministic
    behavior exactly. Otherwise temperature-scaled + optional top-k truncation, then multinomial
    sample -- this treats the dedicated EOS class (see ar_model.py) as just another candidate, so
    it can be sampled like any other code instead of needing to literally win an argmax."""
    if temperature <= 0:
        return int(logits.argmax())
    scaled = logits.float() / temperature
    if top_k and top_k > 0:
        k = min(top_k, scaled.shape[-1])
        kth_val = torch.topk(scaled, k).values[-1]
        scaled = torch.where(scaled < kth_val, torch.full_like(scaled, float("-inf")), scaled)
    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, 1))


def make_batch(ref_text_ids, ref_codec, target_text_ids, target_codec, device):
    return {
        "ref_text_ids": ref_text_ids.unsqueeze(0).to(device),
        "ref_text_len": torch.tensor([ref_text_ids.shape[0]], device=device),
        "ref_codec": ref_codec.unsqueeze(0).to(device),
        "ref_codec_len": torch.tensor([ref_codec.shape[0]], device=device),
        "target_text_ids": target_text_ids.unsqueeze(0).to(device),
        "target_text_len": torch.tensor([target_text_ids.shape[0]], device=device),
        "target_codec": target_codec.unsqueeze(0).to(device),
        "target_codec_len": torch.tensor([target_codec.shape[0]], device=device),
    }


def build_sample_from_config(sample_cfg, codec, sp, pipeline):
    """Builds (ref_text_ids, ref_codec, target_text_ids) from a fixed ref_audio/ref_text/target_text
    triple specified in config -- used both by infer.py and by train.py's periodic checkpoint
    sampling, so the same example is tracked across a whole training run."""
    ref_text = sample_cfg["ref_text"]
    target_text = sample_cfg["target_text"]
    ref_phoneme, target_phoneme = pipeline.phonemize_batch([ref_text, target_text])
    ref_text_ids = torch.tensor(sp.encode(ref_phoneme, out_type=int), dtype=torch.long)
    target_text_ids = torch.tensor(sp.encode(target_phoneme, out_type=int), dtype=torch.long)
    ref_codec_kt = codec.encode_file(sample_cfg["ref_audio"])  # [n_q, T]
    ref_codec = ref_codec_kt.transpose(0, 1).cpu()  # [T, n_q]
    return ref_text_ids, ref_codec, target_text_ids


@torch.no_grad()
def ar_generate_rvq0(ar_model, special, codebook_size, ref_text_ids, ref_codec,
                      target_text_ids, device, dtype, max_frames=200, temperature=1.0, top_k=0):
    """AR generation using KV-cache: one prefill pass over the prompt (everything up to and
    including [CODE_TARGET]), then one incremental step per generated frame -- O(T) total instead
    of re-running the whole growing sequence at every step.

    The AR head's classification space is just [0, codebook_size) for codec ids plus one
    dedicated EOS class at index `codebook_size` (see ar_model.py) -- decoupled from the text/
    special SentencePiece vocab entirely.

    Sampling (temperature/top_k, see sample_from_logits) is used instead of pure argmax by
    default: greedy decoding on an imperfectly-trained AR is prone to a self-reinforcing repeat
    loop (the model gets increasingly confident in its own most-recent code once a short run of
    repeats appears in its own KV-cache context, which it never saw during teacher-forced
    training, so it never escapes and never reaches the EOS probability mass either) -- this was
    observed directly at step 30k: 20 reasonably varied frames, then a lock into a single
    repeated code for the rest of max_frames, decoding to near-silence. Pass temperature<=0 for
    the old deterministic argmax behavior.

    Also collects the AR's own per-frame hidden state as it generates (the same `hidden` the NAR
    conditions on during training, see `extract_target_hidden`) so the NAR can reuse it directly
    afterward instead of recomputing ref/text context from scratch. Returns (rvq0_codes,
    ar_hidden[1,T,d_model])."""
    empty_target_codec = torch.zeros(0, ref_codec.shape[1], dtype=torch.long)
    batch = make_batch(ref_text_ids, ref_codec, target_text_ids, empty_target_codec, device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        input_embeds, _, pad_mask, _ = build_input_embeds(ar_model.embed, special, batch)
        logits, _, cache = ar_model(input_embeds, padding_mask=pad_mask, use_cache=True)

    next_id = sample_from_logits(logits[0, -1], temperature, top_k)
    generated = []
    hidden_states = []
    for _ in range(max_frames):
        if next_id == codebook_size:  # dedicated EOS class
            break
        code = next_id
        generated.append(code)

        code_tensor = torch.tensor([code], device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            step_embed = ar_model.embed.embed_codec_layer(code_tensor, 0).unsqueeze(0)  # [1,1,d]
            logits, hidden, cache = ar_model(step_embed, past_key_values=cache, use_cache=True)
        hidden_states.append(hidden[0, 0])  # [d_model], this frame's own hidden state
        next_id = sample_from_logits(logits[0, -1], temperature, top_k)

    if hidden_states:
        ar_hidden = torch.stack(hidden_states, dim=0).unsqueeze(0)  # [1, T, d_model]
    else:
        d_model = ar_model.embed.text_table.weight.shape[1]
        ar_hidden = torch.zeros(1, 0, d_model, device=device)

    return generated, ar_hidden


@torch.no_grad()
def nar_fill_layers(nar_model, ar_hidden, rvq0_codes, num_quantizers, device, dtype):
    # ar_hidden [1, T, d_model] -- the AR's own per-frame hidden state (from ar_generate_rvq0 during
    # inference, or extract_target_hidden(...) during training), NOT recomputed from ref_codec here.
    T = len(rvq0_codes)
    target_codec = torch.zeros(1, T, num_quantizers, dtype=torch.long, device=device)
    target_codec[0, :, 0] = torch.tensor(rvq0_codes, dtype=torch.long, device=device)
    pad_mask = torch.zeros(1, T, dtype=torch.bool, device=device)

    for k in range(1, num_quantizers):
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = nar_model(ar_hidden, target_codec[..., :k], k, pad_mask)
        target_codec[0, :, k] = logits.argmax(dim=-1)[0]

    return target_codec[0]  # [T, num_quantizers]


def generate_sample(cfg, ar_model, nar_model, codec, sp, special, codebook_size,
                     num_quantizers, device, dtype, pipeline, max_frames=200):
    """End-to-end: config sample -> AR (RVQ0 + hidden states) -> NAR (RVQ1..K-1, conditioned on the
    AR's own hidden states) -> decoded waveform. Caller is responsible for ar_model.eval()/
    nar_model.eval() (and restoring .train() after). Returns (gen_wav, ref_wav) -- ref_wav is the
    ref-codec decoded through the same codec path, so it's directly comparable to gen_wav (same
    sample rate/channels), not just a copy of the raw ref_audio file. Returns (None, ref_wav) if the
    AR produced 0 frames."""
    ref_text_ids, ref_codec, target_text_ids = build_sample_from_config(
        cfg["sample"], codec, sp, pipeline)
    ref_wav = codec.decode(ref_codec.transpose(0, 1))  # [T,n_q] -> [n_q,T]

    sample_cfg = cfg["sample"]
    rvq0_codes, ar_hidden = ar_generate_rvq0(ar_model, special, codebook_size,
                                              ref_text_ids, ref_codec, target_text_ids, device,
                                              dtype, max_frames=max_frames,
                                              temperature=sample_cfg.get("temperature", 1.0),
                                              top_k=sample_cfg.get("top_k", 0))
    if len(rvq0_codes) == 0:
        return None, ref_wav

    full_codes = nar_fill_layers(nar_model, ar_hidden, rvq0_codes, num_quantizers, device, dtype)
    codes_for_decode = full_codes.transpose(0, 1)  # [T, n_q] -> [n_q, T]

    gen_wav = codec.decode(codes_for_decode)
    return gen_wav, ref_wav
