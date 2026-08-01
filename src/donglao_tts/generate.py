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


def _encode_phonemes(phonemes, sp):
    return torch.tensor(sp.encode(phonemes, out_type=int), dtype=torch.long)


def _split_phoneme_sentences(target_phonemes, sp):
    target_sentences = [sentence.strip() for sentence in target_phonemes.split(".")]
    target_sentences = [sentence for sentence in target_sentences if sentence]
    if not target_sentences:
        raise ValueError("target text produced no non-empty sentences after phonemization")
    return [_encode_phonemes(sentence, sp) for sentence in target_sentences]


def build_batch_samples_from_config(sample_cfg, target_texts, codec, sp, pipeline):
    """Encode one shared reference and sentence-token groups for multiple target texts."""
    phonemes = pipeline.phonemize_batch([sample_cfg["ref_text"], *target_texts])
    ref_phonemes, target_phonemes = phonemes[0], phonemes[1:]

    ref_text_ids = _encode_phonemes(ref_phonemes, sp)
    target_text_ids = [
        _split_phoneme_sentences(phoneme_text, sp) for phoneme_text in target_phonemes
    ]
    ref_codec_kt = codec.encode_file(sample_cfg["ref_audio"])  # [n_q, T]
    ref_codec = ref_codec_kt.transpose(0, 1).cpu()  # [T, n_q]
    return ref_text_ids, ref_codec, target_text_ids


def build_samples_from_config(sample_cfg, codec, sp, pipeline):
    """Build the shared reference and one target-token tensor per phonemized sentence."""
    ref_text_ids, ref_codec, target_groups = build_batch_samples_from_config(
        sample_cfg, [sample_cfg["target_text"]], codec, sp, pipeline
    )
    return ref_text_ids, ref_codec, target_groups[0]


def build_sample_from_config(sample_cfg, codec, sp, pipeline):
    """Builds (ref_text_ids, ref_codec, target_text_ids) from a fixed ref_audio/ref_text/target_text
    triple specified in config -- used both by infer.py and by train.py's periodic checkpoint
    sampling, so the same example is tracked across a whole training run."""
    ref_phonemes, target_phonemes = pipeline.phonemize_batch(
        [sample_cfg["ref_text"], sample_cfg["target_text"]]
    )
    ref_text_ids = _encode_phonemes(ref_phonemes, sp)
    target_text_ids = _encode_phonemes(target_phonemes, sp)
    ref_codec_kt = codec.encode_file(sample_cfg["ref_audio"])  # [n_q, T]
    ref_codec = ref_codec_kt.transpose(0, 1).cpu()  # [T, n_q]
    return ref_text_ids, ref_codec, target_text_ids


@torch.no_grad()
def ar_generate_rvq0(ar_model, special, codebook_size, ref_text_ids, ref_codec,
                      target_text_ids, device, dtype, max_frames=200, temperature=0.8, top_k=10):
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


def ar_generate_rvq0_stream(ar_model, special, codebook_size, ref_text_ids, ref_codec,
                            target_text_ids, device, dtype, max_frames=200, temperature=0.8,
                            top_k=10, chunk_frames=5):
    """Yield RVQ0 codes and matching AR hidden states while preserving the AR KV-cache."""
    empty_target_codec = torch.zeros(0, ref_codec.shape[1], dtype=torch.long)
    batch = make_batch(ref_text_ids, ref_codec, target_text_ids, empty_target_codec, device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
        input_embeds, _, pad_mask, _ = build_input_embeds(ar_model.embed, special, batch)
        logits, _, cache = ar_model(input_embeds, padding_mask=pad_mask, use_cache=True)

    next_id = sample_from_logits(logits[0, -1], temperature, top_k)
    chunk_codes = []
    chunk_hidden = []
    for _ in range(max_frames):
        if next_id == codebook_size:
            break

        code = next_id
        code_tensor = torch.tensor([code], device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            step_embed = ar_model.embed.embed_codec_layer(code_tensor, 0).unsqueeze(0)
            logits, hidden, cache = ar_model(
                step_embed, past_key_values=cache, use_cache=True
            )
        chunk_codes.append(code)
        chunk_hidden.append(hidden[0, 0])
        next_id = sample_from_logits(logits[0, -1], temperature, top_k)

        if len(chunk_codes) == chunk_frames:
            yield chunk_codes, torch.stack(chunk_hidden, dim=0).unsqueeze(0)
            chunk_codes = []
            chunk_hidden = []

    if chunk_codes:
        yield chunk_codes, torch.stack(chunk_hidden, dim=0).unsqueeze(0)


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


def _generate_sentence_waveforms(sample_cfg, target_text_ids_by_sentence, ref_text_ids,
                                  ref_codec, ar_model, nar_model, codec, special, codebook_size,
                                  num_quantizers, device, dtype, max_frames):
    for target_text_ids in target_text_ids_by_sentence:
        rvq0_codes, ar_hidden = ar_generate_rvq0(
            ar_model, special, codebook_size, ref_text_ids, ref_codec, target_text_ids,
            device, dtype, max_frames=max_frames,
            temperature=sample_cfg.get("temperature", 0.8),
            top_k=sample_cfg.get("top_k", 10),
        )
        if len(rvq0_codes) == 0:
            return

        full_codes = nar_fill_layers(
            nar_model, ar_hidden, rvq0_codes, num_quantizers, device, dtype
        )
        yield codec.decode(full_codes.transpose(0, 1))


def generate_batch_samples(cfg, target_texts, ar_model, nar_model, codec, sp, special,
                           codebook_size, num_quantizers, device, dtype, pipeline,
                           max_frames=200):
    """Generate multiple targets while phonemizing and encoding the shared reference once."""
    ref_text_ids, ref_codec, target_groups = build_batch_samples_from_config(
        cfg["sample"], target_texts, codec, sp, pipeline
    )
    ref_wav = codec.decode(ref_codec.transpose(0, 1))
    waveforms = []
    for target_text_ids_by_sentence in target_groups:
        sentence_waveforms = list(_generate_sentence_waveforms(
            cfg["sample"], target_text_ids_by_sentence, ref_text_ids, ref_codec,
            ar_model, nar_model, codec, special, codebook_size, num_quantizers,
            device, dtype, max_frames,
        ))
        if len(sentence_waveforms) != len(target_text_ids_by_sentence):
            waveforms.append(None)
        else:
            waveforms.append(torch.cat(sentence_waveforms, dim=-1))
    return waveforms, ref_wav


def generate_sample_stream(cfg, ar_model, nar_model, codec, sp, special, codebook_size,
                           num_quantizers, device, dtype, pipeline, max_frames=200,
                           chunk_frames=5):
    """Yield decoded audio while AR generation continues from its existing KV-cache."""
    ref_text_ids, ref_codec, target_text_ids_by_sentence = build_samples_from_config(
        cfg["sample"], codec, sp, pipeline
    )
    sample_cfg = cfg["sample"]
    for target_text_ids in target_text_ids_by_sentence:
        yielded = False
        for rvq0_codes, ar_hidden in ar_generate_rvq0_stream(
            ar_model, special, codebook_size, ref_text_ids, ref_codec, target_text_ids,
            device, dtype, max_frames=max_frames,
            temperature=sample_cfg.get("temperature", 0.8),
            top_k=sample_cfg.get("top_k", 10),
            chunk_frames=chunk_frames,
        ):
            yielded = True
            full_codes = nar_fill_layers(
                nar_model, ar_hidden, rvq0_codes, num_quantizers, device, dtype
            )
            yield codec.decode(full_codes.transpose(0, 1))
        if not yielded:
            raise RuntimeError("the AR model generated zero audio frames for a sentence")


def generate_sample(cfg, ar_model, nar_model, codec, sp, special, codebook_size,
                     num_quantizers, device, dtype, pipeline, max_frames=200):
    """Phonemize, split the target on periods, synthesize each sentence, then concatenate audio.

    Caller is responsible for ar_model.eval()/nar_model.eval() (and restoring .train() after).
    Returns (gen_wav, ref_wav); ref_wav is decoded through the same codec path. Returns
    (None, ref_wav) if the AR produces 0 frames for any sentence.
    """
    waveforms, ref_wav = generate_batch_samples(
        cfg, [cfg["sample"]["target_text"]], ar_model, nar_model, codec, sp, special,
        codebook_size, num_quantizers, device, dtype, pipeline, max_frames=max_frames,
    )
    return waveforms[0], ref_wav
