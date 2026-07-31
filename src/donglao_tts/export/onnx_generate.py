"""Reimplements generate.py's ar_generate_rvq0/nar_fill_layers against onnxruntime.InferenceSession.
Embedding lookups (SplitEmbedding, CodecEmbeddingTable) still go through
the real PyTorch `ar_model.embed`/`nar_model` modules passed in here -- only the exported
Transformer forward passes run through ONNX Runtime; see onnx_export.py for that scoping
rationale. Sampling/EOS-stopping stays in Python/numpy, matching the PyTorch path exactly."""

import numpy as np
import onnxruntime as ort
import torch


def _softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sample_from_logits_np(logits, temperature, top_k, rng):
    """numpy counterpart of generate.py's sample_from_logits -- same contract (temperature<=0 is
    plain argmax, otherwise temperature-scaled + optional top-k + multinomial sample)."""
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits.astype(np.float64) / temperature
    if top_k and top_k > 0:
        k = min(top_k, scaled.shape[-1])
        kth_val = np.partition(scaled, -k)[-k]
        scaled = np.where(scaled < kth_val, -np.inf, scaled)
    probs = _softmax_np(scaled)
    return int(rng.choice(len(probs), p=probs))


class OnnxARGenerator:
    """Wraps the two AR ONNX sessions (see onnx_export.export_ar_prefill/export_ar_decode_step).
    `embed` is the real (lightweight) SplitEmbedding module from an ARTransformerLM instance --
    used only for per-step embedding lookups, never for a Transformer forward pass."""

    def __init__(self, prefill_path, decode_step_path, embed, codebook_size, providers=None):
        providers = providers or ["CPUExecutionProvider"]
        self.prefill_session = ort.InferenceSession(prefill_path, providers=providers)
        self.decode_session = ort.InferenceSession(decode_step_path, providers=providers)
        self.embed = embed
        self.codebook_size = codebook_size
        self.decode_input_names = {item.name for item in self.decode_session.get_inputs()}

    @torch.no_grad()
    def generate_rvq0(self, input_embeds, padding_mask, max_frames=200, temperature=1.0, top_k=0,
                       seed=None, return_hidden=False):
        """input_embeds [1,L,D] float32, padding_mask [1,L] bool -- the already-assembled prompt
        (see generate.py's build_input_embeds), computed in PyTorch exactly as before. Returns a
        list of generated RVQ0 codec ids (no EOS included), matching ar_generate_rvq0's contract."""
        rng = np.random.default_rng(seed)
        input_embeds_np = input_embeds.numpy().astype(np.float32)
        padding_mask_np = padding_mask.numpy()

        logits, present_keys, present_values = self.prefill_session.run(
            None, {"input_embeds": input_embeds_np, "padding_mask": padding_mask_np})
        next_id = sample_from_logits_np(logits[0, -1], temperature, top_k, rng)

        generated = []
        hidden_states = []
        for _ in range(max_frames):
            if next_id == self.codebook_size:
                break
            generated.append(next_id)

            code_tensor = torch.tensor([next_id])
            step_embed = self.embed.embed_codec_layer(code_tensor, 0).unsqueeze(0)  # [1,1,D]
            step_embed_np = step_embed.numpy().astype(np.float32)

            decode_inputs = {
                "input_embeds": step_embed_np,
                "past_keys": present_keys,
                "past_values": present_values,
            }
            if "position_ids" in self.decode_input_names:
                past_len = present_keys.shape[3]
                decode_inputs["position_ids"] = np.asarray([[past_len]], dtype=np.int64)
            outputs = self.decode_session.run(None, decode_inputs)
            if len(outputs) == 4:
                logits, hidden, present_keys, present_values = outputs
                hidden_states.append(hidden[0, 0])
            else:  # compatibility with legacy custom-backbone decode graphs
                logits, present_keys, present_values = outputs
                if return_hidden:
                    raise RuntimeError(
                        "decode-step graph does not expose hidden states; re-export the bundle"
                    )
            next_id = sample_from_logits_np(logits[0, -1], temperature, top_k, rng)

        if not return_hidden:
            return generated
        d_model = self.embed.text_table.weight.shape[1]
        if hidden_states:
            ar_hidden = torch.from_numpy(np.stack(hidden_states)).unsqueeze(0)
        else:
            ar_hidden = torch.zeros(1, 0, d_model)
        return generated, ar_hidden


class OnnxNARGenerator:
    """Wraps the single NAR ONNX session (see onnx_export.export_nar_layer)."""

    def __init__(self, nar_layer_path, num_quantizers, providers=None):
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(nar_layer_path, providers=providers)
        self.num_quantizers = num_quantizers

    def fill_layers(self, ar_hidden, rvq0_codes):
        """ar_hidden [1,T,D] float32 (from OnnxARGenerator.generate_rvq0's caller -- still
        produced by the real PyTorch AR model's hidden states during/after generation, see
        generate.py's extract_target_hidden), rvq0_codes: list[int] of length T. Returns
        [T, num_quantizers] int64 numpy array, matching nar_fill_layers' contract."""
        T = len(rvq0_codes)
        ar_hidden_np = ar_hidden.numpy().astype(np.float32)
        target_codec = np.zeros((1, T, self.num_quantizers), dtype=np.int64)
        target_codec[0, :, 0] = np.asarray(rvq0_codes, dtype=np.int64)
        pad_mask = np.zeros((1, T), dtype=bool)

        for k in range(1, self.num_quantizers):
            known = target_codec[:, :, :k]
            layer_ids = np.arange(k, dtype=np.int64)
            (logits,) = self.session.run(None, {
                "ar_hidden": ar_hidden_np, "known_target_codec": known,
                "layer_ids": layer_ids, "k": np.array(k, dtype=np.int64),
                "target_padding_mask": pad_mask,
            })
            target_codec[0, :, k] = logits.argmax(axis=-1)[0]

        return target_codec[0]
