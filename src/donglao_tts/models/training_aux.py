"""Training-only objectives that are deliberately absent from inference/export graphs."""

import torch
import torch.nn.functional as F
from torch import nn


class CTCAuxiliaryHead(nn.Module):
    """Upsample target-frame AR states and classify the target text with CTC.

    The codec produces only 12.5 states/second.  A learned stride-2 transposed convolution gives
    CTC twice as many, non-identical alignment positions without changing the codec or runtime AR.
    ``blank_id`` is one extra class after the SentencePiece vocabulary.
    """

    def __init__(self, d_model, vocab_size, upsample_factor=2):
        super().__init__()
        if upsample_factor != 2:
            raise ValueError("CTCAuxiliaryHead currently supports upsample_factor=2 only")
        self.upsample_factor = upsample_factor
        self.blank_id = vocab_size
        self.upsample = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=4, stride=2, padding=1
        )
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, vocab_size + 1)

    def forward(self, hidden):
        """Return unnormalized CTC logits shaped ``[B, 2*T, vocab_size+1]``."""
        hidden = self.upsample(hidden.transpose(1, 2)).transpose(1, 2)
        return self.projection(self.norm(hidden))

    def loss(self, hidden, input_lengths, targets, target_lengths):
        logits = self(hidden)
        # CTCLoss is numerically sensitive and is not safe in bf16/fp16.  Casting here keeps the
        # AR/head forward under autocast while doing log-softmax and the dynamic program in fp32.
        log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
        return F.ctc_loss(
            log_probs,
            targets,
            input_lengths=input_lengths * self.upsample_factor,
            target_lengths=target_lengths,
            blank=self.blank_id,
            reduction="mean",
            zero_infinity=True,
        )


def eos_auxiliary_loss(logits, target_codec_lengths, target_start_indices, codebook_size):
    """Balanced binary EOS loss derived from the existing shared AR logits.

    No new EOS head is introduced: the score is the log-odds of the existing EOS class against
    all codec classes.  Each sample gives equal weight to its single positive EOS position and the
    mean of all preceding negative positions, avoiding the roughly 1/T dilution in the main CE.
    """
    sample_losses = []
    for b in range(logits.shape[0]):
        target_length = int(target_codec_lengths[b])
        prediction_start = int(target_start_indices[b]) - 1  # [CODE_TARGET] predicts frame zero
        sequence_logits = logits[
            b, prediction_start:prediction_start + target_length + 1
        ].float()
        eos_score = (
            sequence_logits[:, codebook_size]
            - torch.logsumexp(sequence_logits[:, :codebook_size], dim=-1)
        )
        negative = F.softplus(eos_score[:-1]).mean()
        positive = F.softplus(-eos_score[-1])
        sample_losses.append(0.5 * (negative + positive))
    return torch.stack(sample_losses).mean()
