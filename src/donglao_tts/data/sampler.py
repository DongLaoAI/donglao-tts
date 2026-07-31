"""Length-aware batch sampling for variable-length TTS sequences."""

import math
import random

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    """Build sortish batches and optionally cap total reference plus target codec frames.

    ``max_frames_per_batch`` uses a conservative reference length from the dataset, so the
    randomly selected reference cannot make the actual codec-frame total exceed the budget.
    A value of zero disables dynamic sizing while retaining length bucketing.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        *,
        max_frames_per_batch=0,
        bucket_size=256,
        shuffle=True,
        drop_last=False,
        seed=42,
    ):
        if batch_size < 1 or bucket_size < 1 or max_frames_per_batch < 0:
            raise ValueError("invalid batch sampler configuration")
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_frames_per_batch = max_frames_per_batch
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def _batches(self, epoch):
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(indices)

        buckets = [
            indices[start : start + self.bucket_size]
            for start in range(0, len(indices), self.bucket_size)
        ]
        for bucket in buckets:
            bucket.sort(key=self.dataset.sequence_length)
        if self.shuffle:
            rng.shuffle(buckets)

        batches = []
        for bucket in buckets:
            batch = []
            frame_count = 0
            for index in bucket:
                projected_size = len(batch) + 1
                exceeds_size = projected_size > self.batch_size
                sample_frames = self.dataset.frame_length(index)
                exceeds_frames = (
                    self.max_frames_per_batch > 0
                    and frame_count + sample_frames > self.max_frames_per_batch
                )
                if batch and (exceeds_size or exceeds_frames):
                    batches.append(batch)
                    batch = []
                    frame_count = 0
                batch.append(index)
                frame_count += sample_frames
            if batch and (
                not self.drop_last
                or self.max_frames_per_batch > 0
                or len(batch) == self.batch_size
            ):
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._batches(self.epoch)
        self.epoch += 1
        yield from batches

    def __len__(self):
        if not self.max_frames_per_batch:
            if self.drop_last:
                return len(self.dataset) // self.batch_size
            return math.ceil(len(self.dataset) / self.batch_size)
        return len(self._batches(0))
