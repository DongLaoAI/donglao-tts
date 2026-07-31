import hashlib
import json
import math
import random
from bisect import bisect_right
from collections import Counter

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

from donglao_tts.data.compiled import _hash_file, load_catalog


class TTSDataset(Dataset):
    """Legacy JSONL dataset.

    This remains available for compatibility. New training runs should use
    :class:`CompiledTTSDataset` to avoid parsing and retaining nested codec lists.
    """

    def __init__(self, phon_manifest_paths, spm_model_path, split="train", val_ratio=0.01, seed=42):
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)

        self.entries = []
        for corpus_idx, path in enumerate(phon_manifest_paths):
            with open(path, "r", encoding="utf-8") as source:
                for line in source:
                    entry = json.loads(line)
                    entry["_corpus"] = corpus_idx
                    self.entries.append(entry)

        def is_val(entry):
            value = f"{seed}:{entry['_corpus']}:{entry['id']}".encode()
            return (int(hashlib.md5(value).hexdigest(), 16) % 10000) < val_ratio * 10000

        want_val = split == "val"
        self.split_indices = [
            index for index, entry in enumerate(self.entries) if is_val(entry) == want_val
        ]
        self.by_speaker = {}
        for index in self.split_indices:
            entry = self.entries[index]
            key = (entry["_corpus"], entry["speaker"])
            self.by_speaker.setdefault(key, []).append(index)
        self.corpus_counts = Counter(self.entries[index]["_corpus"] for index in self.split_indices)
        self.num_speakers = len(self.by_speaker)
        speaker_max_frames = {
            key: max(len(self.entries[index]["codec"]) for index in candidates)
            for key, candidates in self.by_speaker.items()
        }
        self._frame_lengths = [
            len(self.entries[index]["codec"])
            + speaker_max_frames[
                (self.entries[index]["_corpus"], self.entries[index]["speaker"])
            ]
            for index in self.split_indices
        ]

    def __len__(self):
        return len(self.split_indices)

    def _sample_ref_idx(self, idx):
        entry = self.entries[idx]
        candidates = self.by_speaker[(entry["_corpus"], entry["speaker"])]
        if len(candidates) == 1:
            return idx
        position = random.randrange(len(candidates) - 1)
        ref_idx = candidates[position]
        return candidates[position + 1] if ref_idx == idx else ref_idx

    def sequence_length(self, i):
        entry = self.entries[self.split_indices[i]]
        return len(entry["codec"]) + len(entry["phoneme"])

    def frame_length(self, i):
        return self._frame_lengths[i]

    def __getitem__(self, i):
        idx = self.split_indices[i]
        target = self.entries[idx]
        ref = self.entries[self._sample_ref_idx(idx)]

        return {
            "ref_text_ids": torch.tensor(
                self.sp.encode(ref["phoneme"], out_type=int), dtype=torch.long
            ),
            "ref_codec": torch.tensor(ref["codec"], dtype=torch.long),
            "target_text_ids": torch.tensor(
                self.sp.encode(target["phoneme"], out_type=int), dtype=torch.long
            ),
            "target_codec": torch.tensor(target["codec"], dtype=torch.long),
        }


class CompiledTTSDataset(Dataset):
    """Memory-mapped compiled dataset with O(1) same-speaker reference sampling."""

    def __init__(self, root_path, spm_model_path, split="train", reference_percentile=90):
        self.root, self.catalog = load_catalog(root_path)
        tokenizer_sha256 = _hash_file(spm_model_path)
        expected_sha256 = self.catalog["tokenizer"]["sha256"]
        if tokenizer_sha256 != expected_sha256:
            raise ValueError(
                "compiled dataset tokenizer does not match tokenizer.model_path: "
                f"expected {expected_sha256}, got {tokenizer_sha256}"
            )
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        try:
            split_id = ("train", "val").index(split)
        except ValueError as exc:
            raise ValueError("split must be 'train' or 'val'") from exc
        if not 0 < reference_percentile <= 100:
            raise ValueError("reference_percentile must be in (0, 100]")

        self.shards = list(self.catalog["shards"])
        self._rows = [
            np.load(self.root / shard["rows_index"], mmap_mode="r") for shard in self.shards
        ]
        self._codec_arrays = [None] * len(self.shards)
        self._text_arrays = [None] * len(self.shards)

        self.samples = []
        self.by_speaker = {}
        self.corpus_counts = Counter()
        for shard_index, (shard, rows) in enumerate(zip(self.shards, self._rows)):
            for row_index in np.flatnonzero(rows["split"] == split_id):
                sample_index = len(self.samples)
                row_index = int(row_index)
                self.samples.append((shard_index, row_index))
                speaker_key = int(rows[row_index]["speaker_key"])
                self.by_speaker.setdefault(speaker_key, []).append(sample_index)
                self.corpus_counts[shard["corpus"]] += 1
        self.num_speakers = len(self.by_speaker)

        # Very long references increase every AR sequence and can collapse batch throughput.
        # Keep every utterance as a target, but sample prompts from the shortest configurable
        # percentile for each speaker (and retain at least two when possible).
        self.reference_by_speaker = {}
        speaker_max_reference_lengths = {}
        speaker_max_reference_frames = {}
        for speaker_key, candidates in self.by_speaker.items():
            ordered = sorted(candidates, key=self._target_length)
            keep = min(
                len(ordered),
                max(2 if len(ordered) > 1 else 1,
                    math.ceil(len(ordered) * reference_percentile / 100)),
            )
            references = ordered[:keep]
            self.reference_by_speaker[speaker_key] = references
            speaker_max_reference_lengths[speaker_key] = max(
                self._target_length(index) for index in references
            )
            speaker_max_reference_frames[speaker_key] = max(
                int(self._row_for_sample(index)["codec_length"]) for index in references
            )
        self._sequence_lengths = np.asarray(
            [
                self._target_length(index)
                + speaker_max_reference_lengths[
                    int(self._row_for_sample(index)["speaker_key"])
                ]
                + 5  # BOS plus the four ref/target section markers
                for index in range(len(self.samples))
            ],
            dtype=np.int64,
        )
        self._frame_lengths = np.asarray(
            [
                int(self._row_for_sample(index)["codec_length"])
                + speaker_max_reference_frames[
                    int(self._row_for_sample(index)["speaker_key"])
                ]
                for index in range(len(self.samples))
            ],
            dtype=np.int64,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_rows"] = None
        state["_codec_arrays"] = [None] * len(self.shards)
        state["_text_arrays"] = [None] * len(self.shards)
        return state

    def _ensure_rows(self):
        if self._rows is None:
            self._rows = [
                np.load(self.root / shard["rows_index"], mmap_mode="r")
                for shard in self.shards
            ]

    def _ensure_shard(self, shard_index):
        self._ensure_rows()
        if self._codec_arrays[shard_index] is None:
            shard = self.shards[shard_index]
            # Copy-on-write mappings are writable from PyTorch's point of view while never
            # modifying the on-disk artifact.
            self._codec_arrays[shard_index] = np.load(
                self.root / shard["codec"], mmap_mode="c"
            )
            self._text_arrays[shard_index] = np.load(
                self.root / shard["text_ids"], mmap_mode="c"
            )

    def _row_for_sample(self, sample_index):
        self._ensure_rows()
        shard_index, row_index = self.samples[sample_index]
        return self._rows[shard_index][row_index]

    def _target_length(self, sample_index):
        row = self._row_for_sample(sample_index)
        return int(row["codec_length"]) + int(row["text_length"])

    def sequence_length(self, sample_index):
        return int(self._sequence_lengths[sample_index])

    def frame_length(self, sample_index):
        return int(self._frame_lengths[sample_index])

    def __len__(self):
        return len(self.samples)

    def _sample_ref(self, sample_index):
        row = self._row_for_sample(sample_index)
        candidates = self.reference_by_speaker[int(row["speaker_key"])]
        if len(candidates) == 1:
            return candidates[0]
        ref_index = random.choice(candidates)
        while ref_index == sample_index:
            ref_index = random.choice(candidates)
        return ref_index

    def _load_sample(self, sample_index):
        shard_index, row_index = self.samples[sample_index]
        self._ensure_shard(shard_index)
        row = self._rows[shard_index][row_index]
        codec_start = int(row["codec_offset"])
        codec_end = codec_start + int(row["codec_length"])
        text_start = int(row["text_offset"])
        text_end = text_start + int(row["text_length"])
        return (
            torch.from_numpy(self._text_arrays[shard_index][text_start:text_end]),
            torch.from_numpy(self._codec_arrays[shard_index][codec_start:codec_end]),
        )

    def __getitem__(self, sample_index):
        ref_text_ids, ref_codec = self._load_sample(self._sample_ref(sample_index))
        target_text_ids, target_codec = self._load_sample(sample_index)
        return {
            "ref_text_ids": ref_text_ids,
            "ref_codec": ref_codec,
            "target_text_ids": target_text_ids,
            "target_codec": target_codec,
        }


class MultiCorpusTTSDataset(Dataset):
    """Combine independent compiled corpus roots without merging their storage."""

    def __init__(self, root_paths, spm_model_path, split="train", reference_percentile=90):
        if not root_paths:
            raise ValueError("at least one compiled dataset path is required")
        self.datasets = [
            CompiledTTSDataset(
                root_path,
                spm_model_path,
                split=split,
                reference_percentile=reference_percentile,
            )
            for root_path in root_paths
        ]
        self.sp = self.datasets[0].sp
        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)
        self.corpus_counts = Counter()
        for dataset in self.datasets:
            self.corpus_counts.update(dataset.corpus_counts)
        self.num_speakers = sum(dataset.num_speakers for dataset in self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _locate(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous_size = self.cumulative_sizes[dataset_index - 1] if dataset_index else 0
        return self.datasets[dataset_index], index - previous_size

    def sequence_length(self, index):
        dataset, local_index = self._locate(index)
        return dataset.sequence_length(local_index)

    def frame_length(self, index):
        dataset, local_index = self._locate(index)
        return dataset.frame_length(local_index)

    def __getitem__(self, index):
        dataset, local_index = self._locate(index)
        return dataset[local_index]
