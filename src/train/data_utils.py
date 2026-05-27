import os
import pickle
import logging

logger = logging.getLogger(__name__)

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from src.train.mel_processing import spectrogram_torch
from src.train.utils import load_filepaths_and_text, load_wav_to_torch


def save_tensor_atomic(tensor, filename):
    tmp_filename = f"{filename}.{os.getpid()}.tmp"
    try:
        torch.save(tensor, tmp_filename, _use_new_zipfile_serialization=False)
        os.replace(tmp_filename, filename)
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


class _TextAudioLoaderBase(Dataset):
    has_f0 = False
    expected_columns = 3
    filelist_label = "training"

    def __init__(
        self,
        audiopaths_and_text,
        hparams,
        return_sample_name=False,
        filelist_label=None,
    ):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.return_sample_name = bool(return_sample_name)
        if filelist_label is not None:
            self.filelist_label = filelist_label
        for row in self.audiopaths_and_text:
            if len(row) != self.expected_columns:
                raise ValueError(
                    f"Expected {self.expected_columns} columns in {self.filelist_label} "
                    f"filelist, got {len(row)}: {row}"
                )
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self._filter()

    def _filter(self):
        filtered = []
        lengths = []
        for row in self.audiopaths_and_text:
            audiopath = row[0]
            if os.path.isfile(audiopath):
                filtered.append(row)
                lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))
        if not filtered:
            raise ValueError(f"No valid audio files found in {self.filelist_label} filelist.")
        self.audiopaths_and_text = filtered
        self.lengths = lengths

    def get_sid(self, sid):
        return torch.LongTensor([int(sid)])

    def get_audio_text_pair(self, row):
        if self.has_f0:
            file, phone_path, pitch_path, pitchf_path, sid = row
            phone, pitch, pitchf = self.get_labels(phone_path, pitch_path, pitchf_path)
        else:
            file, phone_path, sid = row
            phone = self.get_labels(phone_path)
            pitch = pitchf = None

        spec, wav = self.get_audio(file)
        sid = self.get_sid(sid)
        sample_name = os.path.splitext(os.path.basename(file))[0]
        len_phone = phone.size(0)
        len_spec = spec.size(-1)
        lengths = [len_phone, len_spec]
        if self.has_f0:
            lengths.extend([pitch.size(0), pitchf.size(0)])
        len_min = min(lengths)
        if any(length != len_min for length in lengths):
            len_wav = len_min * self.hop_length
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]
            if self.has_f0:
                pitch = pitch[:len_min]
                pitchf = pitchf[:len_min]

        if self.has_f0:
            if self.return_sample_name:
                return spec, wav, phone, pitch, pitchf, sid, sample_name
            return spec, wav, phone, pitch, pitchf, sid
        if self.return_sample_name:
            return spec, wav, phone, sid, sample_name
        return spec, wav, phone, sid

    def get_labels(self, phone, pitch=None, pitchf=None):
        phone = np.repeat(np.load(phone), 2, axis=0)
        n_num = min(phone.shape[0], 900)
        phone = torch.FloatTensor(phone[:n_num, :])
        if not self.has_f0:
            return phone
        pitch = torch.LongTensor(np.load(pitch)[:n_num])
        pitchf = torch.FloatTensor(np.load(pitchf)[:n_num])
        return phone, pitch, pitchf

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio.unsqueeze(0)
        spec_filename = self.get_spec_filename(filename)
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename)
            except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
                logger.warning("Failed to load spec cache %s: %s", spec_filename, exc)
                spec = self.compute_spec(audio_norm)
                save_tensor_atomic(spec, spec_filename)
        else:
            spec = self.compute_spec(audio_norm)
            save_tensor_atomic(spec, spec_filename)
        return spec, audio_norm

    def compute_spec(self, audio_norm):
        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        return torch.squeeze(spec, 0)

    def get_spec_filename(self, filename):
        stem, _ = os.path.splitext(filename)
        return (
            f"{stem}.sr{self.sampling_rate}.fft{self.filter_length}"
            f".hop{self.hop_length}.win{self.win_length}.spec.pt"
        )

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioLoaderMultiNSFsid(_TextAudioLoaderBase):
    has_f0 = True
    expected_columns = 5
    filelist_label = "f0 training"


class TextAudioLoader(_TextAudioLoaderBase):
    has_f0 = False
    expected_columns = 3
    filelist_label = "training"


class _TextAudioCollateBase:
    has_f0 = False

    def __init__(self, return_ids=False, return_sample_names=False):
        self.return_ids = return_ids
        self.return_sample_names = return_sample_names

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.zeros(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.zeros(len(batch), 1, max_wave_len)

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.zeros(len(batch), max_phone_len, batch[0][2].shape[1])
        sid = torch.LongTensor(len(batch))
        sample_names = [] if self.return_sample_names else None
        if self.has_f0:
            pitch_padded = torch.zeros(len(batch), max_phone_len, dtype=torch.long)
            pitchf_padded = torch.zeros(len(batch), max_phone_len)

        for i, sorted_idx in enumerate(ids_sorted_decreasing):
            row = batch[sorted_idx]
            spec, wave, phone = row[0], row[1], row[2]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)
            if self.has_f0:
                pitch, pitchf = row[3], row[4]
                pitch_padded[i, : pitch.size(0)] = pitch
                pitchf_padded[i, : pitchf.size(0)] = pitchf
                sid[i] = row[5].item()
                if self.return_sample_names:
                    sample_names.append(row[6])
            else:
                sid[i] = row[3].item()
                if self.return_sample_names:
                    sample_names.append(row[4])

        if self.has_f0:
            result = (
                phone_padded,
                phone_lengths,
                pitch_padded,
                pitchf_padded,
                spec_padded,
                spec_lengths,
                wave_padded,
                wave_lengths,
                sid,
            )
        else:
            result = (
                phone_padded,
                phone_lengths,
                spec_padded,
                spec_lengths,
                wave_padded,
                wave_lengths,
                sid,
            )
        extras = []
        if self.return_sample_names:
            extras.append(tuple(sample_names))
        if self.return_ids:
            extras.append(ids_sorted_decreasing)
        return result + tuple(extras) if extras else result


class TextAudioCollateMultiNSFsid(_TextAudioCollateBase):
    has_f0 = True


class TextAudioCollate(_TextAudioCollateBase):
    has_f0 = False


class DistributedBucketSampler(DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = list(boundaries)

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        if not self.buckets:
            raise ValueError("No training samples fit the bucket boundaries.")
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):  #
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        if len(self.batches) * self.batch_size != self.num_samples:
            raise RuntimeError("Bucket sampler produced an inconsistent number of samples")
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size

