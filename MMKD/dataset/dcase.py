import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio
from torch.utils.data import Dataset, DataLoader


unique_labels = {
    'scene': ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall', 'street_pedestrian',
              'street_traffic', 'tram'],
    'device': ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6'],
    'city': ['barcelona', 'helsinki', 'lisbon', 'london', 'lyon', 'milan', 'paris', 'prague', 'stockholm', 'vienna']
}


class BEATsMel(torch.nn.Module):
    def __init__(self, dataset_mean: float = 15.41663, dataset_std: float = 6.55582,
                 sample_frequency: int = 16000):
        super().__init__()
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.sample_frequency = sample_frequency

    def forward(self, x):
        fbanks = []
        for waveform in x:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=self.sample_frequency,
                frame_length=25,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - self.dataset_mean) / (2 * self.dataset_std)
        return fbank


class FbankExtractor(torch.nn.Module):
    def __init__(self, dataset_mean=None, dataset_std=None, norm_div=1.0, sample_frequency: int = 16000):
        super().__init__()
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.norm_div = norm_div
        self.sample_frequency = sample_frequency

    def forward(self, x):
        fbanks = []
        for waveform in x:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=self.sample_frequency,
                frame_length=25,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        if self.dataset_mean is not None and self.dataset_std is not None:
            fbank = (fbank - self.dataset_mean) / (self.norm_div * self.dataset_std)
        return fbank


class PaSSTMel(FbankExtractor):
    def __init__(self, dataset_mean=None, dataset_std=None):
        super().__init__(dataset_mean=dataset_mean, dataset_std=dataset_std, norm_div=1.0)


class EfficientNetMel(FbankExtractor):
    def __init__(self, dataset_mean=None, dataset_std=None):
        super().__init__(dataset_mean=dataset_mean, dataset_std=dataset_std, norm_div=1.0)


class CpMel(torch.nn.Module):
    def __init__(self, n_mels=512, sr=32000, win_length=3072, hop_size=500, n_fft=4096, fmin=0.0, fmax=None):
        super().__init__()
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        self.fmax = sr // 2 if fmax is None else fmax
        self.hop_size = hop_size
        self.register_buffer('window', torch.hann_window(win_length, periodic=False), persistent=False)
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)

    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient.to(x.device)).squeeze(1)
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window.to(x.device),
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = (x ** 2).sum(dim=-1)
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sr,
            self.fmin,
            self.fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
            device=x.device,
        )
        with torch.amp.autocast('cuda', enabled=False):
            melspec = torch.matmul(mel_basis, x)
        melspec = (melspec + 0.00001).log()
        melspec = (melspec + 4.5) / 5.0
        return melspec


def _build_extractors(input_keys, dataset_mean, dataset_std,
                      passt_mean=None, passt_std=None,
                      efficientnet_mean=None, efficientnet_std=None,
                      sample_frequency_map=None, default_sample_frequency: int = 16000):
    passt_mean = dataset_mean if passt_mean is None else passt_mean
    passt_std = dataset_std if passt_std is None else passt_std
    efficientnet_mean = dataset_mean if efficientnet_mean is None else efficientnet_mean
    efficientnet_std = dataset_std if efficientnet_std is None else efficientnet_std
    sample_frequency_map = sample_frequency_map or {}
    extractors = {}
    for key in input_keys:
        sample_frequency = sample_frequency_map.get(key, default_sample_frequency)
        if key == 'beats':
            extractors[key] = BEATsMel(
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
                sample_frequency=sample_frequency,
            )
        elif key == 'tfsepnext':
            extractors[key] = CpMel(
                n_mels=512,
                sr=sample_frequency,
            )
        elif key == 'passt':
            extractors[key] = PaSSTMel(
                dataset_mean=passt_mean,
                dataset_std=passt_std,
                sample_frequency=sample_frequency,
            )
        elif key == 'efficientnet':
            extractors[key] = EfficientNetMel(
                dataset_mean=efficientnet_mean,
                dataset_std=efficientnet_std,
                sample_frequency=sample_frequency,
            )
        else:
            raise ValueError(f"Unsupported input key: {key}")
    return extractors


class DCASEAudioDataset(Dataset):
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000,
                 target_frames: int = 1000, dataset_mean: float = 15.41663, dataset_std: float = 6.55582,
                 input_keys=None, passt_mean=None, passt_std=None, efficientnet_mean=None, efficientnet_std=None,
                 sampling_rate_map=None):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        self.sr_map = sampling_rate_map or {}
        self.target_frames = target_frames
        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='	')
        self.input_keys = input_keys or ['beats']
        self.extractors = _build_extractors(
            self.input_keys,
            dataset_mean,
            dataset_std,
            passt_mean=passt_mean,
            passt_std=passt_std,
            efficientnet_mean=efficientnet_mean,
            efficientnet_std=efficientnet_std,
            sample_frequency_map=self.sr_map,
            default_sample_frequency=self.sr,
        )

    def __len__(self):
        return len(self.meta_subset)

    def _pad_or_trim(self, fbank):
        if self.target_frames is None:
            return fbank
        frames = fbank.shape[0]
        if frames == self.target_frames:
            return fbank
        if frames < self.target_frames:
            pad = torch.zeros((self.target_frames - frames, fbank.shape[1]), dtype=fbank.dtype)
            return torch.cat([fbank, pad], dim=0)
        return fbank[:self.target_frames]

    def __getitem__(self, i):
        row_i = self.meta_subset.iloc[i]
        filename = row_i["filename"]

        inputs = {}
        wav_cache = {}
        for key in self.input_keys:
            sample_rate = self.sr_map.get(key, self.sr)
            if sample_rate not in wav_cache:
                wav, _ = librosa.load(f"{self.audio_dir}/{filename}", sr=sample_rate)
                wav_cache[sample_rate] = torch.from_numpy(wav)
            fbank = self.extractors[key](wav_cache[sample_rate].unsqueeze(0))[0]
            fbank = self._pad_or_trim(fbank)
            inputs[key] = fbank.unsqueeze(0)

        scene_label = filename.split('/')[-1].split('-')[0]
        scene_label = unique_labels['scene'].index(scene_label)
        scene_label = torch.from_numpy(np.array(scene_label, dtype=np.int64))
        if len(self.input_keys) == 1:
            return inputs[self.input_keys[0]], scene_label
        return inputs, scene_label


def get_dcase_dataloaders(meta_dir, audio_dir, batch_size=64, num_workers=4, sampling_rate=16000,
                          train_subset='split5', target_frames=1000, dataset_mean=15.41663, dataset_std=6.55582,
                          input_keys=None, passt_mean=None, passt_std=None, efficientnet_mean=None,
                          efficientnet_std=None, sampling_rate_map=None):
    train_set = DCASEAudioDataset(meta_dir, audio_dir, subset=train_subset, sampling_rate=sampling_rate,
                                  target_frames=target_frames, dataset_mean=dataset_mean, dataset_std=dataset_std,
                                  input_keys=input_keys, passt_mean=passt_mean, passt_std=passt_std,
                                  efficientnet_mean=efficientnet_mean, efficientnet_std=efficientnet_std,
                                  sampling_rate_map=sampling_rate_map)
    val_set = DCASEAudioDataset(meta_dir, audio_dir, subset='valid', sampling_rate=sampling_rate,
                                target_frames=target_frames, dataset_mean=dataset_mean, dataset_std=dataset_std,
                                input_keys=input_keys, passt_mean=passt_mean, passt_std=passt_std,
                                efficientnet_mean=efficientnet_mean, efficientnet_std=efficientnet_std,
                                sampling_rate_map=sampling_rate_map)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_loader, val_loader
