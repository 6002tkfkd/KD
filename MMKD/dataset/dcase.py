import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio.compliance.kaldi as ta_kaldi
from torch.utils.data import Dataset, DataLoader


unique_labels = {
    'scene': ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall', 'street_pedestrian',
              'street_traffic', 'tram'],
    'device': ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6'],
    'city': ['barcelona', 'helsinki', 'lisbon', 'london', 'lyon', 'milan', 'paris', 'prague', 'stockholm', 'vienna']
}


class BEATsMel(torch.nn.Module):
    def __init__(self, dataset_mean: float = 15.41663, dataset_std: float = 6.55582):
        super().__init__()
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

    def forward(self, x):
        fbanks = []
        for waveform in x:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - self.dataset_mean) / (2 * self.dataset_std)
        return fbank


class DCASEAudioDataset(Dataset):
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000,
                 target_frames: int = 1000, dataset_mean: float = 15.41663, dataset_std: float = 6.55582):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        self.target_frames = target_frames
        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='	')
        self.spec_extractor = BEATsMel(dataset_mean=dataset_mean, dataset_std=dataset_std)

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
        wav, _ = librosa.load(f"{self.audio_dir}/{filename}", sr=self.sr)
        wav = torch.from_numpy(wav)

        fbank = self.spec_extractor(wav.unsqueeze(0))[0]
        fbank = self._pad_or_trim(fbank)
        fbank = fbank.unsqueeze(0)

        scene_label = filename.split('/')[-1].split('-')[0]
        scene_label = unique_labels['scene'].index(scene_label)
        scene_label = torch.from_numpy(np.array(scene_label, dtype=np.int64))
        return fbank, scene_label


def get_dcase_dataloaders(meta_dir, audio_dir, batch_size=64, num_workers=4, sampling_rate=16000,
                          train_subset='split5', target_frames=1000, dataset_mean=15.41663, dataset_std=6.55582):
    train_set = DCASEAudioDataset(meta_dir, audio_dir, subset=train_subset, sampling_rate=sampling_rate,
                                  target_frames=target_frames, dataset_mean=dataset_mean, dataset_std=dataset_std)
    val_set = DCASEAudioDataset(meta_dir, audio_dir, subset='valid', sampling_rate=sampling_rate,
                                target_frames=target_frames, dataset_mean=dataset_mean, dataset_std=dataset_std)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_loader, val_loader
