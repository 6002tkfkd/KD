#!/usr/bin/env python
import argparse
import csv
import os

import librosa
import numpy as np
import torch
import torchaudio.compliance.kaldi as ta_kaldi


def parse_args():
    parser = argparse.ArgumentParser("Compute DCASE fbank mean/std")
    parser.add_argument("--meta_dir", type=str, required=True, help="Path to DCASE meta dir")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to DCASE audio dir")
    parser.add_argument("--subset", type=str, default="split5", help="CSV subset name without extension")
    parser.add_argument("--sampling_rate", type=int, default=16000, help="Sampling rate")
    parser.add_argument("--target_frames", type=int, default=1000, help="Target frames for padding/trimming")
    parser.add_argument("--max_files", type=int, default=0, help="Limit number of files (0 = all)")
    return parser.parse_args()


def pad_or_trim(fbank, target_frames):
    if target_frames is None:
        return fbank
    frames = fbank.shape[0]
    if frames == target_frames:
        return fbank
    if frames < target_frames:
        pad = torch.zeros((target_frames - frames, fbank.shape[1]), dtype=fbank.dtype)
        return torch.cat([fbank, pad], dim=0)
    return fbank[:target_frames]


def iter_filenames(meta_csv_path):
    with open(meta_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row["filename"]


def compute_stats(args):
    meta_csv = os.path.join(args.meta_dir, f"{args.subset}.csv")
    filenames = list(iter_filenames(meta_csv))
    if args.max_files and args.max_files > 0:
        filenames = filenames[:args.max_files]

    total_count = 0
    mean = 0.0
    m2 = 0.0

    for idx, rel_path in enumerate(filenames):
        wav, _ = librosa.load(os.path.join(args.audio_dir, rel_path), sr=args.sampling_rate)
        wav = torch.from_numpy(wav)

        waveform = wav.unsqueeze(0) * 2 ** 15
        fbank = ta_kaldi.fbank(
            waveform,
            num_mel_bins=128,
            sample_frequency=args.sampling_rate,
            frame_length=25,
            frame_shift=10,
        )
        fbank = pad_or_trim(fbank, args.target_frames)
        data = fbank.reshape(-1).numpy().astype(np.float64)

        for value in data:
            total_count += 1
            delta = value - mean
            mean += delta / total_count
            delta2 = value - mean
            m2 += delta * delta2

        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1}/{len(filenames)} files...")

    if total_count < 2:
        raise RuntimeError("Not enough data to compute statistics.")

    variance = m2 / (total_count - 1)
    std = np.sqrt(variance)
    return float(mean), float(std)


def main():
    args = parse_args()
    mean, std = compute_stats(args)
    print(f"Mean: {mean:.5f}")
    print(f"Std:  {std:.5f}")


if __name__ == "__main__":
    main()
