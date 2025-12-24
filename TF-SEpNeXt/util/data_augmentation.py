import os
import random
import librosa
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.autograd import Variable


class _DataAugmentation(nn.Module):
    """ Base Module for data augmentation techniques. """


class MixUp(_DataAugmentation):
    def __init__(self, alpha=0.3):
        super(MixUp, self).__init__()
        self.alpha = alpha
        self.lam = 1

    def forward(self, x, y):
        self.lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        x = self.lam * x + (1 - self.lam) * x[index, :]
        y_a, y_b = y, y[index]
        x, y_a, y_b = map(Variable, (x, y_a, y_b))
        return x, (y_a, y_b)


class SoftMixUp(_DataAugmentation):
    def __init__(self, alpha=0.3):
        super(SoftMixUp, self).__init__()
        self.alpha = alpha
        self.lam = 1

    def forward(self, x, y, s):
        self.lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        x = self.lam * x + (1 - self.lam) * x[index, :]
        y_a, y_b = y, y[index]
        s_a, s_b = s, s[index]
        x, y_a, y_b, s_a, s_b = map(Variable, (x, y_a, y_b, s_a, s_b))
        return x, (y_a, y_b), (s_a, s_b)


class FreqMixStyle(_DataAugmentation):
    def __init__(self, alpha=0.3, p=0.7, eps=1e-6):
        super(FreqMixStyle, self).__init__()
        self.alpha = alpha
        self.p = p
        self.eps = eps

    def forward(self, x):
        if np.random.rand() > self.p:
            return x
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        batch_size = x.size(0)
        # Changed from dim=[2,3] to dim=[1,3] from channel-wise statistics to frequency-wise statistics
        f_mu = x.mean(dim=[1, 3], keepdim=True)
        f_var = x.var(dim=[1, 3], keepdim=True)
        # Compute instance standard deviation
        f_sig = (f_var + self.eps).sqrt()
        # Block gradients
        f_mu, f_sig = f_mu.detach(), f_sig.detach()
        # Normalize x
        x_normed = (x - f_mu) / f_sig
        # Generate shuffling indices
        perm = torch.randperm(batch_size).to(x.device)
        # Shuffling
        f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]
        # Generate mixed mean
        mu_mix = f_mu * lam + f_mu_perm * (1 - lam)
        # Generate mixed standard deviation
        sig_mix = f_sig * lam + f_sig_perm * (1 - lam)
        # Denormalize x using the mixed statistics
        return x_normed * sig_mix + mu_mix


class SpecAugmentation(_DataAugmentation):
    def __init__(self, mask_size=0.1, p=0.8):
        super().__init__()
        self.mask_size = mask_size
        self.p = p

    def forward(self, x):
        _, _, f, t = x.size()
        # Create frequency mask and time mask
        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=round(f * self.mask_size), iid_masks=True)
        time_masking = torchaudio.transforms.TimeMasking(time_mask_param=round(t * self.mask_size), iid_masks=True)
        # Apply mask according to random probability
        x = freq_masking(x) if np.random.uniform(0, 1) < self.p else x
        x = time_masking(x) if np.random.uniform(0, 1) < self.p else x
        return x


class DeviceImpulseResponseAugmentation(_DataAugmentation):
    def __init__(self, path_ir, p=0.4, mode="full"):
        super().__init__()
        self.path_ir = path_ir
        self.ir_files = os.listdir(path_ir)
        self.p = p
        self.mode = mode

    def forward(self, x, d):
        batch_size = x.size(0)
        for i in range(batch_size):
            # Only apply for data from device A
            if d[i] == torch.tensor([0]).to(x.device) and np.random.uniform(0, 1) < self.p:
                # Randomly select an impulse response
                random_file = random.choice(self.ir_files)
                ir, _ = librosa.load(f"{self.path_ir}/{random_file}", sr=32000)
                ir = torch.from_numpy(ir).to(x.device)
                y = torchaudio.functional.fftconvolve(x[i], ir, mode=self.mode)
                x[i] = y[:len(x[i])]
        return x
#기존 data_augmentation + filterAugment

class FilterAugmentation(_DataAugmentation):
    def __init__(self, db_range=(-6, 6), n_band=(3, 6), min_bw=6, filter_type="linear", p=0.7, mix_prob=0.5):
        super().__init__()
        self.db_range=db_range
        self.n_band=n_band
        self.min_bw=min_bw
        self.filter_type=filter_type
        self.p=p
        self.mix_prob = mix_prob if isinstance(mix_prob, float) else 0.5

    def forward(self, x):
        if np.random. uniform(0, 1) > self.p:
            return x
        
        batch_size, n_freq_bin, time_steps=x.shape

        filter_type=self.filter_type
        n_band=self.n_band
        min_bw=self.min_bw

        if filter_type == "mix":
            filter_type = "step" if torch.rand(1).item() < self.mix_prob else "linear"
            if filter_type == "step":
                n_band = (2, 5)
                min_bw = 4
            else:
                n_band = (3, 6)
                min_bw = 6

        elif filter_type == "step":
            n_band = (2, 5)
            min_bw = 4

        elif filter_type == "linear":
            n_band = (3, 6)
            min_bw = 6

        n_freq_band=torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()

        if n_freq_band > 1:
            while n_freq_bin - n_freq_band * min_bw + 1 < 0:
                min_bw -= 1
            band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1, (n_freq_band - 1,)))[0] + \
                               torch.arange(1, n_freq_band) * min_bw
            band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))
            
            if filter_type=="step":
                band_factors=torch.rand((batch_size, n_freq_band), device=x.device) * \
                               (self.db_range[1] - self.db_range[0]) + self.db_range[0]
                band_factors = 10 ** (band_factors / 20)

                freq_filt = torch.ones((batch_size, n_freq_bin, 1), device=x.device)
                for i in range(n_freq_band):
                    freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                        band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

            elif filter_type == "linear":
                band_factors = torch.rand((batch_size, n_freq_band + 1), device=x.device) * \
                               (self.db_range[1] - self.db_range[0]) + self.db_range[0]
                freq_filt = torch.ones((batch_size, n_freq_bin, 1), device=x.device)
                for i in range(n_freq_band):
                    for j in range(batch_size):
                        freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                            torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                           band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
                freq_filt = 10 ** (freq_filt / 20)

            return x * freq_filt

        else:
            return x