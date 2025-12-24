import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_passt_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    passt_root = os.path.join(repo_root, 'passt')
    if passt_root not in sys.path:
        sys.path.insert(0, passt_root)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model', 'net', 'module'):
            val = ckpt.get(key)
            if isinstance(val, dict):
                return val
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith('module.') for k in state_dict.keys()):
        return state_dict
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}


def _filter_state_dict(model, state_dict):
    filtered = {}
    for key, val in state_dict.items():
        if key not in model.state_dict():
            continue
        if model.state_dict()[key].shape != val.shape:
            if key.endswith('time_new_pos_embed') and val.dim() == 4:
                target_shape = model.state_dict()[key].shape
                val = F.interpolate(val, size=(val.shape[2], target_shape[3]), mode='bilinear', align_corners=False)
            else:
                continue
        filtered[key] = val
    return filtered


class PaSSTWrapper(nn.Module):
    def __init__(
        self,
        pretrained=None,
        num_classes=10,
        arch='passt_l_kd_p16_128_ap47',
        input_fdim=128,
        input_tdim=1000,
        fstride=10,
        tstride=10,
        **kwargs,
    ):
        super().__init__()
        _ensure_passt_on_path()
        from passt.models.passt import get_model

        self.input_tdim = input_tdim
        self.model = get_model(
            arch=arch,
            pretrained=False,
            n_classes=num_classes,
            in_channels=1,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            fstride=fstride,
            tstride=tstride,
        )

        if pretrained:
            ckpt = torch.load(pretrained, map_location='cpu')
            state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
            state_dict = _filter_state_dict(self.model, state_dict)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f'[PaSST] load_state_dict missing={len(missing)} unexpected={len(unexpected)}')

    def _to_passt_layout(self, x):
        if x.dim() == 4 and x.size(1) == 1 and x.size(2) != 128 and x.size(3) == 128:
            x = x.transpose(2, 3)
        if x.dim() == 4 and x.size(3) > self.input_tdim:
            # Keep PaSST input length consistent with its pretrained config.
            return x[:, :, :, :self.input_tdim]
        return x

    def forward(self, x, is_feat=False, preact=False, **kwargs):
        x = self._to_passt_layout(x)
        f_patch = self.model.patch_embed(x)
        logits, pooled = self.model(x)

        if is_feat:
            f0 = f_patch
            f1 = f_patch
            f2 = f_patch
            f3 = f_patch
            return [f0, f1, f2, f3, pooled], logits
        return logits


def passt_base(**kwargs):
    return PaSSTWrapper(**kwargs)
