import os
import sys
import torch
import torch.nn as nn


def _ensure_efficientnet_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    eff_root = os.path.join(repo_root, 'EfficientNet-PyTorch')
    if eff_root not in sys.path:
        sys.path.insert(0, eff_root)


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
            continue
        filtered[key] = val
    return filtered


class EfficientNetWrapper(nn.Module):
    def __init__(
        self,
        pretrained=None,
        num_classes=10,
        model_name='efficientnet-b7',
        image_size=None,
        **kwargs,
    ):
        super().__init__()
        _ensure_efficientnet_on_path()
        from efficientnet_pytorch import EfficientNet

        override_params = {'num_classes': num_classes, 'include_top': True}
        if image_size is not None:
            override_params['image_size'] = image_size

        self.model = EfficientNet.from_name(model_name, **override_params)

        if pretrained:
            ckpt = torch.load(pretrained, map_location='cpu')
            state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
            state_dict = _filter_state_dict(self.model, state_dict)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f'[EfficientNet] load_state_dict missing={len(missing)} unexpected={len(unexpected)}')

    def _prep_input(self, x):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def _collect_reductions(self, endpoints):
        keys = sorted(endpoints.keys(), key=lambda k: int(k.split('_')[-1]))
        reductions = [endpoints[k] for k in keys]
        if not reductions:
            raise RuntimeError('EfficientNet endpoints are empty')
        while len(reductions) < 4:
            reductions.append(reductions[-1])
        return reductions

    def forward(self, x, is_feat=False, preact=False, **kwargs):
        x = self._prep_input(x)
        endpoints = self.model.extract_endpoints(x)
        reductions = self._collect_reductions(endpoints)

        f0, f1, f2, f3 = reductions[:4]
        last = reductions[-1]

        pooled = self.model._avg_pooling(last).flatten(start_dim=1)
        pooled = self.model._dropout(pooled)
        logits = self.model._fc(pooled)

        if is_feat:
            return [f0, f1, f2, f3, pooled], logits
        return logits


def efficientnet_b7(**kwargs):
    return EfficientNetWrapper(**kwargs)
