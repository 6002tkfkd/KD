import torch
import torch.nn as nn

from pathlib import Path

from .beats.BEATs import BEATsConfig, BEATs


class BEATsWrapper(nn.Module):
    def __init__(self, pretrained=None, num_classes=10, **kwargs):
        super().__init__()
        ckpt = torch.load(pretrained, map_location='cpu') if pretrained else None
        hyperparams = None
        base_ckpt = None
        if ckpt is not None:
            if 'cfg' in ckpt:
                hyperparams = ckpt['cfg']
            elif 'state_dict' in ckpt:
                base_ckpt = self._load_base_beats_ckpt(pretrained)
                hyperparams = base_ckpt['cfg']
        if hyperparams is None:
            hyperparams = dict(kwargs)
        if ckpt is None and 'input_patch_size' not in hyperparams:
            hyperparams['input_patch_size'] = 16
        cfg = BEATsConfig(hyperparams)
        if not cfg.input_patch_size or cfg.input_patch_size <= 0:
            raise ValueError('input_patch_size must be set when pretrained is None')

        self.encoder = BEATs(cfg)
        self.classifier = nn.Linear(cfg.encoder_embed_dim, num_classes)
        self.num_classes = num_classes
        if ckpt is not None:
            if 'model' in ckpt:
                self.encoder.load_state_dict(ckpt['model'], strict=False)
            elif 'state_dict' in ckpt:
                self._load_lightning_state(ckpt['state_dict'])

    def _load_base_beats_ckpt(self, lightning_ckpt_path):
        config_path = Path(lightning_ckpt_path).parents[1] / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(
                f'Could not find config.yaml for {lightning_ckpt_path}. '
                'Please provide a BEATs checkpoint with cfg.'
            )
        base_path = None
        with config_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith('pretrained:'):
                    base_path = stripped.split(':', 1)[1].strip().strip('\'"')
                    break
        if not base_path:
            raise ValueError(f'Could not find pretrained path in {config_path}')
        return torch.load(base_path, map_location='cpu')

    def _load_lightning_state(self, state_dict):
        encoder_state = {}
        classifier_state = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.encoder.'):
                encoder_state[key.replace('backbone.encoder.', '', 1)] = value
            elif key.startswith('backbone.classifier.linear.'):
                classifier_state[key.replace('backbone.classifier.linear.', '', 1)] = value
        self.encoder.load_state_dict(encoder_state, strict=False)
        if classifier_state:
            self.classifier.load_state_dict(classifier_state, strict=False)

    def _patch_embed(self, x):
        features = self.encoder.patch_embedding(x)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.encoder.layer_norm(features)
        return features

    def forward(self, x, is_feat=False, preact=False, **kwargs):
        if x.dim() == 4 and x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)

        features = self._patch_embed(x)
        f0 = features.transpose(1, 2).unsqueeze(-1)

        if self.encoder.post_extract_proj is not None:
            features = self.encoder.post_extract_proj(features)
        f1 = features.transpose(1, 2).unsqueeze(-1)

        features = self.encoder.dropout_input(features)

        layer_count = self.encoder.cfg.encoder_layers
        tgt_layer = max(layer_count - 1, 0)
        x_enc, layer_results = self.encoder.encoder.extract_features(
            features, padding_mask=None, tgt_layer=tgt_layer
        )

        if layer_results:
            mid_idx = min(len(layer_results) - 1, layer_count // 2)
            mid_x = layer_results[mid_idx][0].transpose(0, 1)
            last_x = layer_results[-1][0].transpose(0, 1)
        else:
            mid_x = x_enc
            last_x = x_enc

        last_x = self.encoder.predictor_dropout(last_x)

        f2 = mid_x.transpose(1, 2).unsqueeze(-1)
        f3 = last_x.transpose(1, 2).unsqueeze(-1)
        pooled = last_x.mean(dim=1)

        logits = self.classifier(last_x).mean(dim=1)

        if is_feat:
            return [f0, f1, f2, f3, pooled], logits
        return logits


def beats_base(**kwargs):
    return BEATsWrapper(**kwargs)
