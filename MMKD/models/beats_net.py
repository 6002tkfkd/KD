import torch
import torch.nn as nn

from .beats.BEATs import BEATsConfig, BEATs


class BEATsWrapper(nn.Module):
    def __init__(self, pretrained=None, num_classes=10, **kwargs):
        super().__init__()
        ckpt = torch.load(pretrained, map_location='cpu') if pretrained else None
        hyperparams = ckpt['cfg'] if ckpt is not None else dict(kwargs)
        if ckpt is None and 'input_patch_size' not in hyperparams:
            hyperparams['input_patch_size'] = 16
        cfg = BEATsConfig(hyperparams)
        if not cfg.input_patch_size or cfg.input_patch_size <= 0:
            raise ValueError('input_patch_size must be set when pretrained is None')

        self.encoder = BEATs(cfg)
        if ckpt is not None:
            self.encoder.load_state_dict(ckpt['model'], strict=False)

        self.classifier = nn.Linear(cfg.encoder_embed_dim, num_classes)
        self.num_classes = num_classes

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
