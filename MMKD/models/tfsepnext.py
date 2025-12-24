import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, SqueezeExcite
except Exception as exc:
    _timm_exc = exc

    class DropPath(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ImportError('timm is required for TFSEpNeXt') from _timm_exc

    class SqueezeExcite(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ImportError('timm is required for TFSEpNeXt') from _timm_exc


class ConvClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean((-1, -2), keepdim=False)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, use_bn=True, use_relu=True):
        super().__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias)
        self._bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self._relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x) if self._bn is not None else x
        x = self._relu(x) if self._relu is not None else x
        return x


class ResNorm(nn.Module):
    def __init__(self, channels: int, lamb=0.1, eps=1e-5):
        super().__init__()
        self._eps = torch.full((1, channels, 1, 1), eps)
        self._lambda = torch.full((1, channels, 1, 1), lamb)

    def forward(self, x):
        self._eps = self._eps.to(x.device)
        self._lambda = self._lambda.to(x.device)
        identity = x
        fi_mean = x.mean((1, 3), keepdim=True)
        fi_var = x.var((1, 3), keepdim=True)
        fin = (x - fi_mean) / (fi_var + self._eps).sqrt()
        return self._lambda * identity + fin


class ShuffleLayer(nn.Module):
    def __init__(self, group: int):
        super().__init__()
        self._group = group

    def forward(self, x):
        b, c, f, t = x.data.size()
        group_channels = c // self._group
        x = x.reshape(b, group_channels, self._group, f, t)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, f, t)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class TimeFreqSepConvNeXtSE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, drop_path: float = 0.0, drop_path_concat: bool = False, expansion: int = 4, layer_scale_init_value=1e-6):
        super().__init__()
        self.drop_path_concat = drop_path_concat
        assert out_channels % 2 == 0, "Out channels must be divisible by 2"
        half_channels = out_channels // 2

        self.padding = 1
        if kernel_size == 5:
            self.padding = 2
        elif kernel_size == 7:
            self.padding = 3
        elif kernel_size == 9:
            self.padding = 4

        self.trans_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ) if in_channels != out_channels else None

        self.freq_dw_conv = nn.Conv2d(half_channels, half_channels, (kernel_size, 1), padding=(self.padding, 0), groups=half_channels)
        self.temp_dw_conv = nn.Conv2d(half_channels, half_channels, (1, kernel_size), padding=(0, self.padding), groups=half_channels)

        self.norm = LayerNorm(half_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(half_channels, expansion * half_channels)
        self.act = nn.GELU()
        self.grn = GRN(expansion * half_channels)
        self.pwconv2 = nn.Linear(expansion * half_channels, half_channels)
        self.se = SqueezeExcite(half_channels, 0.25)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.shuffle_layer = ShuffleLayer(group=half_channels)

    def forward(self, x):
        x = self.trans_conv(x) if self.trans_conv is not None else x
        x = self.shuffle_layer(x)
        x1, x2 = torch.split(x, x.data.size(1) // 2, dim=1)
        identity1 = x1
        identity2 = x2

        x1 = self.freq_dw_conv(x1)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.norm(x1)
        x1 = self.pwconv1(x1)
        x1 = self.act(x1)
        x1 = self.grn(x1)
        x1 = self.pwconv2(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.se(x1)
        if self.drop_path_concat:
            x1 = identity1 + x1
        else:
            x1 = identity1 + self.drop_path(x1)

        x2 = self.temp_dw_conv(x2)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.norm(x2)
        x2 = self.pwconv1(x2)
        x2 = self.act(x2)
        x2 = self.grn(x2)
        x2 = self.pwconv2(x2)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.se(x2)
        if self.drop_path_concat:
            x2 = identity2 + x2
        else:
            x2 = identity2 + self.drop_path(x2)

        x = torch.cat((x1, x2), dim=1)
        if self.drop_path_concat:
            x = self.drop_path(x)
        return x


class TFSEpNeXt(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 32, depth: int = 22621,
                 dropout: float = 0.1, kernel_size: int = 7, drop_path_concat: bool = False, expansion: int = 4):
        super().__init__()
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.drop_path_concat = drop_path_concat
        self.expansion = expansion

        cfg = {
            22621: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'G', 2, 2, 2, 2, 2, 2, 'N', 'M', 2.5, 2.5, 'N'],
            22622: ['N', 1, 1, 'N', 'G', 1.5, 1.5, 'N', 'M', 2, 2, 2, 2, 2, 2, 'N', 'M', 2.5, 2.5, 'N'],
            22623: ['N', 1, 1, 'N', 'G', 1.5, 1.5, 'N', 'G', 2, 2, 2, 2, 2, 2, 'N', 'M', 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 2 * base_channels, 3, stride=2, padding=1, groups=base_channels // 2),
            nn.BatchNorm2d(2 * base_channels),
            nn.GELU(),
        )

        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)

        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = width * 2
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'D':
                layers += [ConvBnRelu(in_channels=vt, out_channels=vt, kernel_size=2, stride=1, padding=1)]
            elif v == 'G':
                layers += [nn.Sequential(
                    nn.Conv2d(in_channels=vt, out_channels=vt, kernel_size=2, stride=1, padding=1),
                    nn.BatchNorm2d(vt),
                    nn.GELU(),
                )]
            elif v != vt:
                layers += [TimeFreqSepConvNeXtSE(vt, v, self.kernel_size, self.dropout, self.drop_path_concat, self.expansion)]
                vt = v
            else:
                layers += [TimeFreqSepConvNeXtSE(vt, vt, self.kernel_size, self.dropout, self.drop_path_concat, self.expansion)]
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv_layers(x)
        f0 = out

        if len(self.middle_layers) > 0:
            split1 = max(len(self.middle_layers) // 3, 1)
            split2 = max(2 * len(self.middle_layers) // 3, split1 + 1)
        else:
            split1 = split2 = 0

        feats = []
        for idx, layer in enumerate(self.middle_layers):
            out = layer(out)
            if idx == split1 - 1:
                feats.append(out)
            if idx == split2 - 1:
                feats.append(out)
        f1 = feats[0] if len(feats) > 0 else out
        f2 = feats[1] if len(feats) > 1 else out
        f3 = out

        logits = self.classifier(out)
        pooled = out.mean((-1, -2))

        if is_feat:
            return [f0, f1, f2, f3, pooled], logits
        return logits


def tfsepnext(**kwargs):
    return TFSEpNeXt(**kwargs)
