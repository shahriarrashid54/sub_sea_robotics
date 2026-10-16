"""Compact convolutional encoder + FPN decoder sized for CPU / embedded inference.

The encoder is a residual network with four stages at strides 4, 8, 16 and 32.
The decoder is a lightweight top-down feature pyramid that produces a single
stride-4 feature map (as in CenterNet). Width is controlled by ``width``.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, act: bool = True):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = ConvBNAct(cin, cout, 3, stride)
        self.c2 = ConvBNAct(cout, cout, 3, 1, act=False)
        self.skip = nn.Identity() if (cin == cout and stride == 1) else ConvBNAct(cin, cout, 1, stride, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 1, width: float = 1.0, depths=(2, 2, 2, 2)):
        super().__init__()
        ch = [max(8, int(c * width)) for c in (16, 32, 64, 128, 192)]
        self.stem = nn.Sequential(ConvBNAct(in_ch, ch[0], 3, 2), ConvBNAct(ch[0], ch[0], 3, 1))  # stride 2
        self.stages = nn.ModuleList()
        cin = ch[0]
        for i, d in enumerate(depths):
            blocks = [ResBlock(cin, ch[i + 1], stride=2)] + [ResBlock(ch[i + 1], ch[i + 1]) for _ in range(d - 1)]
            self.stages.append(nn.Sequential(*blocks))
            cin = ch[i + 1]
        self.out_channels = ch[1:]

    def forward(self, x) -> List[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for st in self.stages:
            x = st(x)
            feats.append(x)
        return feats  # strides 4, 8, 16, 32


class FPNDecoder(nn.Module):
    """Top-down pathway that fuses all levels into a stride-4 map."""

    def __init__(self, in_channels: List[int], out_ch: int = 64):
        super().__init__()
        self.lateral = nn.ModuleList([ConvBNAct(c, out_ch, 1) for c in in_channels])
        self.smooth = nn.ModuleList([ConvBNAct(out_ch, out_ch, 3) for _ in in_channels[:-1]])
        self.out_ch = out_ch

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = self.lateral[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="nearest") + self.lateral[i](feats[i])
            x = self.smooth[i](x)
        return x
