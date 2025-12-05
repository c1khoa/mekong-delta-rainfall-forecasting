import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f = self.conv(x)
        d = self.pool(f)
        return f, d


class DecoderBlock(nn.Module):
    def __init__(self, in_ch_list, out_ch):
        super().__init__()
        self.block = ConvBlock(sum(in_ch_list), out_ch)

    def forward(self, feats, size):
        r = [F.interpolate(f, size=size, mode='bilinear', align_corners=False) for f in feats]
        x = torch.cat(r, dim=1)
        return self.block(x)


class UNet3Plus(nn.Module):
    def __init__(self, in_ch=24, out_ch=1, base=8, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.e1 = EncoderBlock(in_ch, base)
        self.e2 = EncoderBlock(base, base * 2)

        self.bottom = ConvBlock(base * 2, base * 4)

        self.d2 = DecoderBlock([base, base * 2, base * 4], base * 2)
        self.d1 = DecoderBlock([base, base * 2, base * 4, base * 2], base)

        if deep_supervision:
            self.s1 = nn.Conv2d(base, out_ch, 1)
            self.s2 = nn.Conv2d(base * 2, out_ch, 1)
            self.s3 = nn.Conv2d(base * 4, out_ch, 1)

        self.final = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        b, c, h, w = x.size()

        e1, p1 = self.e1(x)
        e2, p2 = self.e2(p1)
        e3 = self.bottom(p2)

        s1 = (h, w)
        s2 = (max(h // 2, 1), max(w // 2, 1))
        s3 = (max(h // 4, 1), max(w // 4, 1))

        d2 = self.d2([e1, e2, e3], s2)
        d1 = self.d1([e1, e2, e3, d2], s1)

        if self.deep_supervision:
            o1 = self.s1(d1)
            o2 = F.interpolate(self.s2(d2), s1, mode='bilinear', align_corners=False)
            o3 = F.interpolate(self.s3(e3), s1, mode='bilinear', align_corners=False)
            return [o1, o2, o3]

        return self.final(d1)
