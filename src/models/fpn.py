import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBackbone(nn.Module):
    """
    Nhận 1 frame [B,C,H,W] -> trả [C2,C3,C4] với downsample 2,4,8 lần.
    """
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        # Stage 1: H,W
        self.conv1 = self._conv_block(in_channels, base_channels)
        # Stage 2: H/2,W/2
        self.conv2 = self._conv_block(base_channels, base_channels*2)
        # Stage 3: H/4,W/4
        self.conv3 = self._conv_block(base_channels*2, base_channels*4)

        self.pool = nn.MaxPool2d(2,2)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B,C,H,W]
        c1 = self.conv1(x)         # [B, F,   H,   W]
        p1 = self.pool(c1)         # [B, F,   H/2, W/2]
        c2 = self.conv2(p1)        # [B, 2F,  H/2, W/2]
        p2 = self.pool(c2)         # [B, 2F,  H/4, W/4]
        c3 = self.conv3(p2)        # [B, 4F,  H/4, W/4]
        return c1, c2, c3          # 3 scale

# ---------- FPN ----------
class FPN(nn.Module):
    """
    Nhận [C1 (H,W), C2 (H/2,W/2), C3 (H/4,W/4)]
    -> trả feature cuối ở full resolution H,W
    """
    def __init__(self, c1_channels, c2_channels, c3_channels, fpn_channels=64):
        super().__init__()
        # lateral convs
        self.lat_c1 = nn.Conv2d(c1_channels, fpn_channels, kernel_size=1)
        self.lat_c2 = nn.Conv2d(c2_channels, fpn_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(c3_channels, fpn_channels, kernel_size=1)

        # smooth convs
        self.smooth1 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

    def forward(self, c1, c2, c3):
        # Top-down pathway
        p3 = self.lat_c3(c3)                     # [B,F,H/4,W/4]
        p2 = self._upsample_add(p3, self.lat_c2(c2))  # [B,F,H/2,W/2]
        p1 = self._upsample_add(p2, self.lat_c1(c1))  # [B,F,H,W]

        # Smooth
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        return p1  # feature ở full resolution

    def _upsample_add(self, x, y):
        # upsample x lên size của y rồi cộng
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

class RainfallFPN(nn.Module):
    """
    Stage 2: dự đoán rainfall từ feature của 1 ngày (C,H,W)
    Input : [B,C,H,W]   (feature 1 ngày, đã scale)
    Output: [B,1,H,W]   (rainfall scaled)
    """
    def __init__(self, in_channels, base_channels=32, fpn_channels=64, dropout_rate=0.1):
        super().__init__()
        self.backbone = SimpleBackbone(in_channels=in_channels,
                                       base_channels=base_channels)

        self.fpn = FPN(
            c1_channels=base_channels,
            c2_channels=base_channels*2,
            c3_channels=base_channels*4,
            fpn_channels=fpn_channels
        )

        self.dropout = nn.Dropout2d(dropout_rate)
        self.head = nn.Conv2d(fpn_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: [B,C,H,W]  (1 ngày feature)
        """
        c1, c2, c3 = self.backbone(x)   # 3 scale
        fpn_feat = self.fpn(c1, c2, c3) # [B,fpn_channels,H,W]
        fpn_feat = self.dropout(fpn_feat)
        out = self.head(fpn_feat)       # [B,1,H,W] rainfall (scaled)
        out = torch.sigmoid(out)
        return out