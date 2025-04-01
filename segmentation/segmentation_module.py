import torch
import torch.nn as nn
import torch.nn.functional as F
from module.aspp import ASPP

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual_conv(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class MultiScaleFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out = out1 + out3 + out5
        return self.relu(self.bn(out))

class SegmentationHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=21, patch_grid=(14, 14)):
        super().__init__()
        self.patch_grid = patch_grid
        self.embed_dim = embed_dim

        self.initial = ResidualConvBlock(embed_dim, embed_dim // 2)
        self.down1 = ResidualConvBlock(embed_dim // 2, embed_dim // 4)
        self.down2 = ResidualConvBlock(embed_dim // 4, embed_dim // 4)

        self.fusion = MultiScaleFusionBlock(embed_dim // 4)

        # ASPP 모듈 추가
        self.aspp = ASPP(in_channels=embed_dim // 4, out_channels=embed_dim // 4)

        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, N, D = x.shape
        H_p, W_p = self.patch_grid
        assert N == H_p * W_p, "Patch count doesn't match grid size"

        x = x.permute(0, 2, 1).reshape(B, D, H_p, W_p)  # (B, D, H_p, W_p)
        x = self.initial(x)        # (B, D/2, H_p, W_p)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.down1(x)          # (B, D/4, 2H_p, 2W_p)
        x = self.down2(x)          # (B, D/4, 2H_p, 2W_p)
        x = self.fusion(x)         # Multi-scale context
        x = self.aspp(x)           # ASPP 적용
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.final_conv(x)     # (B, num_classes, H, W)
        return x