import torch
import torch.nn as nn
import torch.nn.functional as F


def resample(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class FAB(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.conv = nn.Conv2d(3 * channels, channels, 3, padding=1)
        self.group_conv = nn.Conv2d(channels, channels, 3, padding=1,
                                    groups=groups)

    def forward(self, f4, f3, f2):
        size = f2.shape[-2:]
        f_cat = torch.cat([resample(f4, size), resample(f3, size), f2],
                          dim=1)                                     # Eq. (17)
        f_tilde = self.group_conv(self.conv(f_cat))
        gap = F.adaptive_avg_pool2d(f_tilde, 1)
        m = torch.sigmoid(resample(gap, f_tilde.shape[-2:]))         # Eq. (18)
        return f_tilde * m + f_tilde
