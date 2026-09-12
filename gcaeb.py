import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return self.gate(x)


class SubSpaceAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden):
        super().__init__()
        self.split_a = in_channels // 2
        self.split_b = in_channels - self.split_a
        self.conv_a = nn.Conv2d(self.split_a, self.split_a, 1)
        self.relu = nn.ReLU(inplace=True)
        self.ca_p = ChannelAttention(in_channels, hidden)
        self.ca_q = ChannelAttention(in_channels, hidden)
        self.fuse = nn.Conv2d(2 * in_channels, out_channels, 1)

    def forward(self, z):
        z_a, z_b = torch.split(z, [self.split_a, self.split_b], dim=1)
        z_a = self.relu(self.conv_a(z_a))                            # Eq. (8)
        p = torch.cat([z_b, z_a], dim=1)                             # Eq. (9)
        q = torch.cat([z_a, z_b], dim=1)
        p = self.ca_p(p) * p                                         # Eq. (10)
        q = self.ca_q(q) * q
        return self.fuse(torch.cat([p, q], dim=1))                   # Eq. (11)


class GCAEB(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, num_blocks=3):
        super().__init__()
        self.branch0 = nn.Conv2d(in_channels, out_channels, 1)
        self.blocks = nn.ModuleList([
            SubSpaceAttentionBlock(in_channels, out_channels, hidden)
            for _ in range(num_blocks)])
        self.conv_cat = nn.Conv2d((num_blocks + 1) * out_channels,
                                  out_channels, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branches = [self.branch0(x)] + [blk(x) for blk in self.blocks]  # Eq. (5)
        b_cat = self.conv_cat(torch.cat(branches, dim=1))               # Eq. (6)
        return self.relu(b_cat + self.conv_res(x))                      # Eq. (7)
