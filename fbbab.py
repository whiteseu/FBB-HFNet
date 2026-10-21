import torch
import torch.nn as nn
import torch.nn.functional as F


def decompose_prediction(logit):
    p = 2.0 * torch.sigmoid(logit) - 1.0
    m_fg = torch.clamp(p, min=0.0)
    m_bg = torch.clamp(-p, min=0.0)
    m_cg = 1.0 - p.abs()
    return torch.cat([m_fg, m_bg, m_cg], dim=1)


class FBBAB(nn.Module):
    def __init__(self, in_channels, out_channels, dk, eps):
        super().__init__()
        self.dk = dk
        self.eps = eps
        self.to_q = nn.Conv2d(in_channels, dk, 1)
        self.to_k = nn.Conv2d(in_channels, dk, 1)
        self.to_v = nn.Conv2d(in_channels, dk, 1)
        self.proj_ctx = nn.Conv2d(dk, in_channels, 1)
        self.fuse = nn.Conv2d(2 * in_channels, out_channels, 1)

    @staticmethod
    def regional_prototypes(feat, prob, eps):
        b, c, h, w = feat.shape
        prob_flat = prob.reshape(b, 3, h * w)
        feat_flat = feat.reshape(b, c, h * w).permute(0, 2, 1)
        return torch.bmm(prob_flat, feat_flat) / (
            prob_flat.sum(dim=-1, keepdim=True) + eps)

    def forward(self, feat, logit):
        b, c, h, w = feat.shape
        logit = F.interpolate(logit, size=(h, w), mode='bilinear',
                              align_corners=False)
        prob = decompose_prediction(logit)
        proto = self.regional_prototypes(feat, prob, self.eps)
        proto_map = proto.permute(0, 2, 1).unsqueeze(-1)

        q = self.to_q(feat).reshape(b, self.dk, h * w).permute(0, 2, 1)
        k = self.to_k(proto_map).reshape(b, self.dk, 3)
        v = self.to_v(proto_map).reshape(b, self.dk, 3).permute(0, 2, 1)

        attn = torch.softmax(torch.bmm(q, k) * self.dk ** -0.5, dim=-1)
        ctx = torch.bmm(attn, v).permute(0, 2, 1).contiguous()
        ctx = self.proj_ctx(ctx.reshape(b, self.dk, h, w))
        return self.fuse(torch.cat([feat, ctx], dim=1))
