"""Small time-conditioned UNet for 28x28 MNIST.

The same backbone is used for both methods. For Mean Flow (`use_r=True`) an
extra sinusoidal embedding of the start time `r` is added to the time
embedding, so the model learns u(x, r, t) instead of v(x, t). All other
architecture and parameter counts are effectively identical.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, scale=1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t):
        # t: (B,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * self.scale * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, ch_out)
        self.skip = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)          # (B, HW, C)
        k = k.reshape(b, c, h * w)                          # (B, C, HW)
        v = v.reshape(b, c, h * w).transpose(1, 2)          # (B, HW, C)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class SmallUNet(nn.Module):
    """3-level UNet: 28x28 -> 14x14 -> 7x7 with attention at 7x7."""

    def __init__(self, in_ch=1, base_ch=64, ch_mult=(1, 2, 2), time_dim=256, use_r=False):
        super().__init__()
        c1, c2, c3 = (base_ch * m for m in ch_mult)
        self.use_r = use_r

        self.time_embed = SinusoidalEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        if use_r:
            self.r_embed = SinusoidalEmbedding(time_dim)
            self.r_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
            )

        # encoder
        self.conv_in = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock(c1, c1, time_dim), ResBlock(c1, c1, time_dim)])
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)          # 28 -> 14
        self.enc2 = nn.ModuleList([ResBlock(c2, c2, time_dim), ResBlock(c2, c2, time_dim)])
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)          # 14 -> 7
        self.enc3 = nn.ModuleList([ResBlock(c3, c3, time_dim), ResBlock(c3, c3, time_dim)])
        self.attn3 = Attention(c3)

        # bottleneck
        self.mid1 = ResBlock(c3, c3, time_dim)
        self.mid_attn = Attention(c3)
        self.mid2 = ResBlock(c3, c3, time_dim)

        # decoder
        self.up1 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)   # 7 -> 14
        self.dec2 = nn.ModuleList([ResBlock(2 * c2, c2, time_dim), ResBlock(c2, c2, time_dim)])
        self.up2 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)   # 14 -> 28
        self.dec1 = nn.ModuleList([ResBlock(2 * c1, c1, time_dim), ResBlock(c1, c1, time_dim)])

        self.norm_out = nn.GroupNorm(8, c1)
        self.conv_out = nn.Conv2d(c1, in_ch, 3, padding=1)

    def forward(self, x, t, r=None):
        """x: (B, C, 28, 28); t: (B,) in [0,1]; r: (B,) in [0,1] (mean flow only)."""
        temb = self.time_mlp(self.time_embed(t))
        if self.use_r:
            assert r is not None, "use_r=True requires r"
            temb = temb + self.r_mlp(self.r_embed(r))

        h = self.conv_in(x)
        for blk in self.enc1:
            h = blk(h, temb)
        s1 = h                                            # 28x28 skip
        h = self.down1(h)
        for blk in self.enc2:
            h = blk(h, temb)
        s2 = h                                            # 14x14 skip
        h = self.down2(h)
        for blk in self.enc3:
            h = blk(h, temb)
        h = self.attn3(h)

        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)

        h = self.up1(h)
        h = torch.cat([h, s2], dim=1)
        for blk in self.dec2:
            h = blk(h, temb)
        h = self.up2(h)
        h = torch.cat([h, s1], dim=1)
        for blk in self.dec1:
            h = blk(h, temb)

        return self.conv_out(F.silu(self.norm_out(h)))
