# File: fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import Mamba

class SSMBranch(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp_x = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3, bias=True)
        self.dt_rank = d_model // 16
        self.d_state = d_state
        self.x_proj = nn.Linear(d_model, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32)).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, f):
        f = self.norm(f)
        x = self.mlp_x(f)
        x = x.permute(0, 2, 1)
        x = self.conv(x)[:, :, :f.shape[1]]
        x = x.permute(0, 2, 1)
        x = F.silu(x)
        y = self.ssm(x)
        return y

    def ssm(self, x):
        b, l, d = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        deltaA = delta.unsqueeze(-1) * A
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        deltaB = deltaB.sum(2)  # Adjust if needed
        h = torch.zeros(b, d, self.d_state, device=x.device)
        ys = []
        for i in range(l):
            h = deltaA[:, i] * h + deltaB[:, i]
            y = (h * C[:, i].unsqueeze(-1)).sum(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + x * D
        return y

class MMB(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.branch1 = SSMBranch(d_model, d_state)
        self.branch2 = SSMBranch(d_model, d_state)
        self.mlp_z = nn.Linear(d_model, d_model)
        self.mlp_f = nn.Linear(d_model, d_model)

    def forward(self, f1, f2, ff):
        y1 = self.branch1(f1)
        y2 = self.branch2(f2)
        z = self.mlp_z(ff)
        y1 = y1 * F.silu(z)
        y2 = y2 * F.silu(z)
        out = self.mlp_f(y1 + y2) + ff
        return out

class LocalFeatureExtraction(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_ch=32, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)  # b (h*w) embed_dim

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_ch=3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(embed_dim, out_ch, 1)

    def forward(self, x, h, w):
        b, n, d = x.shape
        h_p = h // self.patch_size
        w_p = w // self.patch_size
        x = x.reshape(b, h_p, w_p, d).permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)
        x = self.proj(x)
        return x

class ModalityEncoder(nn.Module):
    def __init__(self, embed_dim, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([Mamba(embed_dim) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def channel_exchange(f1, f2):
    assert f1.shape == f2.shape
    b, n, c = f1.shape
    f1_ex = f1.clone()
    f2_ex = f2.clone
    mask = torch.zeros(c, device=f1.device)
    mask[c // 2 :] = 1
    f1_ex[:, :, mask == 1] = f2[:, :, mask == 1]
    f2_ex[:, :, mask == 1] = f1[:, :, mask == 1]
    return f1_ex, f2_ex

class MultimodalFusion(nn.Module):
    def __init__(self, in_ch_rgb=3, in_ch_ir=1, base_ch=32, embed_dim=96, patch_size=4, num_blocks=4):
        super().__init__()
        self.local1 = LocalFeatureExtraction(in_ch_rgb, base_ch)
        self.local2 = LocalFeatureExtraction(in_ch_ir, base_ch)
        self.patch_embed1 = PatchEmbed(patch_size, base_ch, embed_dim)
        self.patch_embed2 = PatchEmbed(patch_size, base_ch, embed_dim)
        self.encoder1 = ModalityEncoder(embed_dim, num_blocks)
        self.encoder2 = ModalityEncoder(embed_dim, num_blocks)
        self.lfl_mamba1 = Mamba(embed_dim)
        self.lfl_mamba2 = Mamba(embed_dim)
        self.mmb = MMB(embed_dim)
        self.patch_unembed = PatchUnEmbed(patch_size, embed_dim, in_ch_rgb)

    def forward(self, i1, i2):
        h, w = i1.shape[2:]
        f_local1 = self.local1(i1)
        f_local2 = self.local2(i2)
        f0_1 = self.patch_embed1(f_local1)
        f0_2 = self.patch_embed2(f_local2)
        f1 = self.encoder1(f0_1)
        f2 = self.encoder2(f0_2)
        f1_ex, f2_ex = channel_exchange(f1, f2)
        f1 = self.lfl_mamba1(f1_ex)
        f2 = self.lfl_mamba2(f2_ex)
        ff = (f1 + f2) / 2
        ff = self.mmb(f1, f2, ff)
        out = self.patch_unembed(ff, h, w)
        return out
