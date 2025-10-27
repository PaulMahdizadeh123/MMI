# File: modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import Mamba

class IAI(nn.Module):
    def __init__(self, channels, heads=8):
        super().__init__()
        self.heads = heads
        self.dk = channels // heads
        self.scale = self.dk ** -0.5
        self.q_proj = nn.Conv2d(channels, channels, 1)  # Adjusted for F_lu channels, but since we project in IISSM
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, f_lu):
        b, c, h, w = x.shape

        q = self.q_proj(f_lu).view(b, self.heads, self.dk, h * w).permute(0, 1, 3, 2)  # b heads (h w) dk

        k = self.k_proj(x).view(b, self.heads, self.dk, h * w).permute(0, 1, 3, 2)

        v = self.v_proj(x).view(b, self.heads, self.dk, h * w).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # b heads (h w) dk

        out = out.transpose(2, 3).reshape(b, c, h, w)

        out = self.out_proj(out)

        return out

class DSFS(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.mamba = Mamba(d_model, d_state=d_state)

    def forward(self, x):
        b, c, h, w = x.shape

        l = h * w

        y_list = []

        for flip_dims in [[], [3], [2], [2, 3]]:
            z = x

            if flip_dims:
                z = torch.flip(z, flip_dims)

            z = z.view(b, c, l).permute(0, 2, 1)  # b l c

            y = self.mamba(z)

            y = y.permute(0, 2, 1).view(b, c, h, w)

            if flip_dims:
                y = torch.flip(y, flip_dims)

            y_list.append(y)

        return sum(y_list) / len(y_list)

class IISSM(nn.Module):
    def __init__(self, channels, base_c=32, heads=8, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Conv2d(base_c, channels, 1)
        self.iai = IAI(channels, heads)
        self.dsfs = DSFS(channels, d_state)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, f_lu):
        b, c, h, w = x.shape

        res = x

        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)

        x = x.permute(0, 3, 1, 2)

        f = F.interpolate(f_lu, size=(h, w), mode='bilinear', align_corners=False)

        f = self.q_proj(f)

        x = self.iai(x, f)

        x = self.dsfs(x)

        x = self.out(x) + res

        return x

class MambaUNet(nn.Module):
    def __init__(self, base_c=32, heads=8):
        super().__init__()

        d_states = [16, 32, 64]

        self.initial_conv = nn.Conv2d(3, base_c, 3, stride=2, padding=1)

        self.iissm0 = IISSM(base_c, base_c, heads, d_states[0])

        self.down1 = nn.Conv2d(base_c, 2 * base_c, 4, stride=2, padding=1)

        self.iissm1 = IISSM(2 * base_c, base_c, heads, d_states[1])

        self.down2 = nn.Conv2d(2 * base_c, 4 * base_c, 4, stride=2, padding=1)

        self.iissm2 = IISSM(4 * base_c, base_c, heads, d_states[2])

        self.up1 = nn.ConvTranspose2d(4 * base_c, 2 * base_c, 2, stride=2)

        self.conv1x1_1 = nn.Conv2d(4 * base_c, 2 * base_c, 1)

        self.iissm_up1 = IISSM(2 * base_c, base_c, heads, d_states[1])

        self.up2 = nn.ConvTranspose2d(2 * base_c, base_c, 2, stride=2)

        self.conv1x1_2 = nn.Conv2d(2 * base_c, base_c, 1)

        self.iissm_up2 = IISSM(base_c, base_c, heads, d_states[0])

        self.final_conv = nn.ConvTranspose2d(base_c, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, i_lu, f_lu):
        x = self.initial_conv(i_lu)

        skip0 = self.iissm0(x, f_lu)

        x = self.down1(skip0)

        skip1 = self.iissm1(x, f_lu)

        x = self.down2(skip1)

        x = self.iissm2(x, f_lu)

        x = self.up1(x)

        x = torch.cat([x, skip1], dim=1)

        x = self.conv1x1_1(x)

        x = self.iissm_up1(x, f_lu)

        x = self.up2(x)

        x = torch.cat([x, skip0], dim=1)

        x = self.conv1x1_2(x)

        x = self.iissm_up2(x, f_lu)

        x = self.final_conv(x)

        return x
