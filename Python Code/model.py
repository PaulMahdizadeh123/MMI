# File: model.py

import torch
import torch.nn as nn

from modules import MambaUNet, IISSM
from fusion import MultimodalFusion
from utils import bright_channel

class IlluminationEstimator(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.merge = nn.Conv2d(4, c, 1)
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.pw = nn.Conv2d(c, c, 1)
        self.agg = nn.Conv2d(c, 3, 1)

    def forward(self, i1, l_p):
        x = torch.cat([i1, l_p], dim=1)
        x = self.merge(x)
        f_lu = self.pw(self.dw(x))
        bar_l = self.agg(f_lu)
        i_lu = i1 * bar_l
        return i_lu, f_lu

class ProposedMethod(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = IlluminationEstimator()
        self.restorer = MambaUNet()
        self.fusion = MultimodalFusion()

    def forward(self, i1, i2):
        i1_prime = bright_channel(i1)
        i2_prime = bright_channel(i2)
        l_p = torch.mean(torch.cat([i1_prime, i2_prime], dim=1), dim=1, keepdim=True)
        i_lu, f_lu = self.estimator(i1, l_p)
        i_en = self.restorer(i_lu, f_lu) + i_lu
        final = self.fusion(i_en, i2)
        return final
