# File: utils2.py

import torch

def bright_channel(img):
    return torch.max(img, dim=1, keepdim=True)[0]
