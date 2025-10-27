# File: losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_gradient(input):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(input.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    gx = F.conv2d(input, sobel_x.repeat(input.shape[1], 1, 1, 1), padding=1, groups=input.shape[1])
    gy = F.conv2d(input, sobel_y.repeat(input.shape[1], 1, 1, 1), padding=1, groups=input.shape[1])
    grad = torch.sqrt(gx**2 + gy**2)
    return grad

def gaussian_window(size, sigma):
    kernel = torch.tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)], dtype=torch.float32)
    return kernel / kernel.sum()

def create_window(size, channel, device):
    _1D = gaussian_window(size, 1.5).unsqueeze(1).to(device)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D.expand(channel, 1, size, size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.shape[1]
    window = create_window(window_size, channel, img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])

class LossA(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=1.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, bar_l, i_en, i_gt):
        # Assume L_hat = 1 / bar_l, L_gt = i1 / i_gt but not used, perhaps adjust if needed
        # The text has L, \hat{L}, but perhaps bar_L is \hat{L}
        l_illum = self.lambda1 * F.l1_loss(bar_l, torch.ones_like(bar_l))  # assuming \bar{L} approx 1/L, but adjust
        l_recon = self.lambda2 * F.l1_loss(i_en, i_gt)
        grad_l = get_gradient(bar_l)
        l_reg = self.lambda3 * F.l1_loss(grad_l, torch.zeros_like(grad_l))
        return l_illum + l_recon + l_reg

class LossB(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=1.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, i_f, i1, i2, i_gt=None):
        l_ssim = self.lambda1 * (0.5 * (1 - ssim(i_f, i1)) + 0.5 * (1 - ssim(i_f, i2)))
        grad_f = get_gradient(i_f)
        max_grad = torch.max(get_gradient(i1), get_gradient(i2))
        l_text = self.lambda2 * F.l1_loss(grad_f, max_grad) / (i_f.shape[2] * i_f.shape[3])
        m = (i1 + i2) / 2
        l_int = self.lambda3 * F.l1_loss(i_f, m) / (i_f.shape[2] * i_f.shape[3])
        return l_ssim + l_text + l_int
