# File: train.py

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import ProposedMethod
from losses import LossA, LossB

if __name__ == "__main__":
    # Example usage
    model = ProposedMethod()
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    # Load datasets, train phase A and B separately as per the paper
    # For phase A: train estimator and restorer
    # For phase B: train fusion
    print("Model ready.")
