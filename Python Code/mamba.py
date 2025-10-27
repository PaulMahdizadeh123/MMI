# File: mamba.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, **factory_kwargs).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, x):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_inner)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = x.permute(0, 2, 1)  # shape (b, d_inner, l)
        x = self.conv1d(x)[:, :, :l]
        x = x.permute(0, 2, 1)  # shape (b, l, d_inner)

        x = self.act(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)  # shape (b, l, d_model)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #   A, D are input independent (see Mamba paper [1] Figure 3)
        #   ∆, B, C are input dependent (see [1] Figure 4)

        A = -torch.exp(self.A_log.float())  # shape (d_inner, d_state)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*d_state)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)  # dt, B, C
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_inner)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in [1] Algorithm 2

        return y

    def selective_scan(self, u, delta, A, B, C, D):

        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - dt is the timestep here, which is the incremental step in time
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (b, l, d_in, n)
        deltaB_u = (delta.unsqueeze(-1) * B.unsqueeze(2)) * u.unsqueeze(-1)  # (b, l, d_in, n) * (b, l, d_in, 1) -> (b, l, d_in, n)

        # Perform selective scan (see scan_SSM() in [1] Algorithm 2)
        x = torch.zeros((b, d_in, n), device=u.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = (x * C[:, i].unsqueeze(-1)).sum(dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y
