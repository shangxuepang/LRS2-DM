import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


class SimpleDenoiseUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + cond_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

    def forward(self, xt, cond):
        x = torch.cat([xt, cond], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(self, cfg, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.cfg = cfg
        self.device = cfg.device

        # beta schedule & derived params
        betas = make_beta_schedule(timesteps).to(self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.model = SimpleDenoiseUNet(in_channels=3, cond_channels=512)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def forward(self, x0, Zs=None, Zc=None):
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=self.device)

        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

     
        if Zs is not None and Zc is not None:
            cond = Zs + Zc.view(B, Zc.shape[1], 1, 1).expand_as(Zs)
        elif Zs is not None:
            cond = Zs
        elif Zc is not None:
            cond = Zc.view(B, Zc.shape[1], 1, 1).expand(B, Zc.shape[1], x0.size(2), x0.size(3))
        else:
            cond = torch.zeros(B, 512, x0.size(2), x0.size(3)).to(self.device)

        pred_noise = self.model(xt, cond)
        return pred_noise, noise, t
