from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class VAEConfig:
    # If arch == 'mlp', set input_dim.
    # If arch == 'conv2d', set input_shape = (channels, freq_bins, frames).
    arch: str = "mlp"  # mlp | conv2d
    input_dim: int | None = None
    input_shape: tuple[int, int, int] | None = None
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (256, 128)
    conv_channels: tuple[int, ...] = (32, 64, 128)
    beta: float = 1.0

    # Conditional / disentanglement options
    condition_dim: int = 0  # if > 0, model expects conditioning vector c
    deterministic: bool = False  # if True, z=mu (no sampling noise)
    use_kl: bool = True  # if False, trains as plain autoencoder (no KL term)


class MLPVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.input_dim is None:
            raise ValueError("MLPVAE requires cfg.input_dim")

        enc_layers = []
        last = cfg.input_dim + int(cfg.condition_dim)
        for h in cfg.hidden_dims:
            enc_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(last, cfg.latent_dim)
        self.fc_logvar = nn.Linear(last, cfg.latent_dim)

        dec_layers = []
        last = cfg.latent_dim + int(cfg.condition_dim)
        for h in reversed(cfg.hidden_dims):
            dec_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        dec_layers += [nn.Linear(last, cfg.input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor, c: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.condition_dim and c is None:
            raise ValueError("Conditional VAE requires conditioning vector c")
        if c is not None:
            x = torch.cat([x, c], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if bool(self.cfg.deterministic):
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if self.cfg.condition_dim and c is None:
            raise ValueError("Conditional VAE requires conditioning vector c")
        if c is not None:
            z = torch.cat([z, c], dim=1)
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar, z


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    use_kl: bool = True,
) -> torch.Tensor:
    # MSE reconstruction loss
    recon = torch.mean((recon_x - x) ** 2)
    # KL divergence between N(mu, sigma) and N(0,1)
    if use_kl:
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl
    return recon


class ConvVAE2D(nn.Module):
    """Convolutional VAE for 2D time-frequency inputs.

    Expects input of shape (N, C, F, T).
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.input_shape is None:
            raise ValueError("ConvVAE2D requires cfg.input_shape")

        base_in_ch, _, _ = cfg.input_shape
        cond_ch = int(cfg.condition_dim)
        enc_in_ch = base_in_ch + cond_ch

        enc_layers: list[nn.Module] = []
        prev = enc_in_ch
        for ch in cfg.conv_channels:
            enc_layers.append(nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            enc_layers.append(nn.ReLU())
            prev = ch
        self.encoder = nn.Sequential(*enc_layers)

        # Determine flattened size after convs
        with torch.no_grad():
            dummy = torch.zeros((1, enc_in_ch, cfg.input_shape[1], cfg.input_shape[2]), dtype=torch.float32)
            h = self.encoder(dummy)
            self._enc_out_shape = tuple(int(x) for x in h.shape[1:])  # (C, F, T)
            flat_dim = int(np.prod(self._enc_out_shape))

        self.fc_mu = nn.Linear(flat_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, cfg.latent_dim)

        self.fc_dec = nn.Linear(cfg.latent_dim + cond_ch, flat_dim)

        dec_layers = []
        chs = list(cfg.conv_channels)
        # reverse conv stack: last channel -> ... -> first channel
        for i in range(len(chs) - 1, 0, -1):
            dec_layers.append(
                nn.ConvTranspose2d(
                    chs[i],
                    chs[i - 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            dec_layers.append(nn.ReLU())

        # final layer back to input channels
        dec_layers.append(
            nn.ConvTranspose2d(
                chs[0],
                base_in_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )
        self.decoder = nn.Sequential(*dec_layers)

        self._base_in_ch = int(base_in_ch)
        self._cond_ch = int(cond_ch)

    def encode(self, x: torch.Tensor, c: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cond_ch and c is None:
            raise ValueError("Conditional VAE requires conditioning vector c")
        if c is not None:
            # Broadcast (N, cond_dim) -> (N, cond_dim, F, T)
            f, t = int(x.shape[2]), int(x.shape[3])
            c_img = c[:, :, None, None].expand(-1, -1, f, t)
            x = torch.cat([x, c_img], dim=1)
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if bool(self.cfg.deterministic):
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if self._cond_ch and c is None:
            raise ValueError("Conditional VAE requires conditioning vector c")
        if c is not None:
            z = torch.cat([z, c], dim=1)
        h = self.fc_dec(z)
        h = h.view(z.size(0), *self._enc_out_shape)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar, z


@torch.no_grad()
def encode_dataset(
    model: nn.Module,
    x: np.ndarray,
    c: np.ndarray | None = None,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    model.to(device)

    zs: list[np.ndarray] = []
    n = x.shape[0]
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).float().to(device)
        cb = None
        if c is not None:
            cb = torch.from_numpy(c[i : i + batch_size]).float().to(device)

        # Both MLPVAE and ConvVAE2D implement encode/reparameterize
        mu, logvar = model.encode(xb, cb)  # type: ignore[attr-defined]
        z = model.reparameterize(mu, logvar)  # type: ignore[attr-defined]
        zs.append(z.detach().cpu().numpy())

    return np.concatenate(zs, axis=0)
