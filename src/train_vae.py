from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .vae import ConvVAE2D, MLPVAE, VAEConfig, vae_loss


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"


def train_vae(x_train: np.ndarray, vae_cfg: VAEConfig, train_cfg: TrainConfig, c_train: np.ndarray | None = None):
    torch.manual_seed(42)

    arch = (vae_cfg.arch or "mlp").lower()
    if arch == "mlp":
        model = MLPVAE(vae_cfg).to(train_cfg.device)
    elif arch == "conv2d":
        model = ConvVAE2D(vae_cfg).to(train_cfg.device)
    else:
        raise ValueError(f"Unknown VAE arch: {vae_cfg.arch}")
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    if c_train is None:
        ds = TensorDataset(torch.from_numpy(x_train).float())
    else:
        if int(c_train.shape[0]) != int(x_train.shape[0]):
            raise ValueError("c_train must have same number of rows as x_train")
        ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(c_train).float())
    dl = DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(train_cfg.epochs):
        running = 0.0
        for batch in tqdm(dl, desc=f"epoch {epoch+1}/{train_cfg.epochs}", leave=False):
            if c_train is None:
                (xb,) = batch
                cb = None
            else:
                xb, cb = batch
                cb = cb.to(train_cfg.device)

            xb = xb.to(train_cfg.device)
            recon, mu, logvar, _ = model(xb, cb)
            loss = vae_loss(recon, xb, mu, logvar, beta=vae_cfg.beta, use_kl=bool(vae_cfg.use_kl))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu()) * xb.size(0)

        avg = running / x_train.shape[0]
        if (epoch + 1) % max(1, train_cfg.epochs // 10) == 0 or epoch == 0:
            print(f"epoch={epoch+1} loss={avg:.6f}")

    return model
