"""
Módulo de cálculo da função de perda rígida do VAE:
- Reconstrução: Soma bruta dos erros quadráticos sobre os 784 pixels (MSE sem divisão por 784)
- Regularização: Divergência de Kullback-Leibler analítica padrão com prior N(0, I)
- Perda Total = Reconstrução (soma dos 784 pixels) + Divergência KL
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcula a perda do VAE conforme as restrições rígidas:
    
    Args:
        recon_x: Tensor (B, 784) com valores preditos em [0.0, 1.0].
        x: Tensor (B, 784) com imagens reais normalizadas em [0.0, 1.0].
        mu: Tensor (B, latent_dim) com as médias latentes.
        logvar: Tensor (B, latent_dim) com o log da variância latente.

    Returns:
        total_loss: Perda total (recon_loss + kl_loss), calculada como a média sobre as B amostras do lote.
        recon_loss: Soma bruta dos 784 erros quadráticos por amostra (sem divisão por 784), média sobre o lote.
        kl_loss: Divergência KL padrão somada sobre as dimensões latentes, média sobre o lote.
    """
    # 1. Soma bruta dos erros quadráticos sobre os 784 pixels de cada imagem (sem 1/784)
    # Shape intermediário: (B,)
    squared_diff = (recon_x - x) ** 2
    recon_per_sample = torch.sum(squared_diff, dim=1)
    recon_loss = torch.mean(recon_per_sample)

    # 2. Divergência KL analítica entre q(z|x) ~ N(mu, diag(exp(logvar))) e p(z) ~ N(0, I)
    # KL = -0.5 * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
    # Shape intermediário: (B,)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_per_sample)

    # 3. Perda Total do ELBO
    total_loss = recon_loss + kl_loss

    return total_loss, recon_loss, kl_loss
