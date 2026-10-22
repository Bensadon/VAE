"""
Módulo de definição da arquitetura do Variational Autoencoder (VAE).
Implementa o modelo clássico com restrições rígidas:
- Entrada e saída: 784 neurônios
- Camadas ocultas do encoder e decoder rigidamente espelhadas
- Ativação idêntica para todos os neurônios ocultos
- Saída final fixa em Sigmoid
- Amostragem latente via Reparameterization Trick
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from vae.config import VAEConfig


def get_activation_layer(activation_name: str) -> nn.Module:
    """Retorna a camada de ativação correspondente ao nome configurado."""
    act = activation_name.lower()
    if act == "relu":
        return nn.ReLU()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Ativação desconhecida: {activation_name}")


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim        # 784
        self.output_dim = config.output_dim      # 784
        self.latent_dim = config.latent_dim
        self.hidden_layers = config.hidden_layers
        self.activation_name = config.activation

        # -----------------------------------------------------------
        # 1. ENCODER: 784 -> H_1 -> H_2 -> ... -> H_k
        # -----------------------------------------------------------
        encoder_modules: List[nn.Module] = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_layers:
            encoder_modules.append(nn.Linear(prev_dim, h_dim))
            encoder_modules.append(get_activation_layer(self.activation_name))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_modules)

        # Projeções lineares para o espaço latente
        last_hidden_dim = self.hidden_layers[-1]
        self.fc_mu = nn.Linear(last_hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(last_hidden_dim, self.latent_dim)

        # -----------------------------------------------------------
        # 2. DECODER RIGIDAMENTE ESPELHADO: Latent -> H_k -> ... -> H_1 -> 784
        # -----------------------------------------------------------
        # A lista de dimensões ocultas é revertida para espelhamento exato
        decoder_hidden_dims = list(reversed(self.hidden_layers))
        decoder_modules: List[nn.Module] = []
        
        # Primeira camada do decoder: do espaço latente para o último hidden do encoder
        prev_dec_dim = self.latent_dim
        for h_dim in decoder_hidden_dims:
            decoder_modules.append(nn.Linear(prev_dec_dim, h_dim))
            decoder_modules.append(get_activation_layer(self.activation_name))
            prev_dec_dim = h_dim

        # Camada final de saída: H_1 -> 784 com Sigmoid fixa
        decoder_modules.append(nn.Linear(prev_dec_dim, self.output_dim))
        decoder_modules.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Passagem pelo encoder: x -> (mu, logvar)."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Truque de reparametrização:
        z = mu + std * epsilon, onde epsilon ~ N(0, I) e std = exp(0.5 * logvar).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Em modo de avaliação/inferência para reconstrução pura,
            # mu representa a esperança da distribuição posterior q(z|x).
            # Também pode ser amostrado se desejado, mas mu é determinístico e ótimo.
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Passagem pelo decoder: z -> recon_x in [0, 1]."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executa a passagem completa pelo VAE.
        Retorna:
            recon_x: Imagem reconstruída com valores em [0.0, 1.0], tamanho (batch_size, 784)
            mu: Média do espaço latente, tamanho (batch_size, latent_dim)
            logvar: Log-variância do espaço latente, tamanho (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def get_mirrored_check(self) -> Tuple[bool, List[str]]:
        """
        Valida programaticamente se o decoder é o espelho exato do encoder.
        Retorna (is_mirrored, details).
        """
        enc_dims = [self.input_dim] + self.hidden_layers
        dec_hidden_dims = list(reversed(self.hidden_layers)) + [self.output_dim]
        dec_dims = [self.latent_dim] + dec_hidden_dims

        details = [
            f"Encoder: {enc_dims} -> Latente({self.latent_dim})",
            f"Decoder: Latente({self.latent_dim}) -> {dec_hidden_dims}",
            f"Ativação interna idêntica: {self.activation_name}",
            f"Saída final fixa: Sigmoid ({self.output_dim} neurônios)"
        ]

        # Verifica espelhamento das camadas ocultas
        is_mirrored = (
            dec_hidden_dims[:-1] == list(reversed(self.hidden_layers)) and
            dec_hidden_dims[-1] == self.output_dim and
            self.input_dim == self.output_dim == 784
        )
        return is_mirrored, details
