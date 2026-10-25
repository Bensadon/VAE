"""
Script de testes automatizados e verificação de regras para o VAE:
- Verifica restrição de 784 neurônios na entrada e na saída
- Verifica espelhamento rígido de encoder e decoder para diferentes profundidades
- Verifica ativações idênticas em todos os neurônios internos (ReLU, Sigmoid, Tanh)
- Verifica saída final fixa com Sigmoid no intervalo [0.0, 1.0]
- Verifica função de custo com soma bruta de 784 pixels (sem divisão por 784)
- Verifica otimizador Adam e passo backward com batch_size = 64
- Verifica carregamento do dataset e normalização
"""

import sys
from pathlib import Path
import torch
import numpy as np

from vae.config import VAEConfig, load_config, RIGID_INPUT_DIM, RIGID_OUTPUT_DIM, RIGID_BATCH_SIZE
from vae.model import VAE
from vae.loss import vae_loss_function
from vae.dataset import get_dataloaders, MNISTIDXDataset


def test_model_mirroring_and_dimensions():
    print("▶ Testando espelhamento do modelo e dimensões 784...")
    test_architectures = [
        [512, 256],
        [400],
        [1024, 512, 256],
        [256, 128, 64],
    ]
    for arch in test_architectures:
        cfg = VAEConfig(hidden_layers=arch, latent_dim=15, activation="relu")
        model = VAE(cfg)
        is_mirrored, details = model.get_mirrored_check()
        assert is_mirrored, f"Falha no espelhamento para arquitetura: {arch}"

        # Verifica se entrada do encoder é 784
        first_linear = model.encoder[0]
        assert first_linear.in_features == 784, f"Entrada esperada 784, obteve {first_linear.in_features}"

        # Verifica se saída do decoder é 784
        last_linear = model.decoder[-2]
        assert last_linear.out_features == 784, f"Saída esperada 784, obteve {last_linear.out_features}"

        # Verifica Sigmoid na saída
        assert isinstance(model.decoder[-1], torch.nn.Sigmoid), "Última camada deve ser Sigmoid"

    print("  ✔ Dimensões 784 e espelhamento estrito validados com sucesso!")


def test_all_activations():
    print("▶ Testando funções de ativação configuráveis (ReLU, Sigmoid, Tanh)...")
    activations = ["relu", "sigmoid", "tanh"]
    for act in activations:
        cfg = VAEConfig(hidden_layers=[256, 128], latent_dim=10, activation=act)
        model = VAE(cfg)
        x = torch.rand(4, 784)
        recon, mu, logvar = model(x)

        assert recon.shape == (4, 784)
        assert mu.shape == (4, 10)
        assert logvar.shape == (4, 10)
        assert torch.all(recon >= 0.0) and torch.all(recon <= 1.0), "Saída deve estar em [0.0, 1.0]"

    print("  ✔ Todas as 3 ativações funcionam e produzem saídas válidas!")


def test_loss_function_raw_sum():
    print("▶ Testando cálculo de loss com soma bruta dos 784 pixels (sem divisão por 784)...")
    batch_size = 2
    x = torch.zeros(batch_size, 784)
    # Suponha que a reconstrução difere por 1.0 em cada um dos 784 pixels
    recon_x = torch.ones(batch_size, 784)
    mu = torch.zeros(batch_size, 10)
    logvar = torch.zeros(batch_size, 10)  # log(1) = 0, mu=0 -> KL ideal deve ser 0

    total, recon, kl = vae_loss_function(recon_x, x, mu, logvar)

    # Cada pixel tem erro quadrático (1 - 0)^2 = 1
    # Soma de 784 pixels = 784.0 por amostra
    assert torch.isclose(recon, torch.tensor(784.0)), f"MSE de reconstrução esperado 784.0, obteve {recon.item()}"
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-5), f"KL esperado 0.0, obteve {kl.item()}"
    assert torch.isclose(total, torch.tensor(784.0)), f"Total esperado 784.0, obteve {total.item()}"

    print("  ✔ Função de custo soma bruta de 784 pixels validada!")


def test_training_step_with_adam():
    print("▶ Testando passo de otimização Adam com lote 64...")
    cfg = VAEConfig(hidden_layers=[128, 64], latent_dim=8, activation="relu", learning_rate=0.001)
    model = VAE(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    x = torch.rand(64, 784)
    optimizer.zero_grad()
    recon, mu, logvar = model(x)
    loss, recon_l, kl_l = vae_loss_function(recon, x, mu, logvar)
    loss.backward()

    # Verifica se os gradientes foram calculados
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradiente nulo para parâmetro: {name}"

    optimizer.step()
    print("  ✔ Passo de treinamento completo (Forward + Backward + Adam Step) executado com sucesso!")


def test_dataset_loader():
    print("▶ Testando integridade do carregamento do dataset...")
    train_loader, val_loader, test_loader, datasets = get_dataloaders("dataset", batch_size=64, num_workers=0)
    
    assert len(datasets["train"]) == 48000, f"Treino deve ter 48000 exemplos, tem {len(datasets['train'])}"
    assert len(datasets["val"]) == 12000, f"Validação deve ter 12000 exemplos, tem {len(datasets['val'])}"
    assert len(datasets["test"]) == 10000, f"Teste deve ter 10000 exemplos, tem {len(datasets['test'])}"

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (64, 784), f"Shape esperado (64, 784), obteve {batch_x.shape}"
    assert batch_x.min() >= 0.0 and batch_x.max() <= 1.0, "Pixels devem estar normalizados em [0.0, 1.0]"
    print("  ✔ Carregamento do dataset e normalização [0, 1] validados com sucesso!")


def main():
    print("==================================================")
    print("INICIANDO SUÍTE DE TESTES E VERIFICAÇÃO DO VAE")
    print("==================================================")
    test_model_mirroring_and_dimensions()
    test_all_activations()
    test_loss_function_raw_sum()
    test_training_step_with_adam()
    test_dataset_loader()
    print("==================================================")
    print("TODOS OS TESTES PASSARAM COM SUCESSO!")
    print("==================================================")


if __name__ == "__main__":
    main()
