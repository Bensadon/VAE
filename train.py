"""
Script principal de treinamento, validação, teste e registro de experimentos do VAE.
Garante:
- Treinamento em lote fixo de 64 imagens
- Atualizações via otimizador Adam
- Monitoramento automático de épocas gravando em CSV
- Atualização contínua de gráficos de curvas de perda (treino, val, teste)
- Salvamento automático de checkpoint no ponto de menor erro de validação
- Geração e salvamento de imagens reconstruídas (Original vs Reconstruída) a cada novo recorde de validação
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from vae.config import VAEConfig, load_config
from vae.dataset import get_dataloaders, select_fixed_eval_samples
from vae.loss import vae_loss_function
from vae.model import VAE


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento do Variational Autoencoder (VAE)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="Caminho para o arquivo JSON de configuração (ex: configs/default_config.json)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Substitui o número de épocas especificado no JSON (opcional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de execução ('cuda', 'cpu' ou 'auto')",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Diretório contendo os arquivos IDX do MNIST",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Diretório raiz onde os experimentos serão salvos",
    )
    return parser.parse_args()


def setup_device(requested_device: str) -> torch.device:
    """Configura o dispositivo de execução com preferência para GPU CUDA."""
    req = requested_device.lower().strip() if requested_device else "auto"
    if req in ["cuda", "gpu"]:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[Aviso] CUDA solicitado mas não disponível. Usando CPU.")
        return torch.device("cpu")
    elif req == "cpu":
        return torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Define a semente aleatória para reprodutibilidade."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Avalia o modelo em um DataLoader e retorna (total_loss, recon_loss, kl_loss)."""
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_size = images.size(0)
            recon_images, mu, logvar = model(images)
            loss, recon, kl = vae_loss_function(recon_images, images, mu, logvar)

            total_loss_sum += loss.item() * batch_size
            recon_loss_sum += recon.item() * batch_size
            kl_loss_sum += kl.item() * batch_size
            total_samples += batch_size

    avg_total = total_loss_sum / total_samples
    avg_recon = recon_loss_sum / total_samples
    avg_kl = kl_loss_sum / total_samples
    return avg_total, avg_recon, avg_kl


def save_reconstruction_grid(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
    title: str,
) -> None:
    """
    Salva uma grade comparativa Original vs Reconstruída.
    Multiplica os valores [0.0, 1.0] da Sigmoid por 255 para recuperar a escala uint8.
    """
    n_samples = min(16, len(original_images))
    orig = original_images[:n_samples].detach().cpu().numpy()
    recon = reconstructed_images[:n_samples].detach().cpu().numpy()
    lbls = labels[:n_samples].detach().cpu().numpy()

    # Desnormalização: [0, 1] * 255.0 -> uint8 [0, 255]
    orig_uint8 = np.clip(orig * 255.0, 0, 255).astype(np.uint8).reshape(-1, 28, 28)
    recon_uint8 = np.clip(recon * 255.0, 0, 255).astype(np.uint8).reshape(-1, 28, 28)

    fig, axes = plt.subplots(2, n_samples, figsize=(1.8 * n_samples, 4.0))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    for i in range(n_samples):
        # Linha 1: Imagem Original
        axes[0, i].imshow(orig_uint8[i], cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"Orig: {lbls[i]}", fontsize=9)
        axes[0, i].axis("off")

        # Linha 2: Imagem Reconstruída
        axes[1, i].imshow(recon_uint8[i], cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_title(f"Recon", fontsize=9)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Reconstruído", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history: List[Dict[str, float]], save_path: Path) -> None:
    """
    Gera e salva o gráfico com a evolução das curvas de erro ao longo das épocas:
    1. Perda Total (Treino vs Validação vs Teste)
    2. MSE de Reconstrução (soma dos 784 pixels)
    3. Divergência KL
    """
    if len(history) == 0:
        return

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Perda Total
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Treino", color="#1f77b4", marker="o", markersize=3)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Validação", color="#ff7f0e", marker="s", markersize=3)
    axes[0].plot(epochs, [h["test_loss"] for h in history], label="Teste", color="#2ca02c", linestyle="--", alpha=0.8)
    axes[0].set_title("Perda Total (ELBO: MSE + KL)", fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda Total")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Subplot 2: MSE de Reconstrução (Soma dos 784 pixels)
    axes[1].plot(epochs, [h["train_recon"] for h in history], label="Treino (MSE 784)", color="#1f77b4", marker="o", markersize=3)
    axes[1].plot(epochs, [h["val_recon"] for h in history], label="Validação (MSE 784)", color="#ff7f0e", marker="s", markersize=3)
    axes[1].plot(epochs, [h["test_recon"] for h in history], label="Teste (MSE 784)", color="#2ca02c", linestyle="--", alpha=0.8)
    axes[1].set_title("Reconstrução: Soma dos Erros Quadráticos (784)", fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MSE (Soma por Amostra)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Subplot 3: Divergência KL
    axes[2].plot(epochs, [h["train_kl"] for h in history], label="Treino (KL)", color="#1f77b4", marker="o", markersize=3)
    axes[2].plot(epochs, [h["val_kl"] for h in history], label="Validação (KL)", color="#ff7f0e", marker="s", markersize=3)
    axes[2].plot(epochs, [h["test_kl"] for h in history], label="Teste (KL)", color="#2ca02c", linestyle="--", alpha=0.8)
    axes[2].set_title("Divergência KL (Latente)", fontweight="bold")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("KL Divergence")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_arguments()

    # 1. Carrega e valida a configuração JSON
    config = load_config(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.device is not None:
        config.device = args.device

    device = setup_device(config.device)
    set_seed(config.seed)

    # 2. Prepara diretório do experimento
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / f"{timestamp}_{config.experiment_name}"
    checkpoints_dir = exp_dir / "checkpoints"
    reconstructions_dir = exp_dir / "reconstructions"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reconstructions_dir.mkdir(parents=True, exist_ok=True)

    # Salva cópia exata do JSON de configuração usado
    config.save_json(exp_dir / "config.json")

    print("=" * 70)
    print("INICIANDO EXPERIMENTO VAE")
    print(f"Diretório do Experimento: {exp_dir}")
    print(f"Dispositivo:              {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Dimensão de Entrada:      {config.input_dim} (MNIST 28x28)")
    print(f"Camadas Ocultas Encoder:  {config.hidden_layers}")
    print(f"Espaço Latente:           {config.latent_dim}")
    print(f"Ativação Interna:         {config.activation.upper()}")
    print(f"Ativação de Saída:        {config.output_activation.upper()} (Fixa)")
    print(f"Otimizador:               {config.optimizer.upper()} (LR = {config.learning_rate})")
    print(f"Tamanho do Lote:          {config.batch_size} (Fixo)")
    print(f"Função de Custo:          Soma Bruta MSE 784 + Divergência KL")
    print(f"Número de Épocas:         {config.epochs}")
    print("=" * 70)

    # 3. Carrega os DataLoaders
    train_loader, val_loader, test_loader, datasets = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Exemplos de Treino:     {len(datasets['train'])} ({len(train_loader)} lotes de {config.batch_size})")
    print(f"Exemplos de Validação:  {len(datasets['val'])} ({len(val_loader)} lotes de {config.batch_size})")
    print(f"Exemplos de Teste:      {len(datasets['test'])} ({len(test_loader)} lotes de {config.batch_size})")

    # Amostras fixas para visualização consistente da reconstrução
    val_fixed_imgs, val_fixed_lbls = select_fixed_eval_samples(datasets["val"], num_samples=16, seed=42)
    test_fixed_imgs, test_fixed_lbls = select_fixed_eval_samples(datasets["test"], num_samples=16, seed=123)

    # 4. Inicializa o Modelo VAE e Otimizador Adam
    model = VAE(config).to(device)
    is_mirrored, mirror_details = model.get_mirrored_check()
    print("\nVerificação de Espelhamento do Modelo:")
    for d in mirror_details:
        print(f"  • {d}")
    if not is_mirrored:
        raise RuntimeError("Erro crítico: O modelo não é estritamente espelhado!")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_params.betas,
        eps=config.adam_params.eps,
        weight_decay=config.adam_params.weight_decay,
    )

    # 5. Arquivo de Métricas CSV
    metrics_csv_path = exp_dir / "metrics.csv"
    csv_headers = [
        "epoch",
        "train_loss",
        "train_recon",
        "train_kl",
        "val_loss",
        "val_recon",
        "val_kl",
        "test_loss",
        "test_recon",
        "test_kl",
        "is_best_val",
    ]
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # 6. Loop de Treinamento
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_samples = 0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            batch_sz = images.size(0)

            optimizer.zero_grad()
            recon_images, mu, logvar = model(images)
            loss, recon, kl = vae_loss_function(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_sz
            train_recon_sum += recon.item() * batch_sz
            train_kl_sum += kl.item() * batch_sz
            train_samples += batch_sz

        epoch_train_loss = train_loss_sum / train_samples
        epoch_train_recon = train_recon_sum / train_samples
        epoch_train_kl = train_kl_sum / train_samples

        # Avaliação em Validação e Teste
        epoch_val_loss, epoch_val_recon, epoch_val_kl = evaluate(model, val_loader, device)
        epoch_test_loss, epoch_test_recon, epoch_test_kl = evaluate(model, test_loader, device)

        is_best = epoch_val_loss < best_val_loss

        # Registro no histórico
        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_recon": epoch_train_recon,
            "train_kl": epoch_train_kl,
            "val_loss": epoch_val_loss,
            "val_recon": epoch_val_recon,
            "val_kl": epoch_val_kl,
            "test_loss": epoch_test_loss,
            "test_recon": epoch_test_recon,
            "test_kl": epoch_test_kl,
            "is_best_val": 1 if is_best else 0,
        }
        history.append(epoch_record)

        # Grava linha no CSV
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch_record[h] for h in csv_headers])

        # Atualiza o gráfico de curvas de perda (a cada época se <= 100 épocas, ou se for melhoria/periódico em runs longos)
        should_plot = (
            config.epochs <= 100 or
            is_best or
            epoch % max(1, config.epochs // 100) == 0 or
            epoch == config.epochs
        )
        if should_plot:
            plot_loss_curves(history, exp_dir / "loss_curve.png")

        # Exibição no console
        should_print = (
            config.epochs <= 100 or
            is_best or
            epoch % max(1, config.epochs // 100) == 0 or
            epoch == 1 or
            epoch == config.epochs
        )
        if should_print:
            star = " ★ [MELHOR VALIDAÇÃO]" if is_best else ""
            print(
                f"Época [{epoch:04d}/{config.epochs:04d}] "
                f"Treino: {epoch_train_loss:.2f} (MSE={epoch_train_recon:.2f}, KL={epoch_train_kl:.2f}) | "
                f"Val: {epoch_val_loss:.2f} (MSE={epoch_val_recon:.2f}, KL={epoch_val_kl:.2f}) | "
                f"Teste: {epoch_test_loss:.2f}{star}"
            )

        # 7. Quando atingir o melhor erro de validação:
        # Salva checkpoint completo e imagens reconstruídas
        if is_best:
            best_val_loss = epoch_val_loss
            best_epoch = epoch

            # Salva o checkpoint com pesos e metadados
            checkpoint_path = checkpoints_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_recon": epoch_val_recon,
                    "val_kl": epoch_val_kl,
                    "test_loss": epoch_test_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                },
                checkpoint_path,
            )

            # Reconstrução das imagens de validação fixas
            model.eval()
            with torch.no_grad():
                val_dev = val_fixed_imgs.to(device)
                val_recon_dev, _, _ = model(val_dev)
                val_save_path = reconstructions_dir / f"best_val_epoch_{epoch:03d}_loss_{epoch_val_loss:.2f}.png"
                save_reconstruction_grid(
                    original_images=val_fixed_imgs,
                    reconstructed_images=val_recon_dev,
                    labels=val_fixed_lbls,
                    save_path=val_save_path,
                    title=f"Reconstrução Validação - Época {epoch} (Val Loss: {epoch_val_loss:.2f})",
                )

                # Reconstrução das imagens de teste fixas
                test_dev = test_fixed_imgs.to(device)
                test_recon_dev, _, _ = model(test_dev)
                test_save_path = reconstructions_dir / f"best_test_epoch_{epoch:03d}_loss_{epoch_test_loss:.2f}.png"
                save_reconstruction_grid(
                    original_images=test_fixed_imgs,
                    reconstructed_images=test_recon_dev,
                    labels=test_fixed_lbls,
                    save_path=test_save_path,
                    title=f"Reconstrução Teste - Época {epoch} (Test Loss: {epoch_test_loss:.2f})",
                )

            print(f"   ↳ Checkpoint gravado em: {checkpoint_path.name}")
            print(f"   ↳ Imagens salvas: {val_save_path.name} e {test_save_path.name}")

    print("\n" + "=" * 70)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"Melhor Época de Validação: {best_epoch} (Val Loss = {best_val_loss:.2f})")
    print(f"Checkpoint final:          {checkpoints_dir / 'best_model.pt'}")
    print(f"Gráfico de Perdas:         {exp_dir / 'loss_curve.png'}")
    print(f"Tabela de Métricas:        {metrics_csv_path}")
    print(f"Reconstruções Visuais em:  {reconstructions_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
