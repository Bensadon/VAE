"""
Script principal de treinamento, validação, teste e registro de experimentos do VAE.

Otimizações de alta performance e desenho experimental implementadas:
1. Dataset 100% residente em VRAM com permutação única por época (index_select) e fatias contíguas (zero cópias por lote).
2. Acúmulo de perdas diretamente na GPU (elimina 2.250 sincronizações .item() por época no treino).
3. Avaliação de validação (12k) e teste (10k) de uma só vez na VRAM (transforma ~340 forwards por época em apenas 2).
4. Otimizador Adam com kernel fundido (fused=True, capturable=True) + captura de CUDA Graph do passo de treino (~0.6s/época).
5. Redução drástica de I/O: backend Matplotlib 'Agg', CSV mantido aberto em buffer, plot e last.pt apenas a cada N épocas e no fim.
6. Guarda contra NaN/divergência no treino e na avaliação com registro em summary.json.
7. Suporte nativo a max_steps (no JSON e via CLI) ou epochs=267 (~200,25k passos).
8. Validação determinística (z = mu em eval()) e split treino/val fixo independente da seed do experimento.
9. Checkpoints best_model.pt e last.pt + suporte a retomada/extensão via --resume.
10. Nomenclatura de diretório com {arch}_{act}_lat{dim}_lr{lr}_seed{s} e reporte do teste na época da melhor validação.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Backend sem GUI para evitar overhead de renderização
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from vae.config import VAEConfig, load_config
from vae.dataset import VRAMDataLoader, get_dataloaders, select_fixed_eval_samples
from vae.loss import vae_loss_function
from vae.model import VAE


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento otimizado do Variational Autoencoder (VAE)")
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
        help="Sobrescreve o número de épocas especificado no JSON (opcional)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Limite máximo de passos de gradiente (ex: 200000). Se informado, calcula as épocas equivalentes.",
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
        help="Diretório raiz onde os experimentos serão salvos (ex: experiments/round0)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nome customizado da pasta do experimento. Se omitido, usa layers{num}_{act}_lat{dim}_lr{lr}_seed{s}",
    )
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Omite o timestamp numérico no nome da pasta de saída",
    )
    parser.add_argument(
        "--with_timestamp",
        action="store_true",
        help="Força inclusão do timestamp numérico no nome da pasta de saída (mesmo para round0)",
    )
    parser.add_argument(
        "--plot_interval",
        type=int,
        default=50,
        help="Intervalo de épocas para atualizar o gráfico loss_curve.png (padrão: 50 e ao fim)",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=50,
        help="Intervalo de épocas para salvar o checkpoint last.pt (padrão: 50 e ao fim)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16384,
        help="Tamanho do super-lote para avaliação de validação e teste na GPU (padrão: 16384 = split inteiro em 1 forward)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Caminho para diretório de experimento ou arquivo last.pt para retomar/estender o treinamento",
    )
    parser.add_argument(
        "--save_intermediate_recons",
        action="store_true",
        help="Se ativado, salva imagens de reconstrução intermediárias além das finais",
    )
    parser.add_argument(
        "--no_cuda_graph",
        action="store_true",
        help="Desativa a captura de CUDA Graph no passo de treino (usa execução eager padrão)",
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
    data: Union[VRAMDataLoader, DataLoader, torch.Tensor],
    eval_batch_size: int = 16384,
) -> Tuple[float, float, float]:
    """
    Avalia o modelo de uma só vez (ou em super-lotes) sem sincronizações intermediárias CPU↔GPU.
    Como os dados já estão na VRAM e o modelo é compacto, 12.000 (val) e 10.000 (teste)
    amostras cabem em 1 único forward por split (2 forwards por época no total).
    """
    model.eval()

    if isinstance(data, torch.Tensor):
        images_tensor = data
    elif hasattr(data, "images"):
        images_tensor = data.images
    else:
        # Fallback para DataLoaders convencionais
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        total_samples = 0
        with torch.no_grad():
            for imgs, _ in data:
                b_sz = imgs.size(0)
                recon, mu, logvar = model(imgs)
                loss, recon_l, kl_l = vae_loss_function(recon, imgs, mu, logvar)
                total_loss_sum += loss.item() * b_sz
                recon_loss_sum += recon_l.item() * b_sz
                kl_loss_sum += kl_l.item() * b_sz
                total_samples += b_sz
        return total_loss_sum / total_samples, recon_loss_sum / total_samples, kl_loss_sum / total_samples

    n_samples = images_tensor.size(0)
    with torch.no_grad():
        if eval_batch_size <= 0 or eval_batch_size >= n_samples:
            recon_images, mu, logvar = model(images_tensor)
            loss, recon, kl = vae_loss_function(recon_images, images_tensor, mu, logvar)
            return torch.stack([loss, recon, kl]).tolist()

        device = images_tensor.device
        eval_sums = torch.zeros(3, device=device)
        for start_idx in range(0, n_samples, eval_batch_size):
            batch = images_tensor[start_idx : start_idx + eval_batch_size]
            b_sz = batch.size(0)
            recon_images, mu, logvar = model(batch)
            loss, recon, kl = vae_loss_function(recon_images, batch, mu, logvar)
            eval_sums += torch.stack([loss, recon, kl]) * b_sz

    # Única sincronização GPU -> CPU
    avg_total, avg_recon, avg_kl = (eval_sums / n_samples).tolist()
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
        axes[1, i].set_title("Recon", fontsize=9)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Reconstruído", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Treino", color="#1f77b4", marker="o", markersize=2)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Validação", color="#ff7f0e", marker="s", markersize=2)
    axes[0].plot(epochs, [h["test_loss"] for h in history], label="Teste", color="#2ca02c", linestyle="--", alpha=0.8)
    axes[0].set_title("Perda Total (ELBO: MSE + KL)", fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda Total")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Subplot 2: MSE de Reconstrução (Soma dos 784 pixels)
    axes[1].plot(epochs, [h["train_recon"] for h in history], label="Treino (MSE 784)", color="#1f77b4", marker="o", markersize=2)
    axes[1].plot(epochs, [h["val_recon"] for h in history], label="Validação (MSE 784)", color="#ff7f0e", marker="s", markersize=2)
    axes[1].plot(epochs, [h["test_recon"] for h in history], label="Teste (MSE 784)", color="#2ca02c", linestyle="--", alpha=0.8)
    axes[1].set_title("Reconstrução: Soma dos Erros Quadráticos (784)", fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MSE (Soma por Amostra)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Subplot 3: Divergência KL
    axes[2].plot(epochs, [h["train_kl"] for h in history], label="Treino (KL)", color="#1f77b4", marker="o", markersize=2)
    axes[2].plot(epochs, [h["val_kl"] for h in history], label="Validação (KL)", color="#ff7f0e", marker="s", markersize=2)
    axes[2].plot(epochs, [h["test_kl"] for h in history], label="Teste (KL)", color="#2ca02c", linestyle="--", alpha=0.8)
    axes[2].set_title("Divergência KL (Latente)", fontweight="bold")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("KL Divergence")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_arguments()

    # Suporte a retomada/extensão de treinamento via --resume
    resume_ckpt_path: Optional[Path] = None
    resume_exp_dir: Optional[Path] = None
    if args.resume is not None:
        r_path = Path(args.resume)
        if r_path.is_dir():
            resume_exp_dir = r_path
            resume_ckpt_path = r_path / "checkpoints" / "last.pt"
        else:
            resume_ckpt_path = r_path
            resume_exp_dir = r_path.parent.parent
        if not resume_ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint para retomada não encontrado: {resume_ckpt_path}")

    # 1. Carrega e valida a configuração JSON
    config_source = (resume_exp_dir / "config.json") if (resume_exp_dir and (resume_exp_dir / "config.json").is_file()) else args.config
    config = load_config(config_source)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.device is not None:
        config.device = args.device

    device = setup_device(config.device)
    set_seed(config.seed)

    # 2. Carrega os DataLoaders diretamente na VRAM da GPU (ou CPU)
    train_loader, val_loader, test_loader, datasets = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=config.batch_size,
        device=device,
    )

    steps_per_epoch = len(train_loader)  # 750 para MNIST 48k com batch 64

    # Ajuste quando max_steps for fornecido (via CLI ou JSON) e --epochs não foi forçado via CLI
    max_steps = config.max_steps
    if max_steps is not None and args.epochs is None:
        config.epochs = math.ceil(max_steps / steps_per_epoch)

    # 3. Prepara diretório do experimento com nomenclatura estruturada por número de camadas
    if resume_exp_dir is not None:
        exp_dir = resume_exp_dir
    else:
        num_layers = len(config.hidden_layers)
        exp_prefix = (
            f"{config.experiment_name}_"
            if config.experiment_name and config.experiment_name not in ("vae_experiment", "vae_default")
            else ""
        )
        default_name = f"{exp_prefix}layers{num_layers}_{config.activation}_lat{config.latent_dim}_lr{config.learning_rate}_seed{config.seed}"

        # Runs destinadas a rounds (ex: experiments/round0) não usam timestamp por padrão
        is_round_dir = "round" in str(args.output_dir).lower()
        use_timestamp = args.with_timestamp or (not args.no_timestamp and not is_round_dir)

        if args.name:
            folder_name = args.name
        elif not use_timestamp:
            folder_name = default_name
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{timestamp}_{default_name}"

        exp_dir = Path(args.output_dir) / folder_name

    checkpoints_dir = exp_dir / "checkpoints"
    reconstructions_dir = exp_dir / "reconstructions"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reconstructions_dir.mkdir(parents=True, exist_ok=True)

    # Salva cópia exata do JSON de configuração usado
    config.save_json(exp_dir / "config.json")

    print("=" * 70)
    print("INICIANDO EXPERIMENTO VAE (ALTA PERFORMANCE VRAM)")
    print(f"Diretório:                {exp_dir}")
    print(f"Dispositivo:              {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Camadas Ocultas Encoder:  {config.hidden_layers}")
    print(f"Espaço Latente:           {config.latent_dim}")
    print(f"Ativação Interna:         {config.activation.upper()}")
    print(f"Taxa de Aprendizado (LR): {config.learning_rate}")
    print(f"Semente Aleatória:        {config.seed}")
    print(f"Total de Épocas:          {config.epochs}" + (f" (Meta: {max_steps} passos)" if max_steps else ""))
    print(f"Passos por Época:         {steps_per_epoch} (Lotes de {config.batch_size})")
    if device.type == "cuda":
        allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        print(f"Memória VRAM Dataset:     {allocated_mb:.1f} MB alocados (Treino + Val + Teste)")
    print("=" * 70)

    # Amostras fixas para visualização consistente da reconstrução (seeds fixas independentes)
    val_fixed_imgs, val_fixed_lbls = select_fixed_eval_samples(datasets["val"], num_samples=16, seed=42)
    test_fixed_imgs, test_fixed_lbls = select_fixed_eval_samples(datasets["test"], num_samples=16, seed=123)

    # 4. Inicializa o Modelo VAE e Otimizador Adam
    model = VAE(config).to(device)
    is_mirrored, mirror_details = model.get_mirrored_check()
    if not is_mirrored:
        raise RuntimeError("Erro crítico: O modelo não é estritamente espelhado!")

    adam_kwargs = {
        "lr": config.learning_rate,
        "betas": config.adam_params.betas,
        "eps": config.adam_params.eps,
        "weight_decay": config.adam_params.weight_decay,
    }
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        # Ativa kernel fundido do Adam no PyTorch e suporte a captura de CUDA Graph
        adam_kwargs["fused"] = True
        adam_kwargs["capturable"] = True

    optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)

    # 5. Arquivo de Métricas CSV e Estado de Treinamento
    metrics_csv_path = exp_dir / "metrics.csv"
    csv_headers = [
        "epoch",
        "step",
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
        "epoch_time_s",
        "status",
    ]

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_val_recon = float("inf")
    best_val_kl = float("inf")
    test_loss_at_best_val = float("inf")
    test_recon_at_best_val = float("inf")
    test_kl_at_best_val = float("inf")
    best_epoch = -1
    best_step = 0
    start_epoch = 1
    global_step = 0
    diverged = False

    if resume_ckpt_path is not None:
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", -1))
        best_step = int(ckpt.get("best_step", 0))
        best_val_recon = float(ckpt.get("best_val_recon", float("inf")))
        best_val_kl = float(ckpt.get("best_val_kl", float("inf")))
        test_loss_at_best_val = float(ckpt.get("test_loss_at_best_val", float("inf")))
        test_recon_at_best_val = float(ckpt.get("test_recon_at_best_val", float("inf")))
        test_kl_at_best_val = float(ckpt.get("test_kl_at_best_val", float("inf")))

        if metrics_csv_path.is_file():
            with open(metrics_csv_path, "r", newline="", encoding="utf-8") as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    if row.get("status") == "OK":
                        history.append({
                            "epoch": int(row["epoch"]),
                            "step": int(row["step"]),
                            "train_loss": float(row["train_loss"]),
                            "train_recon": float(row["train_recon"]),
                            "train_kl": float(row["train_kl"]),
                            "val_loss": float(row["val_loss"]),
                            "val_recon": float(row["val_recon"]),
                            "val_kl": float(row["val_kl"]),
                            "test_loss": float(row["test_loss"]),
                            "test_recon": float(row["test_recon"]),
                            "test_kl": float(row["test_kl"]),
                            "is_best_val": int(row["is_best_val"]),
                            "epoch_time_s": float(row.get("epoch_time_s", 0.0)),
                            "status": "OK",
                        })
        print(f"[Retomada] Continuando da época {start_epoch} (passo {global_step}, melhor val {best_val_loss:.2f} na época {best_epoch})")
        csv_mode = "a"
    else:
        csv_mode = "w"

    # 6. Preparação de CUDA Graph (elimina gargalo de despacho de 78 kernels CPU->GPU por passo)
    use_cuda_graph = (device.type == "cuda") and (not args.no_cuda_graph)
    train_graph: Optional[torch.cuda.CUDAGraph] = None
    static_x: Optional[torch.Tensor] = None
    static_sums: Optional[torch.Tensor] = None

    if use_cuda_graph:
        # Preserva estado exato dos pesos, otimizador e RNG antes do warmup de captura
        saved_rng_state = torch.cuda.get_rng_state(device)
        saved_params = [p.detach().clone() for p in model.parameters()]
        saved_opt_state = {
            p: {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in st.items()}
            for p, st in optimizer.state.items()
        }

        static_x = torch.zeros(config.batch_size, config.input_dim, device=device)
        static_sums = torch.zeros(3, device=device)
        model.train()

        # Warmup na stream auxiliar para inicializar buffers do autograd e estados do Adam
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False)
                r_w, m_w, lv_w = model(static_x)
                l_w, rec_w, kl_w = vae_loss_function(r_w, static_x, m_w, lv_w)
                l_w.backward()
                optimizer.step()
                static_sums += torch.stack([l_w.detach(), rec_w.detach(), kl_w.detach()]) * config.batch_size
                del r_w, m_w, lv_w, l_w, rec_w, kl_w
        torch.cuda.current_stream().wait_stream(s)

        # Captura o passo completo (zero_grad + forward + loss + backward + step + acumulação)
        train_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(train_graph):
            optimizer.zero_grad(set_to_none=False)
            r_g, m_g, lv_g = model(static_x)
            l_g, rec_g, kl_g = vae_loss_function(r_g, static_x, m_g, lv_g)
            l_g.backward()
            optimizer.step()
            static_sums += torch.stack([l_g.detach(), rec_g.detach(), kl_g.detach()]) * config.batch_size

        # Restaura exatamente os pesos, estados do otimizador e RNG prévios à captura
        with torch.no_grad():
            for p, sp in zip(model.parameters(), saved_params):
                p.copy_(sp)
            for p, st in optimizer.state.items():
                if p in saved_opt_state:
                    for k, v in saved_opt_state[p].items():
                        if torch.is_tensor(v):
                            st[k].copy_(v)
                        else:
                            st[k] = v
                else:
                    for v in st.values():
                        if torch.is_tensor(v):
                            v.zero_()
            static_sums.zero_()
        torch.cuda.set_rng_state(saved_rng_state, device)

    # 7. Loop de Treinamento (mantendo CSV aberto para evitar overhead de open/close a cada época)
    total_train_start = time.perf_counter()
    with open(metrics_csv_path, csv_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if csv_mode == "w":
            writer.writerow(csv_headers)
            csv_file.flush()

        for epoch in range(start_epoch, config.epochs + 1):
            epoch_t0 = time.perf_counter()
            model.train()
            train_sums = torch.zeros(3, device=device)
            train_samples = 0
            if static_sums is not None:
                static_sums.zero_()

            # Treino otimizado: CUDA Graph (ou eager) com acúmulo na GPU sem sincronização .item() intermediária
            for images, _ in train_loader:
                batch_sz = images.size(0)

                if train_graph is not None and static_x is not None and batch_sz == config.batch_size:
                    static_x.copy_(images)
                    train_graph.replay()
                else:
                    optimizer.zero_grad()
                    recon_images, mu, logvar = model(images)
                    loss, recon, kl = vae_loss_function(recon_images, images, mu, logvar)
                    loss.backward()
                    optimizer.step()
                    train_sums += torch.stack([loss.detach(), recon.detach(), kl.detach()]) * batch_sz

                train_samples += batch_sz
                global_step += 1

                if max_steps is not None and global_step >= max_steps:
                    break

            if static_sums is not None:
                train_sums += static_sums

            # Única sincronização GPU -> CPU ao final da época de treino
            epoch_train_loss, epoch_train_recon, epoch_train_kl = (train_sums / train_samples).tolist()

            # Guarda contra NaN / Divergência no treino (interrupção precoce)
            if not math.isfinite(epoch_train_loss):
                epoch_dt = time.perf_counter() - epoch_t0
                print(
                    f"\n[ALERTA DIVERGÊNCIA] Perda de treino não-finita detectada na época {epoch}: {epoch_train_loss}. "
                    f"Interrompendo run para economizar tempo da GPU."
                )
                writer.writerow([epoch, global_step, "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", 0, f"{epoch_dt:.4f}", "DIVERGED"])
                csv_file.flush()
                diverged = True
                break

            # Avaliação de uma vez só (ou em super-lotes) na VRAM: 1 forward val + 1 forward teste
            epoch_val_loss, epoch_val_recon, epoch_val_kl = evaluate(model, val_loader, args.eval_batch_size)
            epoch_test_loss, epoch_test_recon, epoch_test_kl = evaluate(model, test_loader, args.eval_batch_size)
            epoch_dt = time.perf_counter() - epoch_t0

            # Guarda contra NaN / Divergência na avaliação
            if not (math.isfinite(epoch_val_loss) and math.isfinite(epoch_test_loss)):
                print(
                    f"\n[ALERTA DIVERGÊNCIA] Perda de validação/teste não-finita na época {epoch} "
                    f"(Val={epoch_val_loss}, Teste={epoch_test_loss}). Interrompendo run."
                )
                writer.writerow([
                    epoch, global_step,
                    epoch_train_loss, epoch_train_recon, epoch_train_kl,
                    "NaN", "NaN", "NaN", "NaN", "NaN", "NaN",
                    0, f"{epoch_dt:.4f}", "DIVERGED"
                ])
                csv_file.flush()
                diverged = True
                break

            # Seleção estritamente pela perda de validação (nunca pelo teste)
            is_best = epoch_val_loss < best_val_loss

            # Registro no histórico
            epoch_record = {
                "epoch": epoch,
                "step": global_step,
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
                "epoch_time_s": round(epoch_dt, 4),
                "status": "OK",
            }
            history.append(epoch_record)

            # Grava linha no CSV aberto
            writer.writerow([epoch_record[h] for h in csv_headers])
            csv_file.flush()

            # Exibição concisa no console
            is_terminal_epoch = (epoch == config.epochs) or (max_steps is not None and global_step >= max_steps)
            should_print = (
                config.epochs <= 50 or
                is_best or
                epoch % 10 == 0 or
                epoch == 1 or
                is_terminal_epoch
            )
            if should_print:
                star = " ★ [MELHOR VAL]" if is_best else ""
                print(
                    f"Época [{epoch:04d}/{config.epochs:04d}] Passo [{global_step:06d}] ({epoch_dt:.2f}s) | "
                    f"Treino: {epoch_train_loss:.2f} (MSE={epoch_train_recon:.2f}, KL={epoch_train_kl:.2f}) | "
                    f"Val: {epoch_val_loss:.2f} (MSE={epoch_val_recon:.2f}, KL={epoch_val_kl:.2f}) | "
                    f"Teste: {epoch_test_loss:.2f}{star}"
                )

            # Tratamento de novo recorde na validação
            if is_best:
                best_val_loss = epoch_val_loss
                best_val_recon = epoch_val_recon
                best_val_kl = epoch_val_kl
                test_loss_at_best_val = epoch_test_loss
                test_recon_at_best_val = epoch_test_recon
                test_kl_at_best_val = epoch_test_kl
                best_epoch = epoch
                best_step = global_step

                # Checkpoint do melhor modelo (selecionado exclusivamente por val_loss)
                checkpoint_path = checkpoints_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "val_recon": epoch_val_recon,
                        "val_kl": epoch_val_kl,
                        "test_loss_at_best_val": test_loss_at_best_val,
                        "test_recon_at_best_val": test_recon_at_best_val,
                        "test_kl_at_best_val": test_kl_at_best_val,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config.to_dict(),
                    },
                    checkpoint_path,
                )

                # Só salva imagens intermediárias se explicitamente solicitado (--save_intermediate_recons)
                if args.save_intermediate_recons and (epoch == 1 or epoch % 20 == 0 or is_terminal_epoch):
                    model.eval()
                    with torch.no_grad():
                        val_recon_dev, _, _ = model(val_fixed_imgs)
                        save_reconstruction_grid(
                            original_images=val_fixed_imgs,
                            reconstructed_images=val_recon_dev,
                            labels=val_fixed_lbls,
                            save_path=reconstructions_dir / f"best_val_epoch_{epoch:03d}.png",
                            title=f"Reconstrução Validação - Época {epoch} (Val Loss: {epoch_val_loss:.2f})",
                        )

            # Salva checkpoint do último estado (last.pt) apenas a cada N épocas e no fim
            if epoch % args.ckpt_interval == 0 or is_terminal_epoch:
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": epoch_train_loss,
                        "val_loss": epoch_val_loss,
                        "test_loss": epoch_test_loss,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config.to_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_recon": best_val_recon,
                        "best_val_kl": best_val_kl,
                        "test_loss_at_best_val": test_loss_at_best_val,
                        "test_recon_at_best_val": test_recon_at_best_val,
                        "test_kl_at_best_val": test_kl_at_best_val,
                        "best_epoch": best_epoch,
                        "best_step": best_step,
                    },
                    checkpoints_dir / "last.pt",
                )

            # Gráfico das curvas de aprendizado: gera apenas a cada N épocas e na conclusão
            if epoch % args.plot_interval == 0 or is_terminal_epoch:
                plot_loss_curves(history, exp_dir / "loss_curve.png")

            if max_steps is not None and global_step >= max_steps:
                print(f"\nMeta de passos atingida: {global_step}/{max_steps} passos concluídos.")
                break

    total_train_time = time.perf_counter() - total_train_start

    # Garante imagem final de reconstrução se o melhor modelo existir
    if best_epoch > 0 and not diverged:
        model.eval()
        with torch.no_grad():
            best_ckpt = torch.load(checkpoints_dir / "best_model.pt", map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            val_recon_dev, _, _ = model(val_fixed_imgs)
            test_recon_dev, _, _ = model(test_fixed_imgs)
            save_reconstruction_grid(
                original_images=val_fixed_imgs,
                reconstructed_images=val_recon_dev,
                labels=val_fixed_lbls,
                save_path=reconstructions_dir / f"final_best_val_epoch_{best_epoch:03d}.png",
                title=f"Melhor Validação - Época {best_epoch} (Val Loss: {best_val_loss:.2f})",
            )
            save_reconstruction_grid(
                original_images=test_fixed_imgs,
                reconstructed_images=test_recon_dev,
                labels=test_fixed_lbls,
                save_path=reconstructions_dir / f"final_best_test_epoch_{best_epoch:03d}.png",
                title=f"Reconstrução Teste na Melhor Val - Época {best_epoch} (Test Loss: {test_loss_at_best_val:.2f})",
            )

    # Gráfico final de perdas
    if len(history) > 0:
        plot_loss_curves(history, exp_dir / "loss_curve.png")

    # Salva resumo estruturado (summary.json) com métricas na época de melhor validação
    avg_epoch_time = (
        sum(h["epoch_time_s"] for h in history) / len(history)
        if len(history) > 0
        else 0.0
    )
    summary_data = {
        "status": "DIVERGED" if diverged else "COMPLETED",
        "diverged": diverged,
        "epochs_completed": len(history),
        "global_step": global_step,
        "total_time_s": round(total_train_time, 3),
        "avg_epoch_time_s": round(avg_epoch_time, 4),
        "best_val": {
            "epoch": best_epoch,
            "step": best_step,
            "val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
            "val_recon": best_val_recon if math.isfinite(best_val_recon) else None,
            "val_kl": best_val_kl if math.isfinite(best_val_kl) else None,
            "test_loss_at_best_val": test_loss_at_best_val if math.isfinite(test_loss_at_best_val) else None,
            "test_recon_at_best_val": test_recon_at_best_val if math.isfinite(test_recon_at_best_val) else None,
            "test_kl_at_best_val": test_kl_at_best_val if math.isfinite(test_kl_at_best_val) else None,
        },
        "config": config.to_dict(),
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2)

    print("\n" + "=" * 70)
    if diverged:
        print("EXPERIMENTO FINALIZADO: DIVERGÊNCIA DETECTADA!")
        print(f"Resumo salvo em:           {exp_dir / 'summary.json'}")
    else:
        print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print(f"Melhor Época (por Val):    {best_epoch} (Passo {best_step})")
        print(
            f"Métricas Melhor Validação: Total = {best_val_loss:.2f} "
            f"(MSE = {best_val_recon:.2f}, KL = {best_val_kl:.2f})"
        )
        print(
            f"Teste na Melhor Validação: Total = {test_loss_at_best_val:.2f} "
            f"(MSE = {test_recon_at_best_val:.2f}, KL = {test_kl_at_best_val:.2f})"
        )
        print(f"Passos Totais Executados:  {global_step}")
        print(f"Tempo Médio por Época:     {avg_epoch_time:.3f}s (Total: {total_train_time:.2f}s)")
        print(f"Checkpoint Melhor Modelo:  {checkpoints_dir / 'best_model.pt'}")
        print(f"Checkpoint Último Estado:  {checkpoints_dir / 'last.pt'}")
        print(f"Resumo Estruturado:        {exp_dir / 'summary.json'}")
        print(f"Curvas de Perda Salvas:    {exp_dir / 'loss_curve.png'}")
        print(f"Histórico CSV:             {metrics_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
