"""
Análise do espaço latente 2D para modelos VAE treinados em Fashion-MNIST.

Para cada run de dimensão latente 2 em experiments/bestmodels (ou diretório/run
especificado), calcula exatamente:
1. Contagem por bin (densidade H[i, j]) em grade BxB para mu e para z amostrado.
2. Score kNN de decodificação dos centros dos bins em relação às 60k imagens de treino.
3. Máscara binária de buracos: bins com H_mu = 0 em que todos os vizinhos de ao
   menos um anel concêntrico de distância d em {1..max_distance} (padrão d=1..5)
   estiverem ativos (H_mu > 0).

Gera por run:
- Figura com 4 painéis: log1p(H) dos mu, log1p(H) dos z amostrados, score kNN e máscara de buracos.
- Arquivo .npz com H_mu, H_z, knn_scores, holes_mask, x_edges e y_edges.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Garante importação dos módulos existentes em vae/ e scripts/train.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import set_seed, setup_device
from vae.config import load_config
from vae.dataset import MNISTIDXDataset, get_dataloaders
from vae.model import VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análise do espaço latente 2D (densidade, score kNN e máscara de buracos)."
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments/bestmodels",
        help="Diretório contendo os runs de origem (padrão: experiments/bestmodels)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/latent_explore",
        help="Diretório onde os resultados e gráficos por run serão salvos (padrão: experiments/latent_explore)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Caminho opcional para rodar em um único diretório de experimento específico",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Diretório do dataset Fashion-MNIST (padrão: dataset)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        choices=["best_model.pt", "last.pt"],
        help="Qual checkpoint carregar de checkpoints/ (padrão: best_model.pt)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Número de bins em cada eixo da grade BxB (padrão: 100)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margem fracionária adicionada além dos percentis 0.5 e 99.5 dos mu (padrão: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Número de vizinhos mais próximos para o score kNN (padrão: 5)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Tamanho do chunk de centros decodificados para torch.cdist na GPU (padrão: 1000)",
    )
    parser.add_argument(
        "--max_ring_distance",
        "--max-ring-distance",
        dest="max_ring_distance",
        type=int,
        default=5,
        help="Distância máxima de anel d=1..max_ring_distance para critério de buracos (padrão: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente fixa para a amostragem de z = mu + sigma * eps (padrão: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de execução ('cuda', 'cpu' ou 'auto')",
    )
    return parser.parse_args()


def discover_latent2_runs(experiments_dir: Path, single_run_dir: str | None = None) -> List[Path]:
    """
    Descobre runs com config.json (na raiz ou em json/) e checkpoints/ cujo latent_dim == 2.
    """
    candidates: List[Path] = []
    if single_run_dir is not None:
        run_path = Path(single_run_dir)
        if not run_path.is_dir():
            raise FileNotFoundError(f"Diretório de run não encontrado: {run_path}")
        candidates.append(run_path)
    else:
        if not experiments_dir.is_dir():
            raise FileNotFoundError(f"Diretório de experimentos não encontrado: {experiments_dir}")
        if (experiments_dir / "config.json").is_file() or (experiments_dir / "json" / "config.json").is_file():
            candidates.append(experiments_dir)
        else:
            for p in sorted(experiments_dir.iterdir()):
                if p.is_dir() and ((p / "config.json").is_file() or (p / "json" / "config.json").is_file()):
                    candidates.append(p)

    lat2_runs: List[Path] = []
    for run_path in candidates:
        cfg_path = run_path / "config.json" if (run_path / "config.json").is_file() else run_path / "json" / "config.json"
        cfg = load_config(cfg_path)
        if cfg.latent_dim == 2:
            lat2_runs.append(run_path)

    return lat2_runs


def load_full_train_60k(dataset_dir: Path, device: torch.device) -> torch.Tensor:
    """
    Carrega as 60.000 imagens originais de treino em [0, 1] na GPU/CPU.
    Se dataset/raw/train-images-idx3-ubyte existir, carrega diretamente as 60k na ordem original;
    caso contrário, concatena os splits train (48k) + val (12k) de get_dataloaders().
    """
    raw_train_img = dataset_dir / "raw" / "train-images-idx3-ubyte"
    raw_train_lbl = dataset_dir / "raw" / "train-labels-idx1-ubyte"

    if raw_train_img.is_file() and raw_train_lbl.is_file():
        ds = MNISTIDXDataset(raw_train_img, raw_train_lbl).to(device)
        return ds.images

    _, _, _, datasets = get_dataloaders(dataset_dir=dataset_dir, device=device)
    full_train = torch.cat([datasets["train"].images, datasets["val"].images], dim=0)
    return full_train


@torch.no_grad()
def encode_dataset(
    model: VAE,
    images: torch.Tensor,
    seed: int,
    batch_size: int = 16384,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Passa as 60k imagens pelo encoder em eval() sem gradientes, coleta mu e logvar,
    e amostra z = mu + sigma * eps com semente fixa.
    """
    model.eval()
    n = images.size(0)
    mus: List[torch.Tensor] = []
    logvars: List[torch.Tensor] = []

    for start in range(0, n, batch_size):
        batch = images[start : start + batch_size]
        mu_b, logvar_b = model.encode(batch)
        mus.append(mu_b)
        logvars.append(logvar_b)

    mu = torch.cat(mus, dim=0)
    logvar = torch.cat(logvars, dim=0)

    gen = torch.Generator(device=mu.device)
    gen.manual_seed(seed)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(mu.shape, generator=gen, device=mu.device, dtype=mu.dtype)
    z_sampled = mu + std * eps

    return mu.cpu().numpy(), z_sampled.cpu().numpy()


def compute_bounds_and_histograms(
    mu: np.ndarray,
    z_sampled: np.ndarray,
    bins: int,
    margin_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Define limites automáticos a partir dos percentis 0.5 e 99.5 dos mu com margem
    e calcula H[i, j] (grade bins x bins) para mu e para z_sampled usando os mesmos bins.
    """
    p_low = np.percentile(mu, 0.5, axis=0)
    p_high = np.percentile(mu, 99.5, axis=0)
    span = p_high - p_low
    margin = margin_frac * span

    x_min, x_max = float(p_low[0] - margin[0]), float(p_high[0] + margin[0])
    y_min, y_max = float(p_low[1] - margin[1]), float(p_high[1] + margin[1])

    x_edges = np.linspace(x_min, x_max, bins + 1, dtype=np.float64)
    y_edges = np.linspace(y_min, y_max, bins + 1, dtype=np.float64)

    h_mu, _, _ = np.histogram2d(mu[:, 0], mu[:, 1], bins=[x_edges, y_edges])
    h_z, _, _ = np.histogram2d(z_sampled[:, 0], z_sampled[:, 1], bins=[x_edges, y_edges])

    return h_mu.astype(np.int64), h_z.astype(np.int64), x_edges, y_edges


@torch.no_grad()
def compute_knn_scores(
    model: VAE,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    train_images: torch.Tensor,
    k: int = 5,
    chunk_size: int = 1000,
) -> np.ndarray:
    """
    Decodifica o centro de cada bin (i, j) apenas pelo decoder e calcula a média das
    distâncias L2 aos k vizinhos mais próximos entre as 60k imagens de treino via torch.cdist.
    """
    model.eval()
    device = train_images.device
    bins_x = len(x_edges) - 1
    bins_y = len(y_edges) - 1

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Grade indexada como (i, j) -> (x_centers[i], y_centers[j]) para alinhar com H[i, j]
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    centers_np = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    centers_t = torch.from_numpy(centers_np).to(device)

    n_centers = centers_t.size(0)
    knn_scores_flat = torch.empty(n_centers, dtype=torch.float32, device=device)

    for start in range(0, n_centers, chunk_size):
        z_chunk = centers_t[start : start + chunk_size]
        decoded_chunk = model.decode(z_chunk)  # (chunk, 784)
        dists = torch.cdist(decoded_chunk, train_images, p=2.0)  # (chunk, 60000)
        topk_dists, _ = torch.topk(dists, k=k, dim=1, largest=False)
        knn_scores_flat[start : start + chunk_size] = topk_dists.mean(dim=1)

    return knn_scores_flat.cpu().numpy().reshape(bins_x, bins_y)


def compute_holes_mask(h_mu: np.ndarray, max_distance: int = 5) -> np.ndarray:
    """
    Calcula a máscara binária de buracos por anéis concêntricos (distância Chebyshev d = 1..max_distance).
    Um bin inativo (H_mu == 0) é classificado como buraco se, para alguma distância
    d em {1, ..., max_distance}, TODOS os 8*d vizinhos do anel à distância d estiverem ativos (H_mu > 0).
    """
    if max_distance < 1:
        raise ValueError(f"max_distance deve ser >= 1, recebido: {max_distance}")

    active = (h_mu > 0).astype(np.float32)
    inactive = (h_mu == 0)
    active_t = torch.from_numpy(active).unsqueeze(0).unsqueeze(0)

    combined = np.zeros_like(inactive, dtype=bool)

    for d in range(1, max_distance + 1):
        k_size = 2 * d + 1
        kernel = np.zeros((k_size, k_size), dtype=np.float32)
        # Borda do quadrado (2d+1)x(2d+1): max(|dx|, |dy|) == d
        kernel[0, :] = 1.0
        kernel[-1, :] = 1.0
        kernel[:, 0] = 1.0
        kernel[:, -1] = 1.0

        num_ring_neighbors = int(kernel.sum())  # 8 * d
        weight = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)

        ring_active_count = F.conv2d(active_t, weight, padding=d).squeeze(0).squeeze(0).numpy()
        ring_all_active = np.isclose(ring_active_count, num_ring_neighbors)

        holes_d = inactive & ring_all_active
        combined |= holes_d

    return combined.astype(np.uint8)


def save_four_panel_figure(
    h_mu: np.ndarray,
    h_z: np.ndarray,
    knn_scores: np.ndarray,
    holes_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    max_distance: int,
    k: int,
    run_name: str,
    save_path: Path,
) -> None:
    """
    Salva figura com 4 painéis usando os mesmos limites e bins:
    1. log1p(H) dos mu
    2. log1p(H) dos z amostrados
    3. Mapa do score kNN
    4. Máscara de buracos
    """
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    log_h_mu = np.log1p(h_mu)
    log_h_z = np.log1p(h_z)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    fig.suptitle(f"Análise do Espaço Latente 2D — {run_name}", fontsize=13, fontweight="bold", y=1.02)

    # Painel 1: log1p(H) dos mu
    im0 = axes[0].imshow(
        log_h_mu.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[0].set_title(r"1. Densidade $\log(1 + H(\mu))$", fontweight="bold")
    axes[0].set_xlabel(r"$z_1$")
    axes[0].set_ylabel(r"$z_2$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Painel 2: log1p(H) dos z amostrados
    im1 = axes[1].imshow(
        log_h_z.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1].set_title(r"2. Densidade $\log(1 + H(z_{\mathrm{amostrado}}))$", fontweight="bold")
    axes[1].set_xlabel(r"$z_1$")
    axes[1].set_ylabel(r"$z_2$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Painel 3: Mapa do score kNN
    im2 = axes[2].imshow(
        knn_scores.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma_r",
        interpolation="nearest",
    )
    axes[2].set_title(f"3. Score kNN (k={k}) da Decodificação", fontweight="bold")
    axes[2].set_xlabel(r"$z_1$")
    axes[2].set_ylabel(r"$z_2$")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Painel 4: Máscara de buracos
    im3 = axes[3].imshow(
        holes_mask.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="binary",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[3].set_title(f"4. Máscara de Buracos (anéis d=1..{max_distance})", fontweight="bold")
    axes[3].set_xlabel(r"$z_1$")
    axes[3].set_ylabel(r"$z_2$")
    cbar3 = fig.colorbar(im3, ax=axes[3], ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar3.ax.set_yticklabels(["0 (Não)", "1 (Buraco)"])

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_run(
    run_dir: Path,
    train_images: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    """Executa a análise completa do espaço latente para um run de dimensão 2."""
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        config_path = run_dir / "json" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json não encontrado em {run_dir} ou {run_dir / 'json'}")

    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    config = load_config(config_path)
    set_seed(args.seed)

    model = VAE(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"\n[Run] {run_dir.name}")
    print(f"  Checkpoint: {ckpt_path.name} (época {ckpt.get('epoch', '?')})")
    print(f"  Imagens de treino: {train_images.shape[0]} amostras em {device}")

    # 1. Contagem por bin (densidade)
    mu, z_sampled = encode_dataset(model, train_images, seed=args.seed)
    h_mu, h_z, x_edges, y_edges = compute_bounds_and_histograms(
        mu=mu,
        z_sampled=z_sampled,
        bins=args.bins,
        margin_frac=args.margin,
    )
    print(
        f"  Limites z1: [{x_edges[0]:.3f}, {x_edges[-1]:.3f}] | "
        f"z2: [{y_edges[0]:.3f}, {y_edges[-1]:.3f}] | Grade: {args.bins}x{args.bins}"
    )
    print(
        f"  Contagem total dentro da grade -> H(mu): {int(h_mu.sum())}/60000 | "
        f"H(z_sampled): {int(h_z.sum())}/60000"
    )

    # 2. Score kNN de decodificação
    knn_scores = compute_knn_scores(
        model=model,
        x_edges=x_edges,
        y_edges=y_edges,
        train_images=train_images,
        k=args.k,
        chunk_size=args.chunk_size,
    )
    print(
        f"  Score kNN (k={args.k}) -> min: {knn_scores.min():.3f}, "
        f"média: {knn_scores.mean():.3f}, max: {knn_scores.max():.3f}"
    )

    # 3. Máscara de buracos
    holes_mask = compute_holes_mask(h_mu=h_mu, max_distance=args.max_ring_distance)
    n_occupied = int((h_mu > 0).sum())
    n_holes = int(holes_mask.sum())
    print(
        f"  Bins ocupados (H_mu > 0): {n_occupied} | "
        f"Buracos em anéis (d=1..{args.max_ring_distance}): {n_holes}"
    )

    # Salva .npz e figura no diretório de saída (figura em analysis/, .npz em npz/)
    out_run_dir = Path(args.output_dir) / run_dir.name
    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_npz_dir = out_run_dir / "npz"
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    out_analysis_dir = out_run_dir / "analysis"
    out_analysis_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_npz_dir / "latent_analysis.npz"
    fig_path = out_analysis_dir / "latent_analysis.png"

    np.savez_compressed(
        npz_path,
        H_mu=h_mu,
        H_z=h_z,
        knn_scores=knn_scores,
        holes_mask=holes_mask,
        x_edges=x_edges,
        y_edges=y_edges,
    )

    save_four_panel_figure(
        h_mu=h_mu,
        h_z=h_z,
        knn_scores=knn_scores,
        holes_mask=holes_mask,
        x_edges=x_edges,
        y_edges=y_edges,
        max_distance=args.max_ring_distance,
        k=args.k,
        run_name=run_dir.name,
        save_path=fig_path,
    )

    print(f"  Salvo: {fig_path}")
    print(f"  Salvo: {npz_path}")
    return fig_path, npz_path


def main() -> None:
    args = parse_args()
    device = setup_device(args.device)

    runs = discover_latent2_runs(Path(args.experiments_dir), args.run_dir)
    if not runs:
        print("Nenhum run com latent_dim == 2 encontrado.")
        return

    print(f"Runs de latente 2 encontrados ({len(runs)}): {[r.name for r in runs]}")
    train_images = load_full_train_60k(Path(args.dataset_dir), device=device)

    for run_dir in runs:
        analyze_run(run_dir, train_images, device, args)


if __name__ == "__main__":
    main()
