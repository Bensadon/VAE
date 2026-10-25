"""
Mapa de rótulos do Fashion-MNIST no plano latente 2D pelos lados do encoder e do decoder.

Reaproveita de analyze_latent.py e sample_holes.py o carregamento do run, do conjunto de
treino completo (60k imagens + rótulos), a codificação dos mu, as arestas dos bins
(x_edges, y_edges) e a máscara de buracos dentro do envelope (holes_mask).

Lado do encoder:
1. Para cada classe c (0 a 9), calcula o histograma 2D dos mu só daquela classe,
   com os mesmos bins (x_edges, y_edges), gerando Hc[c].
2. Classe dominante por bin = argmax de Hc sobre as classes; bins sem nenhuma imagem
   ficam em branco (NaN).
3. Pureza por bin = contagem da classe dominante / contagem total. Mascara (NaN) os
   bins com contagem total abaixo de um mínimo (--min-count, padrão 5).
4. Figura com 10 painéis, um por classe, mostrando log1p(Hc[c]), com o nome da classe no título.

Lado do decoder:
5. Passa o centro de cada bin só pelo decoder e calcula os 5 vizinhos mais próximos no treino
   (torch.cdist em chunks). O rótulo do bin é o voto majoritário entre os 5 vizinhos
   (em empate, o do vizinho mais próximo).

Figuras e saídas:
- Figura 10 painéis: log1p(Hc[c]) por classe.
- Figura A: classe dominante (encoder), com colormap categórico de 10 cores e legenda.
- Figura B: pureza (encoder), mascarando contagem < mínimo.
- Figura C: rótulo do decoder, mesma colormap e legenda da Figura A.
- Figura D: bins onde o rótulo do encoder e o do decoder discordam (só entre bins com
  contagem >= mínimo), com a fração de discordância impressa.
- Imprime a fração de bins vazios dentro do envelope atribuídos a cada classe pelo decoder.
- Salva tudo em .npz (Hc, dominante, pureza, rotulo_decoder).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_latent import compute_bounds_and_histograms, compute_holes_mask, encode_dataset
from sample_holes import resolve_run_paths
from train import set_seed, setup_device
from vae.config import load_config
from vae.dataset import MNISTIDXDataset, get_dataloaders
from vae.model import VAE

CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
NUM_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mapa de rótulos do Fashion-MNIST no espaço latente 2D (encoder vs decoder)."
    )
    parser.add_argument(
        "run_arg",
        nargs="?",
        default=None,
        help="Caminho ou nome do run a ser processado (opcional se --run / --run_dir for usado)",
    )
    parser.add_argument(
        "--run",
        "--run_dir",
        "--run-dir",
        dest="run_opt",
        type=str,
        default=None,
        help="Caminho ou nome do run (ex: rede4_layers4_relu_lat2_lr0.0001_seed24)",
    )
    parser.add_argument(
        "--experiments_dir",
        "--experiments-dir",
        dest="experiments_dir",
        type=str,
        default="experiments/bestmodels",
        help="Diretório base dos experimentos (padrão: experiments/bestmodels)",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="experiments/latent_explore",
        help="Diretório base onde está latent_analysis.npz e onde as saídas serão salvas (padrão: experiments/latent_explore)",
    )
    parser.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        dest="dataset_dir",
        type=str,
        default="dataset",
        help="Diretório do dataset Fashion-MNIST (padrão: dataset)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        choices=["best_model.pt", "last.pt"],
        help="Checkpoint do modelo a carregar (padrão: best_model.pt)",
    )
    parser.add_argument(
        "--min_count",
        "--min-count",
        "--min_contagem",
        "--min-contagem",
        dest="min_count",
        type=int,
        default=5,
        help="Contagem total mínima por bin para pureza e discordância (padrão: 5)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Número de vizinhos mais próximos no treino para votação do decoder (padrão: 5)",
    )
    parser.add_argument(
        "--chunk_size",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=1000,
        help="Tamanho do chunk para torch.cdist na GPU/CPU (padrão: 1000)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Número de bins caso latent_analysis.npz não exista (padrão: 100)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margem fracionária dos bins caso latent_analysis.npz não exista (padrão: 0.05)",
    )
    parser.add_argument(
        "--max_ring_distance",
        "--max-ring-distance",
        dest="max_ring_distance",
        type=int,
        default=5,
        help="Distância máxima de anel d=1..max_ring_distance para máscara de buracos caso não exista no .npz (padrão: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente fixa (padrão: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de execução ('cuda', 'cpu' ou 'auto')",
    )
    return parser.parse_args()


def load_full_train_60k_with_labels(
    dataset_dir: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Carrega as 60.000 imagens e rótulos originais de treino na mesma ordem de analyze_latent.py.
    """
    raw_train_img = dataset_dir / "raw" / "train-images-idx3-ubyte"
    raw_train_lbl = dataset_dir / "raw" / "train-labels-idx1-ubyte"

    if raw_train_img.is_file() and raw_train_lbl.is_file():
        ds = MNISTIDXDataset(raw_train_img, raw_train_lbl).to(device)
        return ds.images, ds.labels

    _, _, _, datasets = get_dataloaders(dataset_dir=dataset_dir, device=device)
    full_images = torch.cat([datasets["train"].images, datasets["val"].images], dim=0)
    full_labels = torch.cat([datasets["train"].labels, datasets["val"].labels], dim=0)
    return full_images, full_labels


def compute_encoder_label_maps(
    mu: np.ndarray,
    labels: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_count: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula para o lado do encoder:
    1. Hc[c]: histograma 2D dos mu apenas da classe c, shape (10, Bx, By).
    2. dominante: argmax de Hc sobre as classes; bins com contagem total == 0 ficam com NaN.
    3. pureza: contagem da classe dominante / contagem total; bins com contagem total < min_count ficam com NaN.
    4. total_counts: soma de Hc sobre as classes, shape (Bx, By).
    """
    bins_x = len(x_edges) - 1
    bins_y = len(y_edges) - 1
    hc = np.zeros((NUM_CLASSES, bins_x, bins_y), dtype=np.int64)

    for c in range(NUM_CLASSES):
        mask_c = labels == c
        mu_c = mu[mask_c]
        h_c, _, _ = np.histogram2d(mu_c[:, 0], mu_c[:, 1], bins=[x_edges, y_edges])
        hc[c] = h_c.astype(np.int64)

    total_counts = hc.sum(axis=0)  # (Bx, By)
    argmax_c = np.argmax(hc, axis=0)  # (Bx, By)
    max_counts = np.max(hc, axis=0)  # (Bx, By)

    dominante = np.where(total_counts > 0, argmax_c.astype(np.float64), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        pureza = np.where(
            total_counts >= min_count,
            max_counts.astype(np.float64) / total_counts.astype(np.float64),
            np.nan,
        )

    return hc, dominante, pureza, total_counts


def majority_vote_with_nearest_tiebreak(neighbor_labels: np.ndarray) -> np.ndarray:
    """
    Recebe neighbor_labels de shape (N, k) ordenado do vizinho mais próximo (col 0)
    ao k-ésimo mais próximo (col k-1).
    Para cada linha, escolhe o rótulo com maior frequência entre os k vizinhos.
    Em caso de empate na contagem máxima, escolhe o rótulo do vizinho mais próximo
    dentre os empatados (isto é, aquele cuja primeira aparição na linha ocorre no menor índice).
    """
    n, k = neighbor_labels.shape
    out = np.empty(n, dtype=np.int64)

    for idx in range(n):
        row = neighbor_labels[idx]
        best_label = int(row[0])
        best_count = 0
        best_first_pos = k

        for pos in range(k):
            lbl = int(row[pos])
            # Conta quantas vezes lbl aparece nos k vizinhos
            cnt = int(np.sum(row == lbl))
            first_pos = int(np.argmax(row == lbl))
            if cnt > best_count or (cnt == best_count and first_pos < best_first_pos):
                best_count = cnt
                best_first_pos = first_pos
                best_label = lbl

        out[idx] = best_label

    return out


@torch.no_grad()
def compute_decoder_label_map(
    model: VAE,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    k: int = 5,
    chunk_size: int = 1000,
) -> np.ndarray:
    """
    Passa o centro de cada bin (i, j) apenas pelo decoder e encontra os k vizinhos
    mais próximos no conjunto de treino via torch.cdist em chunks.
    Retorna a matriz rotulo_decoder de shape (Bx, By) com int64 em {0..9}.
    """
    model.eval()
    device = train_images.device
    bins_x = len(x_edges) - 1
    bins_y = len(y_edges) - 1

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    centers_np = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    centers_t = torch.from_numpy(centers_np).to(device)

    n_centers = centers_t.size(0)
    topk_labels_all = np.empty((n_centers, k), dtype=np.int64)

    for start in range(0, n_centers, chunk_size):
        end = min(start + chunk_size, n_centers)
        z_chunk = centers_t[start:end]
        decoded_chunk = model.decode(z_chunk)  # (chunk, 784)
        dists = torch.cdist(decoded_chunk, train_images, p=2.0)  # (chunk, 60000)
        _, topk_idx = torch.topk(dists, k=k, dim=1, largest=False, sorted=True)
        chunk_labels = train_labels[topk_idx].cpu().numpy()  # (chunk, k)
        topk_labels_all[start:end] = chunk_labels

    voted_flat = majority_vote_with_nearest_tiebreak(topk_labels_all)
    return voted_flat.reshape(bins_x, bins_y)


def get_categorical_cmap_and_norm() -> Tuple[ListedColormap, BoundaryNorm, list]:
    """
    Retorna colormap categórico de 10 cores distintas (tab10), BoundaryNorm para 0..9
    e handles de legenda com os nomes das 10 classes.
    Valores NaN são renderizados em branco.
    """
    base_cmap = plt.get_cmap("tab10")
    colors = [base_cmap(i) for i in range(NUM_CLASSES)]
    cmap = ListedColormap(colors, name="fashion_tab10")
    cmap.set_bad(color="white")
    bounds = np.arange(-0.5, NUM_CLASSES + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    legend_handles = [
        Patch(facecolor=colors[c], edgecolor="black", linewidth=0.5, label=f"{c}: {CLASS_NAMES[c]}")
        for c in range(NUM_CLASSES)
    ]
    return cmap, norm, legend_handles


def save_10panel_class_histograms(
    hc: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    run_name: str,
    save_path: Path,
) -> None:
    """
    Item 4: Figura com 10 painéis (2x5), um por classe, mostrando log1p(Hc[c]),
    com o nome da classe no título e mesma orientação das demais figuras.
    """
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    log_hc = np.log1p(hc)
    vmax = float(log_hc.max()) if log_hc.size > 0 else 1.0

    fig, axes = plt.subplots(2, 5, figsize=(20, 8.2), sharex=True, sharey=True)
    fig.suptitle(
        f"Densidade por Classe log(1 + Hc[c]) — {run_name}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    axes_flat = axes.ravel()
    for c in range(NUM_CLASSES):
        ax = axes_flat[c]
        im = ax.imshow(
            log_hc[c].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{c}: {CLASS_NAMES[c]}", fontweight="bold", fontsize=11)
        if c >= 5:
            ax.set_xlabel(r"$z_1$")
        if c % 5 == 0:
            ax.set_ylabel(r"$z_2$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fig_a_dominant_encoder(
    dominante: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    run_name: str,
    save_path: Path,
) -> None:
    """
    Figura A: classe dominante (encoder), com colormap categórico de 10 cores e
    legenda com os nomes das classes. Bins sem nenhuma imagem ficam em branco (NaN).
    """
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    cmap, norm, legend_handles = get_categorical_cmap_and_norm()

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.set_facecolor("white")
    ax.imshow(
        dominante.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(
        f"Figura A — Classe Dominante do Encoder (bins vazios em branco)\n{run_name}",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.legend(
        handles=legend_handles,
        title="Classes",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fig_b_purity_encoder(
    pureza: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_count: int,
    run_name: str,
    save_path: Path,
) -> None:
    """
    Figura B: pureza (encoder) = contagem da classe dominante / contagem total,
    mascarando bins com contagem total < min_count.
    """
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.set_facecolor("white")
    im = ax.imshow(
        pureza.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pureza (contagem dominante / total)")

    ax.set_title(
        f"Figura B — Pureza por Bin no Encoder (contagem >= {min_count})\n{run_name}",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fig_c_decoder_labels(
    rotulo_decoder: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    k: int,
    run_name: str,
    save_path: Path,
) -> None:
    """
    Figura C: rótulo do decoder (voto majoritário dos k vizinhos mais próximos),
    mesma colormap e legenda da Figura A.
    """
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    cmap, norm, legend_handles = get_categorical_cmap_and_norm()

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.imshow(
        rotulo_decoder.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(
        f"Figura C — Rótulo do Decoder (voto kNN, k={k})\n{run_name}",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.legend(
        handles=legend_handles,
        title="Classes",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fig_d_disagreement(
    disagreement_map: np.ndarray,
    disagreement_frac: float,
    n_disagree: int,
    n_valid: int,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_count: int,
    run_name: str,
    save_path: Path,
) -> None:
    """
    Figura D: bins onde o rótulo do encoder e o do decoder discordam (só entre bins
    com contagem >= min_count; demais bins em branco/NaN).
    """
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    cmap = ListedColormap(["#d9ead3", "#cc0000"], name="agree_disagree")
    cmap.set_bad(color="white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    legend_handles = [
        Patch(facecolor="#d9ead3", edgecolor="black", linewidth=0.5, label="0: Concordam (Encoder == Decoder)"),
        Patch(facecolor="#cc0000", edgecolor="black", linewidth=0.5, label="1: Discordam (Encoder != Decoder)"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.5, label=f"Contagem < {min_count} (mascarado)"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.set_facecolor("white")
    ax.imshow(
        disagreement_map.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(
        f"Figura D — Discordância Encoder vs Decoder (contagem >= {min_count})\n"
        f"Discordância: {n_disagree}/{n_valid} bins ({disagreement_frac:.2%}) — {run_name}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = setup_device(args.device)

    run_name, npz_path, model_run_dir, out_run_dir = resolve_run_paths(args)

    # Carrega modelo e conjunto de treino completo (60k imagens + rótulos)
    cfg_path = model_run_dir / "config.json"
    if not cfg_path.is_file():
        cfg_path = model_run_dir / "json" / "config.json"
    config = load_config(cfg_path)
    ckpt_path = model_run_dir / "checkpoints" / args.checkpoint
    model = VAE(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_images, train_labels_t = load_full_train_60k_with_labels(
        Path(args.dataset_dir), device=device
    )
    train_labels_np = train_labels_t.cpu().numpy()

    # Codifica os mu das 60k imagens de treino
    mu, z_sampled = encode_dataset(model, train_images, seed=args.seed)

    # Reaproveita x_edges, y_edges, H_mu e holes_mask do latent_analysis.npz
    if npz_path.is_file():
        with np.load(npz_path) as data:
            x_edges = data["x_edges"]
            y_edges = data["y_edges"]
            h_mu_npz = data["H_mu"]
            holes_mask = data["holes_mask"]
    else:
        h_mu_npz, _, x_edges, y_edges = compute_bounds_and_histograms(
            mu=mu,
            z_sampled=z_sampled,
            bins=args.bins,
            margin_frac=args.margin,
        )
        holes_mask = compute_holes_mask(h_mu=h_mu_npz, max_distance=args.max_ring_distance)

    print(f"[Run] {run_name}")
    print(f"  Reaproveitando bins e envelope de: {npz_path}")

    # Lado do encoder (itens 1, 2 e 3)
    hc, dominante, pureza, total_counts = compute_encoder_label_maps(
        mu=mu,
        labels=train_labels_np,
        x_edges=x_edges,
        y_edges=y_edges,
        min_count=args.min_count,
    )

    # Verifica consistência com H_mu de analyze_latent.py
    if not np.array_equal(total_counts, h_mu_npz):
        print("  [Aviso] Soma de Hc difere de H_mu salvo em latent_analysis.npz.")

    # Lado do decoder (item 5)
    rotulo_decoder = compute_decoder_label_map(
        model=model,
        x_edges=x_edges,
        y_edges=y_edges,
        train_images=train_images,
        train_labels=train_labels_t,
        k=args.k,
        chunk_size=args.chunk_size,
    )

    # Discordância entre encoder e decoder apenas nos bins com contagem >= min_count
    valid_mask = total_counts >= args.min_count
    disagree_bool = valid_mask & (dominante != rotulo_decoder)
    disagreement_map = np.where(valid_mask, disagree_bool.astype(np.float64), np.nan)

    n_valid = int(valid_mask.sum())
    n_disagree = int(disagree_bool.sum())
    disagreement_frac = float(n_disagree / n_valid) if n_valid > 0 else 0.0

    print(
        f"\n[Discordância Encoder vs Decoder (bins com contagem >= {args.min_count})]"
    )
    print(
        f"  Bins avaliados: {n_valid} | Bins discordantes: {n_disagree} | "
        f"Fração de discordância: {disagreement_frac:.4f} ({disagreement_frac:.2%})"
    )

    # Fração de bins vazios dentro do envelope (holes_mask > 0) atribuídos a cada classe pelo decoder
    holes_bool = holes_mask > 0
    n_holes = int(holes_bool.sum())
    decoder_labels_in_holes = rotulo_decoder[holes_bool]

    print(f"\n[Bins vazios dentro do envelope (total = {n_holes}) atribuídos pelo decoder]")
    hole_class_fractions = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        cnt_c = int(np.sum(decoder_labels_in_holes == c))
        frac_c = float(cnt_c / n_holes) if n_holes > 0 else 0.0
        hole_class_fractions[c] = frac_c
        print(f"  Classe {c} ({CLASS_NAMES[c]:<10s}): {cnt_c:4d}/{n_holes} = {frac_c:.4f} ({frac_c:.2%})")

    # Salva tudo em .npz dentro de npz/
    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_npz_dir = out_run_dir / "npz"
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    out_npz_path = out_npz_dir / "label_map.npz"
    np.savez_compressed(
        out_npz_path,
        Hc=hc,
        dominante=dominante,
        pureza=pureza,
        rotulo_decoder=rotulo_decoder,
        discordancia=disagreement_map,
        fator_discordancia=np.float64(disagreement_frac),
        fracao_buracos_por_classe=hole_class_fractions,
        x_edges=x_edges,
        y_edges=y_edges,
        min_count=np.int64(args.min_count),
        class_names=np.array(CLASS_NAMES),
    )

    # Salva figuras em analysis/
    out_analysis_dir = out_run_dir / "analysis"
    out_analysis_dir.mkdir(parents=True, exist_ok=True)
    fig_10panels_path = out_analysis_dir / "label_map_10classes.png"
    fig_a_path = out_analysis_dir / "label_map_figA_dominante_encoder.png"
    fig_b_path = out_analysis_dir / "label_map_figB_pureza_encoder.png"
    fig_c_path = out_analysis_dir / "label_map_figC_rotulo_decoder.png"
    fig_d_path = out_analysis_dir / "label_map_figD_discordancia.png"

    save_10panel_class_histograms(
        hc=hc,
        x_edges=x_edges,
        y_edges=y_edges,
        run_name=run_name,
        save_path=fig_10panels_path,
    )
    save_fig_a_dominant_encoder(
        dominante=dominante,
        x_edges=x_edges,
        y_edges=y_edges,
        run_name=run_name,
        save_path=fig_a_path,
    )
    save_fig_b_purity_encoder(
        pureza=pureza,
        x_edges=x_edges,
        y_edges=y_edges,
        min_count=args.min_count,
        run_name=run_name,
        save_path=fig_b_path,
    )
    save_fig_c_decoder_labels(
        rotulo_decoder=rotulo_decoder,
        x_edges=x_edges,
        y_edges=y_edges,
        k=args.k,
        run_name=run_name,
        save_path=fig_c_path,
    )
    save_fig_d_disagreement(
        disagreement_map=disagreement_map,
        disagreement_frac=disagreement_frac,
        n_disagree=n_disagree,
        n_valid=n_valid,
        x_edges=x_edges,
        y_edges=y_edges,
        min_count=args.min_count,
        run_name=run_name,
        save_path=fig_d_path,
    )

    print(f"\n[Arquivos salvos em {out_run_dir}]")
    print(f"  .npz:       {out_npz_path}")
    print(f"  10 painéis: {fig_10panels_path}")
    print(f"  Figura A:   {fig_a_path}")
    print(f"  Figura B:   {fig_b_path}")
    print(f"  Figura C:   {fig_c_path}")
    print(f"  Figura D:   {fig_d_path}")


if __name__ == "__main__":
    main()
