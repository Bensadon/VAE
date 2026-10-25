"""
Amostra pontos dentro dos buracos do espaço latente 2D e os decodifica.

Reaproveita os resultados gerados por analyze_latent.py (latent_analysis.npz):
1. Lê a máscara de buracos, H_mu e as arestas dos bins (x_edges, y_edges) de um run.
2. Com --max-contagem (padrão 0), usa bins com H_mu <= max_contagem dentro do envelope
   definido por janela WxW (padrão 5x5); para max_contagem == 0, equivale à holes_mask.
3. Sorteia N bins da máscara sem reposição (N = 2000 por padrão; se houver menos que N,
   usa todos) e amostra um ponto uniforme dentro de cada bin sorteado (semente fixa).
4. Passa os pontos z apenas pelo decoder, sem gradientes.
5. Calcula o score kNN (k = 5, distância L2 às 60k imagens de treino via torch.cdist em chunks).
6. Salva:
   - .npz com os z, as imagens decodificadas, os scores e os índices do bin de origem;
   - figura com grade 10x10 (100 imagens) ordenadas por score crescente, com o score abaixo;
   - figura com scatter dos pontos amostrados sobre o mapa log1p(H_mu), coloridos pelo score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_latent import load_full_train_60k
from train import set_seed, setup_device
from vae.config import load_config
from vae.model import VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Amostra pontos dos buracos no espaço latente 2D, decodifica e avalia via score kNN."
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
        help="Diretório base dos experimentos onde estão config.json e checkpoints/ (padrão: experiments/bestmodels)",
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
        "-n",
        "--n_samples",
        "--n-samples",
        dest="n_samples",
        type=int,
        default=2000,
        help="Número de bins a sortear sem reposição da máscara (padrão: 2000)",
    )
    parser.add_argument(
        "--max-contagem",
        "--max_contagem",
        dest="max_contagem",
        type=int,
        default=0,
        help="Usa bins com H <= esse valor em vez de H == 0, dentro do envelope 5x5 (padrão: 0)",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="desc",
        choices=["desc", "asc"],
        help="Ordenação das imagens na grade 10x10: 'desc' (maior erro primeiro) ou 'asc' (menor erro primeiro) (padrão: desc)",
    )
    parser.add_argument(
        "--max_ring_distance",
        "--max-ring-distance",
        dest="max_ring_distance",
        type=int,
        default=5,
        help="Distância máxima de anel d=1..max_ring_distance para recálculo de buracos caso necessário (padrão: 5)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Número de vizinhos mais próximos para o score kNN (padrão: 5)",
    )
    parser.add_argument(
        "--chunk_size",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=1000,
        help="Tamanho do chunk para torch.cdist (padrão: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente fixa para o sorteio dos bins e pontos uniformes (padrão: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de execução ('cuda', 'cpu' ou 'auto')",
    )
    return parser.parse_args()


def resolve_run_paths(args: argparse.Namespace) -> Tuple[str, Path, Path, Path]:
    """
    Resolve os caminhos do run:
    Retorna (run_name, npz_path, model_run_dir, out_run_dir).
    """
    run_spec = args.run_opt or args.run_arg
    output_dir = Path(args.output_dir)
    experiments_dir = Path(args.experiments_dir)

    if run_spec is None:
        # Se não foi passado, tenta descobrir o primeiro run com latent_analysis.npz em output_dir
        candidates = []
        if output_dir.is_dir():
            for p in sorted(output_dir.iterdir()):
                if p.is_dir() and ((p / "latent_analysis.npz").is_file() or (p / "npz" / "latent_analysis.npz").is_file()):
                    candidates.append(p)
        if not candidates:
            raise ValueError(
                "Nenhum run informado e nenhum latent_analysis.npz encontrado em "
                f"{output_dir}. Passe o run via linha de comando (ex: --run <run_dir>)."
            )
        target_dir = candidates[0]
        run_name = target_dir.name
    else:
        p = Path(run_spec)
        if p.is_file() and p.suffix == ".npz":
            target_dir = p.parent.parent if p.parent.name == "npz" else p.parent
            run_name = target_dir.name
        elif p.is_dir():
            target_dir = p
            run_name = p.name
        else:
            run_name = p.name
            target_dir = output_dir / run_name

    # Localiza latent_analysis.npz (na raiz do modelo ou em npz/)
    npz_candidates = [
        target_dir / "npz" / "latent_analysis.npz",
        target_dir / "latent_analysis.npz",
        output_dir / run_name / "npz" / "latent_analysis.npz",
        output_dir / run_name / "latent_analysis.npz",
        experiments_dir / run_name / "npz" / "latent_analysis.npz",
        experiments_dir / run_name / "latent_analysis.npz",
    ]
    npz_path = None
    for cand in npz_candidates:
        if cand.is_file():
            npz_path = cand
            break
    if npz_path is None:
        raise FileNotFoundError(
            f"Arquivo latent_analysis.npz não encontrado para o run '{run_name}'. "
            f"Caminhos verificados: {[str(c) for c in npz_candidates]}"
        )

    # Localiza diretório com config.json (raiz ou json/) e checkpoints/
    model_dir_candidates = [
        target_dir,
        output_dir / run_name,
        experiments_dir / run_name,
    ]
    model_run_dir = None
    for cand in model_dir_candidates:
        has_cfg = (cand / "config.json").is_file() or (cand / "json" / "config.json").is_file()
        has_ckpt = (cand / "checkpoints" / args.checkpoint).is_file()
        if has_cfg and has_ckpt:
            model_run_dir = cand
            break
    if model_run_dir is None:
        raise FileNotFoundError(
            f"Config ou checkpoint ({args.checkpoint}) não encontrado para o run '{run_name}'. "
            f"Caminhos verificados: {[str(c) for c in model_dir_candidates]}"
        )

    out_run_dir = target_dir if target_dir.is_dir() else output_dir / run_name
    return run_name, npz_path, model_run_dir, out_run_dir
    return run_name, npz_path, model_run_dir, out_run_dir


def build_candidate_mask(
    h_mu: np.ndarray,
    holes_mask_npz: np.ndarray,
    max_contagem: int,
    max_ring_distance: int = 5,
) -> np.ndarray:
    """
    Constrói a máscara de bins elegíveis:
    - Se max_contagem == 0, reaproveita diretamente a máscara de buracos do .npz (holes_mask_npz).
    - Se max_contagem > 0, seleciona bins com H_mu <= max_contagem que satisfaçam a condição
      de buraco por anel (todos os vizinhos de algum anel d=1..max_ring_distance ativos).
    """
    if max_contagem < 0:
        raise ValueError(f"--max-contagem deve ser >= 0, recebido: {max_contagem}")
    if max_contagem == 0:
        return (holes_mask_npz > 0).astype(np.uint8)

    # Recalcula condição de anel para bins com H_mu <= max_contagem
    from analyze_latent import compute_holes_mask
    ring_mask = compute_holes_mask(h_mu=h_mu, max_distance=max_ring_distance)
    mask = (h_mu <= max_contagem) & (ring_mask > 0)
    return mask.astype(np.uint8)


def sample_points_from_mask(
    mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorteia até n_samples bins da máscara sem reposição e, para cada bin sorteado (i, j),
    sorteia um ponto uniforme dentro dos limites [x_edges[i], x_edges[i+1]) x [y_edges[j], y_edges[j+1]).

    Retorna:
        z_samples: array float32 de shape (M, 2)
        sampled_bins: array int64 de shape (M, 2) com os índices (i, j) do bin de origem
    """
    candidate_bins = np.argwhere(mask > 0)  # shape (K, 2), cada linha é [i, j]
    total_candidates = len(candidate_bins)
    if total_candidates == 0:
        raise RuntimeError("Nenhum bin elegível encontrado na máscara.")

    m = min(n_samples, total_candidates)
    rng = np.random.default_rng(seed)

    chosen_idx = rng.choice(total_candidates, size=m, replace=False)
    sampled_bins = candidate_bins[chosen_idx].astype(np.int64)

    i_idx = sampled_bins[:, 0]
    j_idx = sampled_bins[:, 1]

    x_low = x_edges[i_idx]
    x_high = x_edges[i_idx + 1]
    y_low = y_edges[j_idx]
    y_high = y_edges[j_idx + 1]

    u_x = rng.uniform(0.0, 1.0, size=m)
    u_y = rng.uniform(0.0, 1.0, size=m)

    z1 = x_low + u_x * (x_high - x_low)
    z2 = y_low + u_y * (y_high - y_low)

    z_samples = np.stack([z1, z2], axis=1).astype(np.float32)
    return z_samples, sampled_bins


@torch.no_grad()
def decode_and_score_knn(
    model: VAE,
    z_samples: np.ndarray,
    train_images: torch.Tensor,
    k: int = 5,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Passa z_samples apenas pelo decoder sem gradientes e calcula, para cada imagem
    decodificada, a distância L2 média aos k vizinhos mais próximos no conjunto de
    treino inteiro (60k imagens) usando torch.cdist em chunks.
    """
    model.eval()
    device = train_images.device
    z_t = torch.from_numpy(z_samples).to(device=device, dtype=torch.float32)
    n = z_t.size(0)

    decoded_list = []
    scores_flat = torch.empty(n, dtype=torch.float32, device=device)

    for start in range(0, n, chunk_size):
        z_chunk = z_t[start : start + chunk_size]
        decoded_chunk = model.decode(z_chunk)  # (chunk, 784)
        dists = torch.cdist(decoded_chunk, train_images, p=2.0)  # (chunk, 60000)
        topk_dists, _ = torch.topk(dists, k=k, dim=1, largest=False)
        scores_flat[start : start + chunk_size] = topk_dists.mean(dim=1)
        decoded_list.append(decoded_chunk.cpu())

    decoded_all = torch.cat(decoded_list, dim=0).numpy().astype(np.float32)
    scores_all = scores_flat.cpu().numpy().astype(np.float32)
    return decoded_all, scores_all


def select_100_ordered_indices(n_total: int) -> np.ndarray:
    """
    Seleciona 100 índices ordenados do menor ao maior score:
    - Se n_total >= 100, seleciona 100 posições igualmente espaçadas de 0 até n_total - 1
      para cobrir dos menores aos maiores scores de toda a amostra.
    - Se n_total < 100, usa todos os disponíveis em ordem crescente.
    """
    if n_total <= 100:
        return np.arange(n_total, dtype=np.int64)
    return np.linspace(0, n_total - 1, 100, dtype=np.int64)


def save_grid_100_figure(
    decoded_images: np.ndarray,
    scores: np.ndarray,
    origin_bins: np.ndarray,
    z_samples: np.ndarray,
    run_name: str,
    max_contagem: int,
    save_path: Path,
    order_mode: str = "desc",
) -> np.ndarray:
    """
    Salva uma grade 10x10 com 100 imagens ordenadas por score kNN, exibindo abaixo de cada
    imagem seu rank (#ID), o bin de origem (i, j), a coordenada (z1, z2) e o score kNN.
    Retorna os índices selecionados (sel_idx) na ordem plotada.
    """
    if order_mode == "desc":
        order = np.argsort(-scores)
        title_order = "Maior Erro (Score kNN Decrescente)"
    else:
        order = np.argsort(scores)
        title_order = "Menor Erro (Score kNN Crescente)"

    chosen_pos = select_100_ordered_indices(len(order))
    sel_idx = order[chosen_pos]

    n_show = len(sel_idx)
    nrows, ncols = 10, 10

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 17.5))
    fig.suptitle(
        f"100 Imagens Decodificadas de Buracos (H <= {max_contagem}) Ordenadas por {title_order} — {run_name}",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    axes_flat = axes.ravel()
    for idx_plot, ax in enumerate(axes_flat):
        if idx_plot < n_show:
            sample_i = sel_idx[idx_plot]
            img = decoded_images[sample_i].reshape(28, 28)
            sc = scores[sample_i]
            bi, bj = origin_bins[sample_i]
            z1, z2 = z_samples[sample_i]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(
                f"#{idx_plot+1} bin({bi},{bj})\nz=({z1:+.2f},{z2:+.2f}) | {sc:.3f}",
                fontsize=7.0,
                labelpad=2,
            )
        else:
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return sel_idx


def save_scatter_figure(
    h_mu: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_samples: np.ndarray,
    scores: np.ndarray,
    origin_bins: np.ndarray,
    grid_sel_idx: np.ndarray,
    run_name: str,
    max_contagem: int,
    k: int,
    save_path: Path,
) -> None:
    """
    Salva um scatter dos pontos amostrados sobre o mapa log1p(H) dos mu,
    coloridos pelo score kNN e anotados com o #ID correspondente da grade de saída.
    """
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]
    log_h_mu = np.log1p(h_mu)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))

    im_bg = ax.imshow(
        log_h_mu.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Greys",
        interpolation="nearest",
        alpha=0.85,
    )
    cbar_bg = fig.colorbar(im_bg, ax=ax, fraction=0.046, pad=0.10, location="left")
    cbar_bg.set_label(r"Densidade $\log(1 + H(\mu))$")

    sc = ax.scatter(
        z_samples[:, 0],
        z_samples[:, 1],
        c=scores,
        cmap="magma_r",
        s=28,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.3,
    )
    cbar_sc = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar_sc.set_label(f"Score kNN (k={k})")

    # Anota os pontos presentes na grade (#1..#100; se > 30 pontos, anota os Top 25 no gráfico para legibilidade)
    max_labels = 25 if len(grid_sel_idx) > 30 else len(grid_sel_idx)
    for rank_idx in range(max_labels):
        sample_i = grid_sel_idx[rank_idx]
        zx, zy = z_samples[sample_i]
        ax.annotate(
            f"#{rank_idx+1}",
            (zx, zy),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="cyan",
            bbox=dict(boxstyle="round,pad=0.12", fc="black", ec="none", alpha=0.65),
        )

    ax.set_title(
        f"Pontos Amostrados (N={len(z_samples)}, H <= {max_contagem}) sobre log(1 + H(μ)) [IDs #1..#{max_labels} da Grade]\n{run_name}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_holes_mapping_csv(
    origin_bins: np.ndarray,
    z_samples: np.ndarray,
    scores: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    order_mode: str,
    save_path: Path,
) -> None:
    """
    Salva um arquivo CSV mapeando cada imagem da saída (#rank_na_grade) para o bin (i, j),
    intervalos do bin e coordenada exata (z1, z2) que a codificou.
    """
    order = np.argsort(-scores) if order_mode == "desc" else np.argsort(scores)
    chosen_pos = select_100_ordered_indices(len(order))
    grid_rank_by_sample = {int(order[pos]): rank_i + 1 for rank_i, pos in enumerate(chosen_pos)}

    lines = [
        "rank_geral,rank_na_grade_10x10,sample_index,bin_i,bin_j,z1,z2,bin_z1_min,bin_z1_max,bin_z2_min,bin_z2_max,score_knn"
    ]
    for rank_all, idx in enumerate(order, start=1):
        idx_int = int(idx)
        grid_pos = grid_rank_by_sample.get(idx_int, "")
        bi, bj = int(origin_bins[idx_int, 0]), int(origin_bins[idx_int, 1])
        z1, z2 = float(z_samples[idx_int, 0]), float(z_samples[idx_int, 1])
        sc = float(scores[idx_int])
        lines.append(
            f"{rank_all},{grid_pos},{idx_int},{bi},{bj},{z1:.6f},{z2:.6f},"
            f"{x_edges[bi]:.6f},{x_edges[bi+1]:.6f},{y_edges[bj]:.6f},{y_edges[bj+1]:.6f},{sc:.6f}"
        )

    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = setup_device(args.device)

    run_name, npz_path, model_run_dir, out_run_dir = resolve_run_paths(args)

    # 1. Lê máscara, H_mu e arestas dos bins do .npz do run
    with np.load(npz_path) as data:
        h_mu = data["H_mu"]
        holes_mask_npz = data["holes_mask"]
        x_edges = data["x_edges"]
        y_edges = data["y_edges"]

    mask = build_candidate_mask(
        h_mu=h_mu,
        holes_mask_npz=holes_mask_npz,
        max_contagem=args.max_contagem,
        max_ring_distance=args.max_ring_distance,
    )
    n_eligible = int((mask > 0).sum())

    print(f"[Run] {run_name}")
    print(f"  Arquivo base: {npz_path}")
    print(f"  Critério: H_mu <= {args.max_contagem} com buracos por anel (d=1..{args.max_ring_distance})")
    print(f"  Bins elegíveis na máscara: {n_eligible}")

    # 2. Sorteia N bins sem reposição e amostra 1 ponto uniforme dentro de cada bin sorteado
    z_samples, sampled_bins = sample_points_from_mask(
        mask=mask,
        x_edges=x_edges,
        y_edges=y_edges,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"  Pontos amostrados: {len(z_samples)} (seed={args.seed})")

    # 3. Carrega modelo e conjunto de treino (60k imagens)
    cfg_path = model_run_dir / "config.json"
    if not cfg_path.is_file():
        cfg_path = model_run_dir / "json" / "config.json"
    config = load_config(cfg_path)
    ckpt_path = model_run_dir / "checkpoints" / args.checkpoint
    model = VAE(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_images = load_full_train_60k(Path(args.dataset_dir), device=device)

    # 4. Decodifica apenas pelo decoder (sem gradientes) e calcula score kNN
    decoded_images, scores = decode_and_score_knn(
        model=model,
        z_samples=z_samples,
        train_images=train_images,
        k=args.k,
        chunk_size=args.chunk_size,
    )
    print(
        f"  Score kNN (k={args.k}) -> min: {scores.min():.3f}, "
        f"média: {scores.mean():.3f}, max: {scores.max():.3f}"
    )

    # 5. Salva .npz, figuras e tabela CSV de mapeamento
    suffix = "" if args.max_contagem == 0 else f"_maxH{args.max_contagem}"
    out_npz_dir = out_run_dir / "npz"
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    out_csv_dir = out_run_dir / "csv"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_analysis_dir = out_run_dir / "analysis"
    out_analysis_dir.mkdir(parents=True, exist_ok=True)

    out_npz_path = out_npz_dir / f"sampled_holes{suffix}.npz"
    grid_fig_path = out_analysis_dir / f"sampled_holes_grid{suffix}.png"
    scatter_fig_path = out_analysis_dir / f"sampled_holes_scatter{suffix}.png"
    csv_map_path = out_csv_dir / f"sampled_holes_map{suffix}.csv"

    np.savez_compressed(
        out_npz_path,
        z=z_samples,
        decoded_images=decoded_images,
        scores=scores,
        origin_bins=sampled_bins,
        max_contagem=np.int64(args.max_contagem),
    )

    grid_sel_idx = save_grid_100_figure(
        decoded_images=decoded_images,
        scores=scores,
        origin_bins=sampled_bins,
        z_samples=z_samples,
        run_name=run_name,
        max_contagem=args.max_contagem,
        save_path=grid_fig_path,
        order_mode=args.order,
    )

    save_scatter_figure(
        h_mu=h_mu,
        x_edges=x_edges,
        y_edges=y_edges,
        z_samples=z_samples,
        scores=scores,
        origin_bins=sampled_bins,
        grid_sel_idx=grid_sel_idx,
        run_name=run_name,
        max_contagem=args.max_contagem,
        k=args.k,
        save_path=scatter_fig_path,
    )

    save_holes_mapping_csv(
        origin_bins=sampled_bins,
        z_samples=z_samples,
        scores=scores,
        x_edges=x_edges,
        y_edges=y_edges,
        order_mode=args.order,
        save_path=csv_map_path,
    )

    print(f"  Salvo (.npz): {out_npz_path}")
    print(f"  Salvo (grade 100): {grid_fig_path}")
    print(f"  Salvo (scatter): {scatter_fig_path}")
    print(f"  Salvo (mapeamento CSV): {csv_map_path}")


if __name__ == "__main__":
    main()
