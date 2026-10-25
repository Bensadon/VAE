"""
Gera visualização 3D interativa em HTML (Plotly) por run de dimensão latente 2,
reaproveitando os resultados salvos em latent_analysis.npz pelo analyze_latent.py.

Padrão:
- x = centros dos bins em z1
- y = centros dos bins em z2
- z (altura) = log1p(H) dos mu
- cor da superfície = score kNN da decodificação (escala invertida: baixo=claro, alto=escuro)

Com --swap:
- z (altura) = score kNN da decodificação
- cor da superfície = log1p(H) dos mu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera superfície 3D interativa em HTML (Plotly) para runs de espaço latente 2D."
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments/bestmodels",
        help="Diretório de origem dos runs caso seja necessário calcular o .npz (padrão: experiments/bestmodels)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/latent_explore",
        help="Diretório onde os .npz e .html por run ficam salvos (padrão: experiments/latent_explore)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Caminho opcional para um único diretório de run (em output_dir ou experiments_dir)",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        help="Troca os papéis: altura (z) = score kNN e cor = densidade log(1+H)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Diretório do dataset caso seja necessário gerar o .npz ausente (padrão: dataset)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        choices=["best_model.pt", "last.pt"],
        help="Checkpoint usado caso seja necessário gerar o .npz ausente (padrão: best_model.pt)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Número de bins caso seja necessário gerar o .npz ausente (padrão: 100)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margem fracionária caso seja necessário gerar o .npz ausente (padrão: 0.05)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Vizinhos kNN caso seja necessário gerar o .npz ausente (padrão: 5)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Chunk size para cdist caso seja necessário gerar o .npz ausente (padrão: 1000)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Janela de buracos caso seja necessário gerar o .npz ausente (padrão: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente caso seja necessário gerar o .npz ausente (padrão: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo ('cuda', 'cpu' ou 'auto') caso seja necessário gerar o .npz ausente",
    )
    return parser.parse_args()


def resolve_targets(args: argparse.Namespace) -> List[Tuple[str, Path, Path | None]]:
    """
    Retorna lista de tuplas (run_name, npz_path, source_run_dir_if_needed).
    Prioriza reaproveitar os .npz já existentes em output_dir ou run_dir.
    """
    output_dir = Path(args.output_dir)
    experiments_dir = Path(args.experiments_dir)

    if args.run_dir is not None:
        run_path = Path(args.run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {run_path}")
        run_name = run_path.name
        direct_npz = run_path / "npz" / "latent_analysis.npz" if (run_path / "npz" / "latent_analysis.npz").is_file() else run_path / "latent_analysis.npz"
        out_npz = output_dir / run_name / "npz" / "latent_analysis.npz" if (output_dir / run_name / "npz" / "latent_analysis.npz").is_file() else output_dir / run_name / "latent_analysis.npz"
        if direct_npz.is_file():
            return [(run_name, direct_npz, None)]
        if out_npz.is_file():
            return [(run_name, out_npz, None)]
        return [(run_name, out_npz, run_path)]

    targets: List[Tuple[str, Path, Path | None]] = []
    seen = set()

    # 1. Procura runs que já possuem latent_analysis.npz em output_dir (na raiz ou em npz/)
    if output_dir.is_dir():
        npz_here = output_dir / "npz" / "latent_analysis.npz" if (output_dir / "npz" / "latent_analysis.npz").is_file() else output_dir / "latent_analysis.npz"
        if npz_here.is_file():
            targets.append((output_dir.name, npz_here, None))
            seen.add(output_dir.name)
        else:
            for p in sorted(output_dir.iterdir()):
                if p.is_dir():
                    cand_npz = p / "npz" / "latent_analysis.npz" if (p / "npz" / "latent_analysis.npz").is_file() else p / "latent_analysis.npz"
                    if cand_npz.is_file():
                        targets.append((p.name, cand_npz, None))
                        seen.add(p.name)

    # 2. Procura runs de latente 2 em experiments_dir ainda não processados
    if experiments_dir.is_dir():
        from analyze_latent import discover_latent2_runs

        for run_path in discover_latent2_runs(experiments_dir):
            if run_path.name not in seen:
                npz_path = output_dir / run_path.name / "npz" / "latent_analysis.npz"
                targets.append((run_path.name, npz_path, run_path))
                seen.add(run_path.name)

    return targets


def orient_for_plotly_surface(
    matrix: np.ndarray,
    n_x: int,
    n_y: int,
) -> np.ndarray:
    """
    Garante a orientação correta para go.Surface(x=x_centers, y=y_centers, z=...).

    No analyze_latent.py, H_mu[i, j] e knn_scores[i, j] têm:
      - índice i (eixo 0) em z1 (x_centers, tamanho n_x)
      - índice j (eixo 1) em z2 (y_centers, tamanho n_y)

    O go.Surface do Plotly com vetores 1D x (tamanho n_x) e y (tamanho n_y) espera que
    a matriz tenha shape (n_y, n_x), onde a linha j corresponde a y[j] (z2) e a coluna i
    corresponde a x[i] (z1). Portanto, transpõe-se a matriz (i, j) -> (j, i) sem alterar os dados.
    """
    if matrix.shape == (n_x, n_y):
        return matrix.T
    if matrix.shape == (n_y, n_x):
        return matrix
    raise ValueError(
        f"Shape incompatível da matriz {matrix.shape} para grade ({n_x}, {n_y})."
    )


def build_surface_html(
    npz_path: Path,
    run_name: str,
    output_dir: Path,
    swap: bool = False,
) -> Path:
    """
    Carrega os dados de latent_analysis.npz e gera o arquivo HTML 3D interativo.
    """
    with np.load(npz_path) as data:
        h_mu = data["H_mu"]
        knn_scores = data["knn_scores"]
        x_edges = data["x_edges"]
        y_edges = data["y_edges"]

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    n_x = len(x_centers)
    n_y = len(y_centers)

    log_h_mu = np.log1p(h_mu.astype(np.float64))

    # Transpõe de (i=z1, j=z2) para a convenção (linha=z2, coluna=z1) do go.Surface
    density_surf = orient_for_plotly_surface(log_h_mu, n_x, n_y)
    knn_surf = orient_for_plotly_surface(knn_scores.astype(np.float64), n_x, n_y)

    if not swap:
        z_data = density_surf
        color_data = knn_surf
        z_label = "log(1+H)"
        color_label = "Score kNN"
        # Escala invertida: score baixo = claro, score alto = escuro
        colorscale = "Magma"
        reversescale = True
        hovertemplate = (
            "z1: %{x:.3f}<br>"
            "z2: %{y:.3f}<br>"
            "log(1+H): %{z:.3f}<br>"
            "Score kNN: %{surfacecolor:.3f}<extra></extra>"
        )
    else:
        z_data = knn_surf
        color_data = density_surf
        z_label = "Score kNN"
        color_label = "log(1+H)"
        colorscale = "Viridis"
        reversescale = False
        hovertemplate = (
            "z1: %{x:.3f}<br>"
            "z2: %{y:.3f}<br>"
            "Score kNN: %{z:.3f}<br>"
            "log(1+H): %{surfacecolor:.3f}<extra></extra>"
        )

    surface = go.Surface(
        x=x_centers,
        y=y_centers,
        z=z_data,
        surfacecolor=color_data,
        colorscale=colorscale,
        reversescale=reversescale,
        colorbar=dict(title=dict(text=color_label)),
        hovertemplate=hovertemplate,
    )

    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=f"{run_name}",
        scene=dict(
            xaxis_title="z1",
            yaxis_title="z2",
            zaxis_title=z_label,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    out_run_dir = output_dir / run_name
    out_analysis_dir = out_run_dir / "analysis"
    out_analysis_dir.mkdir(parents=True, exist_ok=True)
    html_filename = "latent_3d_swapped.html" if swap else "latent_3d.html"
    html_path = out_analysis_dir / html_filename

    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
    )
    return html_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    targets = resolve_targets(args)

    if not targets:
        print("Nenhum run de latente 2 encontrado.")
        return

    train_images = None
    device = None

    for run_name, npz_path, source_run_dir in targets:
        if not npz_path.is_file():
            if source_run_dir is None:
                raise FileNotFoundError(f"Arquivo .npz não encontrado: {npz_path}")
            from analyze_latent import analyze_run, load_full_train_60k
            from train import setup_device

            if device is None:
                device = setup_device(args.device)
            if train_images is None:
                train_images = load_full_train_60k(Path(args.dataset_dir), device=device)
            _, npz_path = analyze_run(source_run_dir, train_images, device, args)
        else:
            print(f"[Run] {run_name} -> Reaproveitando {npz_path}")

        html_path = build_surface_html(
            npz_path=npz_path,
            run_name=run_name,
            output_dir=output_dir,
            swap=args.swap,
        )
        print(f"  HTML salvo em: {html_path}")


if __name__ == "__main__":
    main()
