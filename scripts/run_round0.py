#!/usr/bin/env python3
"""
Script de orquestração automatizada para o Round 0 de experimentos VAE (Fashion-MNIST).
Regras atualizadas:
- Taxa de Aprendizado (LR): 1e-4 para todas as runs
- Adam:
  - beta2 = 0.999 para arquiteturas com < 5 camadas ocultas (Redes 4, 7, 8)
  - beta2 = 0.99  para arquiteturas com >= 5 camadas ocultas (Redes 1, 2, 3, 5, 6)
- 8 redes x 3 ativações x 2 seeds (42 e 24) = 48 runs de 500k passos (~667 épocas)
- Ranqueamento e relatório consolidado final em Markdown e JSON
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Definição das 8 redes conforme plans/round0.md
NETWORKS = {
    1: {"name": "rede1", "hidden_layers": [512, 256, 128, 64, 32, 16, 8], "latent_dim": 2},     # 7 camadas (>= 5 -> b2=0.99)
    2: {"name": "rede2", "hidden_layers": [392, 196, 98, 46, 22, 8], "latent_dim": 2},           # 6 camadas (>= 5 -> b2=0.99)
    3: {"name": "rede3", "hidden_layers": [648, 512, 384, 256, 192, 128, 98, 64, 32, 16, 8], "latent_dim": 2}, # 11 camadas (>= 5 -> b2=0.99)
    4: {"name": "rede4", "hidden_layers": [512, 256, 98, 32], "latent_dim": 2},                   # 4 camadas (< 5  -> b2=0.999)
    5: {"name": "rede5", "hidden_layers": [512, 256, 128, 64, 32, 16], "latent_dim": 8},          # 6 camadas (>= 5 -> b2=0.99)
    6: {"name": "rede6", "hidden_layers": [648, 512, 384, 256, 192, 128, 98, 64, 32, 16], "latent_dim": 8}, # 10 camadas (>= 5 -> b2=0.99)
    7: {"name": "rede7", "hidden_layers": [256, 98, 32], "latent_dim": 8},                        # 3 camadas (< 5  -> b2=0.999)
    8: {"name": "rede8", "hidden_layers": [384, 192, 64, 16], "latent_dim": 8},                   # 4 camadas (< 5  -> b2=0.999)
}

ACTIVATIONS = ["relu", "sigmoid", "tanh"]
SEEDS = [42, 24]
LEARNING_RATE = 0.0001  # LR 10^-4 para todas as runs
MAX_STEPS = 250000      # 250k passos (~334 épocas, ~4 a 5 min por run)
OUTPUT_DIR = Path("experiments/round0")
CONFIGS_DIR = Path("configs/round0")


def get_activations_for_network(latent_dim: int) -> List[str]:
    """
    Retorna as ativações a testar para cada rede:
    - Grupo A (latente 2): relu, sigmoid, tanh
    - Grupo B (latente 8): somente relu
    """
    if latent_dim == 8:
        return ["relu"]
    return ["relu", "sigmoid", "tanh"]


def get_adam_betas(num_layers: int) -> Tuple[float, float]:
    """
    Retorna os betas do Adam:
    - Se num_layers < 5:  beta2 = 0.999 (betas = [0.9, 0.999])
    - Se num_layers >= 5: beta2 = 0.99  (betas = [0.9, 0.99])
    """
    if num_layers < 5:
        return (0.9, 0.999)
    else:
        return (0.9, 0.99)


def create_config_file(net_info: Dict[str, Any], activation: str, seed: int) -> Path:
    """Gera o arquivo JSON de configuração para a run."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    num_layers = len(net_info["hidden_layers"])
    betas = get_adam_betas(num_layers)

    cfg_data = {
        "experiment_name": net_info["name"],
        "hidden_layers": net_info["hidden_layers"],
        "latent_dim": net_info["latent_dim"],
        "activation": activation,
        "learning_rate": LEARNING_RATE,
        "epochs": math.ceil(MAX_STEPS / 750),  # 667 épocas
        "max_steps": MAX_STEPS,
        "device": "cuda",
        "seed": seed,
        "adam_params": {
            "betas": list(betas),
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
    }

    cfg_filename = f"{net_info['name']}_layers{num_layers}_{activation}_lat{net_info['latent_dim']}_lr{LEARNING_RATE}_seed{seed}.json"
    cfg_path = CONFIGS_DIR / cfg_filename
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)
    return cfg_path


def get_expected_folder(net_info: Dict[str, Any], activation: str, seed: int) -> Path:
    num_layers = len(net_info["hidden_layers"])
    folder_name = f"{net_info['name']}_layers{num_layers}_{activation}_lat{net_info['latent_dim']}_lr{LEARNING_RATE}_seed{seed}"
    return OUTPUT_DIR / folder_name


def is_run_completed(run_dir: Path) -> bool:
    """Verifica se uma run já foi finalizada com sucesso ou divergiu."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("status") in ("COMPLETED", "DIVERGED")
    except Exception:
        return False


def get_run_best_val(run_dir: Path) -> float:
    """Recupera o menor val_loss do summary.json."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return float("inf")
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("diverged"):
            return float("inf")
        val = data.get("best_val", {}).get("val_loss")
        return float(val) if val is not None else float("inf")
    except Exception:
        return float("inf")


def get_run_full_summary(run_dir: Path) -> Dict[str, Any]:
    """Retorna o conteúdo completo de summary.json."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def execute_run(net_info: Dict[str, Any], activation: str, seed: int, idx: int, total: int) -> None:
    """Executa uma run via train.py se ainda não estiver concluída."""
    run_dir = get_expected_folder(net_info, activation, seed)
    if is_run_completed(run_dir):
        best_val = get_run_best_val(run_dir)
        print(f"[{idx:02d}/{total:02d}] RUN JÁ CONCLUÍDA: {run_dir.name} (Best Val: {best_val:.2f}) - Pulando...")
        return

    num_layers = len(net_info["hidden_layers"])
    betas = get_adam_betas(num_layers)
    cfg_path = create_config_file(net_info, activation, seed)

    # Suporte a retomar de last.pt se existir
    cmd = [
        sys.executable,
        "train.py",
        "--config", str(cfg_path),
        "--output_dir", str(OUTPUT_DIR),
        "--max_steps", str(MAX_STEPS),
    ]
    if (run_dir / "checkpoints" / "last.pt").is_file():
        cmd.extend(["--resume", str(run_dir)])
        print(f"[{idx:02d}/{total:02d}] RETOMANDO RUN EM PROGRESSO: {run_dir.name}")
    else:
        print("\n" + "=" * 80)
        print(f"[{idx:02d}/{total:02d}] INICIANDO: {run_dir.name}")
        print(f"Rede: {net_info['name']} | Camadas: {num_layers} | Latente: {net_info['latent_dim']} | Ativação: {activation.upper()}")
        print(f"LR: {LEARNING_RATE} | Adam betas: {betas} | Seed: {seed} | Passos: {MAX_STEPS}")
        print("=" * 80)

    t0 = time.perf_counter()
    res = subprocess.run(cmd)
    dt = time.perf_counter() - t0

    if res.returncode != 0:
        print(f"[ERRO] Falha ao executar {run_dir.name} (código de saída: {res.returncode})")
    else:
        best_val = get_run_best_val(run_dir)
        print(f"[{idx:02d}/{total:02d}] CONCLUÍDO em {dt:.1f}s ({dt/60:.1f} min) | Best Val Loss: {best_val:.2f}")


def generate_consolidated_report(all_models_scores: List[Dict[str, Any]]) -> None:
    """Gera o relatório consolidado em Markdown com o ranking geral do Round 0."""
    group_lat2 = [m for m in all_models_scores if m["latent_dim"] == 2]
    group_lat8 = [m for m in all_models_scores if m["latent_dim"] == 8]

    group_lat2.sort(key=lambda x: x["avg_score"])
    group_lat8.sort(key=lambda x: x["avg_score"])

    top3_lat2 = group_lat2[:3]
    top3_lat8 = group_lat8[:3]

    lines = [
        "# Relatório Consolidado — Round 0 (Fashion-MNIST VAE)",
        "",
        f"**Data de Conclusão:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Diretório:** `{OUTPUT_DIR}`",
        f"**Configuração Geral:** Taxa de Aprendizado $\\text{{LR}} = 10^{{-4}}$ fixa, 500.000 passos por run (~667 épocas).",
        "**Regra do Adam:** $\\beta_2 = 0.999$ para arquiteturas com $<5$ camadas; $\\beta_2 = 0.99$ para arquiteturas com $\\ge 5$ camadas.",
        "",
        "---",
        "",
        "## 1. Destaques: Top 3 Modelos por Espaço Latente",
        "",
        "### Top 3 — Grupo A (Latente = 2)",
        "| Posição | Rede | Camadas | Adam $\\beta_2$ | Ativação | Seed 42 | Seed 24 | Média Val Loss |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for rank, m in enumerate(top3_lat2, 1):
        lines.append(f"| #{rank} | **{m['name']}** | {m['num_layers']} | {m['beta2']} | `{m['activation']}` | {m['val_seed42']:.2f} | {m['val_seed24']:.2f} | **{m['avg_score']:.2f}** |")

    lines.extend([
        "",
        "### Top 3 — Grupo B (Latente = 8)",
        "| Posição | Rede | Camadas | Adam $\\beta_2$ | Ativação | Seed 42 | Seed 24 | Média Val Loss |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for rank, m in enumerate(top3_lat8, 1):
        lines.append(f"| #{rank} | **{m['name']}** | {m['num_layers']} | {m['beta2']} | `{m['activation']}` | {m['val_seed42']:.2f} | {m['val_seed24']:.2f} | **{m['avg_score']:.2f}** |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Ranking Geral Completo (Todas as 24 Combinações)",
        "",
        "| Posição | Rede | Latente | Camadas | Estrutura | Adam $\\beta_2$ | Ativação | Val Seed 42 | Val Seed 24 | Média Val |",
        "| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    sorted_all = sorted(all_models_scores, key=lambda x: x["avg_score"])
    for rank, m in enumerate(sorted_all, 1):
        v42_str = f"{m['val_seed42']:.2f}" if not math.isinf(m['val_seed42']) else "Divergiu"
        v24_str = f"{m['val_seed24']:.2f}" if not math.isinf(m['val_seed24']) else "Divergiu"
        avg_str = f"**{m['avg_score']:.2f}**" if not math.isinf(m['avg_score']) else "—"
        arch_str = str(m['hidden_layers'])
        lines.append(f"| {rank} | {m['name']} | {m['latent_dim']} | {m['num_layers']} | `{arch_str}` | {m['beta2']} | `{m['activation']}` | {v42_str} | {v24_str} | {avg_str} |")

    report_path = OUTPUT_DIR / "round0_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Salva também os dados estruturados em JSON
    json_path = OUTPUT_DIR / "round0_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "learning_rate": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "top3_lat2": top3_lat2,
            "top3_lat8": top3_lat8,
            "all_models": sorted_all,
        }, f, indent=2)

    print(f"\n[SUCESSO] Relatório consolidado gerado em: {report_path}")
    print(f"[SUCESSO] Resultados em JSON salvos em:     {json_path}")


def main() -> None:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Grupo A (latente 2): 4 redes x 3 ativações x 2 seeds = 24 runs
    # Grupo B (latente 8): 4 redes x 1 ativação (relu) x 2 seeds = 8 runs
    # Total = 32 runs
    total_runs = sum(
        len(get_activations_for_network(net["latent_dim"])) * len(SEEDS)
        for net in NETWORKS.values()
    )

    print("=" * 80)
    print("INICIALIZANDO PIPELINE DE TESTES COMPLETO — ROUND 0")
    print(f"Diretório de saída:     {OUTPUT_DIR}")
    print(f"Total de runs:          {total_runs} runs consecutivas (Grupo B exclusivamente ReLU)")
    print(f"Passos por run:         {MAX_STEPS} passos (~334 épocas)")
    print(f"Taxa de aprendizado:    LR = {LEARNING_RATE} (fixa)")
    print("Regra Adam beta2:       < 5 camadas -> 0.999 | >= 5 camadas -> 0.99")
    print("=" * 80)

    # Executa todas as runs sequencialmente
    run_idx = 1
    for net_id in sorted(NETWORKS.keys()):
        net_info = NETWORKS[net_id]
        acts = get_activations_for_network(net_info["latent_dim"])
        for act in acts:
            for seed in SEEDS:
                execute_run(net_info, act, seed=seed, idx=run_idx, total=total_runs)
                run_idx += 1

    # Coleta métricas e calcula rankings
    print("\n" + "=" * 80)
    print("TODAS AS RUNS CONCLUÍDAS! CALCULANDO RANKINGS E GERANDO RELATÓRIO...")
    print("=" * 80)

    models_scores: List[Dict[str, Any]] = []
    for net_id in sorted(NETWORKS.keys()):
        net_info = NETWORKS[net_id]
        num_layers = len(net_info["hidden_layers"])
        betas = get_adam_betas(num_layers)
        acts = get_activations_for_network(net_info["latent_dim"])

        for act in acts:
            v42 = get_run_best_val(get_expected_folder(net_info, act, seed=42))
            v24 = get_run_best_val(get_expected_folder(net_info, act, seed=24))

            if math.isinf(v42) or math.isinf(v24):
                avg_score = float("inf")
            else:
                avg_score = (v42 + v24) / 2.0

            models_scores.append({
                "net_id": net_id,
                "name": net_info["name"],
                "hidden_layers": net_info["hidden_layers"],
                "num_layers": num_layers,
                "latent_dim": net_info["latent_dim"],
                "activation": act,
                "beta2": betas[1],
                "val_seed42": v42,
                "val_seed24": v24,
                "avg_score": avg_score,
            })

    generate_consolidated_report(models_scores)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ROUND 0 FINALIZADO COM SUCESSO EM {total_time/3600:.2f} HORAS ({total_time/60:.1f} MINUTOS)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
