"""
Módulo de configuração do modelo VAE.
Garante a separação estrita entre parâmetros rígidos/imutáveis e
parâmetros configuráveis via arquivo JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Constantes rígidas e imutáveis da especificação
RIGID_INPUT_DIM: int = 784
RIGID_OUTPUT_DIM: int = 784
RIGID_BATCH_SIZE: int = 64
RIGID_OUTPUT_ACTIVATION: str = "sigmoid"
RIGID_OPTIMIZER: str = "adam"
RIGID_LOSS_TYPE: str = "mse_sum_plus_kl"

VALID_ACTIVATIONS = {"relu", "sigmoid", "tanh"}


@dataclass
class AdamConfig:
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class VAEConfig:
    # Parâmetros configuráveis via JSON
    experiment_name: str = "vae_experiment"
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256])
    latent_dim: int = 20
    activation: str = "relu"  # relu, sigmoid, tanh
    learning_rate: float = 0.001
    epochs: int = 267
    max_steps: Optional[int] = None
    adam_params: AdamConfig = field(default_factory=AdamConfig)
    device: str = "auto"  # auto, cuda, cpu
    seed: int = 42

    # Parâmetros rígidos/imutáveis (read-only)
    input_dim: int = RIGID_INPUT_DIM
    output_dim: int = RIGID_OUTPUT_DIM
    batch_size: int = RIGID_BATCH_SIZE
    output_activation: str = RIGID_OUTPUT_ACTIVATION
    optimizer: str = RIGID_OPTIMIZER
    loss_type: str = RIGID_LOSS_TYPE

    def __post_init__(self):
        # Validação da função de ativação
        act_lower = str(self.activation).strip().lower()
        if act_lower not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Função de ativação inválida: '{self.activation}'. "
                f"Opções permitidas: {sorted(VALID_ACTIVATIONS)}"
            )
        self.activation = act_lower

        # Validação do número de épocas e max_steps
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(f"O número de épocas deve ser um inteiro positivo, recebido: {self.epochs}")
        if self.max_steps is not None:
            if not isinstance(self.max_steps, int) or self.max_steps <= 0:
                raise ValueError(f"max_steps deve ser um inteiro positivo ou None, recebido: {self.max_steps}")

        # Validação da dimensão latente
        if not isinstance(self.latent_dim, int) or self.latent_dim <= 0:
            raise ValueError(f"A dimensão latente deve ser um inteiro positivo, recebido: {self.latent_dim}")

        # Validação das camadas ocultas
        if not isinstance(self.hidden_layers, list) or len(self.hidden_layers) == 0:
            raise ValueError(f"hidden_layers deve ser uma lista não-vazia de inteiros, recebido: {self.hidden_layers}")
        for idx, h in enumerate(self.hidden_layers):
            if not isinstance(h, int) or h <= 0:
                raise ValueError(f"Camada oculta #{idx} deve ter um número positivo de neurônios, recebido: {h}")

        # Validação da taxa de aprendizado
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError(f"A taxa de aprendizado deve ser positiva, recebido: {self.learning_rate}")
        self.learning_rate = float(self.learning_rate)

        # Garantia de imutabilidade dos parâmetros rígidos
        if self.input_dim != RIGID_INPUT_DIM:
            raise ValueError(f"input_dim é estritamente rígido e deve ser {RIGID_INPUT_DIM}")
        if self.output_dim != RIGID_OUTPUT_DIM:
            raise ValueError(f"output_dim é estritamente rígido e deve ser {RIGID_OUTPUT_DIM}")
        if self.batch_size != RIGID_BATCH_SIZE:
            raise ValueError(f"batch_size é estritamente rígido e deve ser {RIGID_BATCH_SIZE}")
        if self.output_activation != RIGID_OUTPUT_ACTIVATION:
            raise ValueError(f"output_activation é estritamente rígido e deve ser '{RIGID_OUTPUT_ACTIVATION}'")
        if self.optimizer != RIGID_OPTIMIZER:
            raise ValueError(f"optimizer é estritamente rígido e deve ser '{RIGID_OPTIMIZER}'")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.adam_params, AdamConfig):
            data["adam_params"] = asdict(self.adam_params)
        return data

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: Path | str) -> VAEConfig:
    """Carrega e valida a configuração a partir de um arquivo JSON."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Suporte para formato num_layers + neurons_per_layer
    if "hidden_layers" not in data and "num_layers" in data and "neurons_per_layer" in data:
        num_layers = int(data["num_layers"])
        neurons = int(data["neurons_per_layer"])
        data["hidden_layers"] = [neurons] * num_layers

    # Parsing dos parâmetros do Adam
    adam_dict = data.get("adam_params", {})
    betas_raw = adam_dict.get("betas", [0.9, 0.999])
    adam_config = AdamConfig(
        betas=(float(betas_raw[0]), float(betas_raw[1])),
        eps=float(adam_dict.get("eps", 1e-8)),
        weight_decay=float(adam_dict.get("weight_decay", 0.0)),
    )

    # Criação da instância com parâmetros permitidos
    max_steps_raw = data.get("max_steps", None)
    max_steps_val = int(max_steps_raw) if max_steps_raw is not None else None
    default_epochs = math.ceil(max_steps_val / 750) if max_steps_val is not None else 267

    return VAEConfig(
        experiment_name=str(data.get("experiment_name", "vae_experiment")),
        hidden_layers=[int(x) for x in data.get("hidden_layers", [512, 256])],
        latent_dim=int(data.get("latent_dim", 20)),
        activation=str(data.get("activation", "relu")),
        learning_rate=float(data.get("learning_rate", 0.001)),
        epochs=int(data.get("epochs", default_epochs)),
        max_steps=max_steps_val,
        adam_params=adam_config,
        device=str(data.get("device", "auto")),
        seed=int(data.get("seed", 42)),
        # Invariantes rígidas forçadas
        input_dim=RIGID_INPUT_DIM,
        output_dim=RIGID_OUTPUT_DIM,
        batch_size=RIGID_BATCH_SIZE,
        output_activation=RIGID_OUTPUT_ACTIVATION,
        optimizer=RIGID_OPTIMIZER,
        loss_type=RIGID_LOSS_TYPE,
    )
