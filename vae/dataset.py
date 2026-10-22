"""
Módulo de carregamento do dataset MNIST a partir dos arquivos binários IDX.
Normaliza as imagens para o intervalo [0.0, 1.0] e configura os DataLoaders
com o tamanho de lote rígido de 64 exemplos.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vae.config import RIGID_BATCH_SIZE


def read_idx_images(path: Path | str) -> np.ndarray:
    """Lê arquivo binário IDX3 de imagens (uint8, N x 28 x 28)."""
    path = Path(path)
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Formato IDX de imagens inválido: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n * rows * cols:
        raise ValueError(f"Quantidade de pixels incompatível com cabeçalho IDX: {path}")
    return data.reshape(n, rows, cols)


def read_idx_labels(path: Path | str) -> np.ndarray:
    """Lê arquivo binário IDX1 de rótulos (uint8, N)."""
    path = Path(path)
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Formato IDX de rótulos inválido: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n:
        raise ValueError(f"Quantidade de rótulos incompatível com cabeçalho IDX: {path}")
    return data


class MNISTIDXDataset(Dataset):
    """
    Dataset PyTorch para MNIST a partir dos arquivos IDX particionados.
    Retorna vetores achatados de 784 dimensões em float32 normalizados em [0.0, 1.0].
    """

    def __init__(self, images_path: Path | str, labels_path: Path | str):
        self.images_raw = read_idx_images(images_path)
        self.labels_raw = read_idx_labels(labels_path)

        if len(self.images_raw) != len(self.labels_raw):
            raise ValueError("Número de imagens e rótulos não coincide.")

        # Normalização estrita: divisão por 255.0 para escala [0.0, 1.0]
        # Formato achatado: (N, 784)
        n_samples = len(self.images_raw)
        images_flat = self.images_raw.reshape(n_samples, 784).astype(np.float32) / 255.0
        self.images = torch.from_numpy(images_flat)
        self.labels = torch.from_numpy(self.labels_raw.astype(np.int64))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def get_dataloaders(
    dataset_dir: Path | str = "dataset",
    batch_size: int = RIGID_BATCH_SIZE,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, MNISTIDXDataset]]:
    """
    Cria DataLoaders de Treino, Validação e Teste a partir do diretório dataset/.
    batch_size é estritamente rígido em 64 por especificação.
    """
    dataset_dir = Path(dataset_dir)
    train_img = dataset_dir / "train-images-idx3-ubyte"
    train_lbl = dataset_dir / "train-labels-idx1-ubyte"
    val_img = dataset_dir / "val-images-idx3-ubyte"
    val_lbl = dataset_dir / "val-labels-idx1-ubyte"
    test_img = dataset_dir / "test-images-idx3-ubyte"
    test_lbl = dataset_dir / "test-labels-idx1-ubyte"

    for p in [train_img, train_lbl, val_img, val_lbl, test_img, test_lbl]:
        if not p.is_file():
            raise FileNotFoundError(f"Arquivo do dataset não encontrado: {p}")

    train_ds = MNISTIDXDataset(train_img, train_lbl)
    val_ds = MNISTIDXDataset(val_img, val_lbl)
    test_ds = MNISTIDXDataset(test_img, test_lbl)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    return train_loader, val_loader, test_loader, datasets


def select_fixed_eval_samples(
    dataset: MNISTIDXDataset,
    num_samples: int = 16,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Seleciona um conjunto fixo e diversificado de amostras (cobrindo dígitos 0..9)
    para visualização consistente da reconstrução ao longo das épocas.
    """
    rng = np.random.default_rng(seed)
    labels = dataset.labels.numpy()

    selected_indices = []
    # Garante ao menos uma ocorrência de cada dígito de 0 a 9
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) > 0:
            selected_indices.append(rng.choice(digit_indices))

    # Preenche o restante até atingir num_samples
    remaining = num_samples - len(selected_indices)
    if remaining > 0:
        all_indices = np.setdiff1d(np.arange(len(dataset)), selected_indices)
        extra = rng.choice(all_indices, size=remaining, replace=False)
        selected_indices.extend(extra.tolist())

    selected_indices = np.array(selected_indices[:num_samples])
    images = dataset.images[selected_indices]
    chosen_labels = dataset.labels[selected_indices]
    return images, chosen_labels
