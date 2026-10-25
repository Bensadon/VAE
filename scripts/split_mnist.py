
import struct
import shutil
from pathlib import Path

import numpy as np

DATASET_DIR = Path("dataset")
RAW_DIR = DATASET_DIR / "raw"
SEED = 42

FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}


def read_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Formato IDX de imagens inválido: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n * rows * cols:
        raise ValueError(f"Quantidade de pixels inválida: {path}")
    return data.reshape(n, rows, cols), (magic, n, rows, cols)


def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Formato IDX de rótulos inválido: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != n:
        raise ValueError(f"Quantidade de rótulos inválida: {path}")
    return data, (magic, n)


def write_idx_images(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, len(images),
                            images.shape[1], images.shape[2]))
        f.write(images.tobytes())


def write_idx_labels(path, labels):
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, len(labels)))
        f.write(labels.tobytes())


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {DATASET_DIR}")

    RAW_DIR.mkdir(exist_ok=True)

    # Não sobrescreve arquivos existentes no destino raw.
    for filename in FILES.values():
        src = DATASET_DIR / filename
        dst = RAW_DIR / filename

        if src.exists():
            if dst.exists():
                raise FileExistsError(
                    f"{dst} já existe; verifique antes de continuar."
                )
            shutil.move(str(src), str(dst))
        elif not dst.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {src} nem {dst}")

    # Lê os dados originais.
    images, _ = read_idx_images(RAW_DIR / FILES["train_images"])
    labels, _ = read_idx_labels(RAW_DIR / FILES["train_labels"])
    test_images, _ = read_idx_images(RAW_DIR / FILES["test_images"])
    test_labels, _ = read_idx_labels(RAW_DIR / FILES["test_labels"])

    if len(images) != len(labels):
        raise ValueError("Imagens e rótulos de treino têm tamanhos diferentes.")
    if len(test_images) != len(test_labels):
        raise ValueError("Imagens e rótulos de teste têm tamanhos diferentes.")
    if len(images) != 60000 or len(test_images) != 10000:
        raise ValueError("Os arquivos não têm os tamanhos esperados do MNIST.")

    # Embaralha de forma reproduzível e separa 80% / 20%.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(images))
    split = int(0.8 * len(images))

    train_idx = indices[:split]
    val_idx = indices[split:]

    # Grava os conjuntos mantendo imagens e rótulos alinhados.
    write_idx_images(DATASET_DIR / "train-images-idx3-ubyte",
                     images[train_idx])
    write_idx_labels(DATASET_DIR / "train-labels-idx1-ubyte",
                     labels[train_idx])

    write_idx_images(DATASET_DIR / "val-images-idx3-ubyte",
                     images[val_idx])
    write_idx_labels(DATASET_DIR / "val-labels-idx1-ubyte",
                     labels[val_idx])

    # O conjunto oficial de teste é preservado, sem reamostragem.
    shutil.copy2(RAW_DIR / FILES["test_images"],
                 DATASET_DIR / "test-images-idx3-ubyte")
    shutil.copy2(RAW_DIR / FILES["test_labels"],
                 DATASET_DIR / "test-labels-idx1-ubyte")

    print("Separação concluída!")
    print(f"Treino:     {len(train_idx)} exemplos")
    print(f"Validação:  {len(val_idx)} exemplos")
    print(f"Teste:      {len(test_images)} exemplos")
    print(f"Arquivos originais preservados em: {RAW_DIR}")


if __name__ == "__main__":
    main()

