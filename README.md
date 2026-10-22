# Variational Autoencoder (VAE) - MNIST

Implementação modular em PyTorch de um **Variational Autoencoder (VAE)** para o dataset MNIST ($28 \times 28$), com restrições arquiteturais rígidas e suporte a experimentação sistemática.

---

## 📌 Especificações do Modelo

| Componente | Especificação |
| :--- | :--- |
| **Entrada / Saída** | `784` neurônios (MNIST achatado) |
| **Ativação de Saída** | `Sigmoid` fixa (escala $[0.0, 1.0]$) |
| **Espelhamento** | O Decoder espelha estritamente a ordem invertida das camadas do Encoder |
| **Ativação Interna** | Idêntica em todas as camadas ocultas (`relu`, `sigmoid` ou `tanh`) |
| **Tamanho do Lote** | `64` exemplos (fixo) |
| **Otimizador** | `Adam` com hiperparâmetros configuráveis |
| **Função de Custo** | $\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}$ (Soma bruta dos erros quadráticos sobre os 784 pixels + Divergência KL analítica) |

---

## 🎯 Objetivos do Projeto ([planejamento.txt](file:///home/bensa/Personal/Projects/RN/ProjVAE/planejamento.txt))

1. **Arquiteturas**: Comparar variações de camadas ocultas e funções de ativação.
2. **Navegabilidade**: Avaliar transições contínuas via interpolação linear no espaço latente.
3. **Clusterização**: Analisar agrupamento e separabilidade das classes dos dígitos no espaço latente.
4. **Geração Sintética**: Gerar novos dígitos decodificando amostras de $z \sim \mathcal{N}(0, I)$.
5. **Correlação de Desempenho**: Investigar se menor erro médio de reconstrução acarreta melhor organização latente.

---

## 📁 Estrutura do Projeto

```plaintext
ProjVAE/
├── configs/               # Presets de hiperparâmetros em JSON
│   ├── default_config.json    # ReLU, [512, 256], latente=20, 20 épocas
│   ├── config_sigmoid.json    # Sigmoid interno, [512, 256], latente=20
│   ├── config_tanh.json       # Tanh interno, [512, 256], latente=20
│   └── config_latent8.json    # ReLU, latente=8, 50 épocas
├── dataset/               # Dados binários IDX particionados (48k / 12k / 10k)
│   └── raw/               # Arquivos brutos originais do MNIST
├── experiments/           # Saídas organizadas por timestamp_nome_experimento
├── vae/                   # Pacote do VAE
│   ├── config.py          # Configurações e validação de invariantes
│   ├── dataset.py         # Leitor IDX e DataLoaders PyTorch
│   ├── loss.py            # Função de custo (MSE 784 bruto + KL)
│   └── model.py           # Definição do Encoder, Decoder e reparametrização
├── split_mnist.py         # Divisão reproduzível do dataset (80% treino / 20% val)
├── verify_vae.py          # Suíte de testes automatizados e validação de regras
└── train.py               # Pipeline de treinamento, métricas e checkpointing
```

---

## 🚀 Guia Rápido de Uso

### 1. Requisitos e Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
```

### 2. Preparar Dataset (Particionamento)

Lê os arquivos de `dataset/raw/`, separa 48.000 amostras para treino e 12.000 para validação (semente 42), e preserva 10.000 para teste:

```bash
python split_mnist.py
```

### 3. Executar Testes Automatizados

Valida dimensões, espelhamento, cálculo do loss, passo do Adam e integridade do dataset:

```bash
python verify_vae.py
```

### 4. Treinar o Modelo

```bash
# Treino padrão (ReLU, latente=20)
python train.py --config configs/default_config.json

# Treino com ativação Sigmoid ou Tanh
python train.py --config configs/config_sigmoid.json
python train.py --config configs/config_tanh.json

# Treino com latente 8 e override de épocas e dispositivo
python train.py --config configs/config_latent8.json --epochs 50 --device cuda
```

---

## 📊 Artefatos Produzidos

A cada execução, uma pasta é criada em `experiments/<timestamp>_<experimento>/` contendo:

- **`config.json`**: Cópia exata dos parâmetros utilizados.
- **`metrics.csv`**: Registro época a época de perdas (total, recon, KL) para treino, validação e teste.
- **`loss_curve.png`**: Gráficos das curvas de aprendizado gerados a cada época.
- **`checkpoints/best_model.pt`**: Checkpoint do modelo e otimizador salvo no ponto de menor erro de validação.
- **`reconstructions/`**: Imagens comparativas (Original vs Reconstruído) geradas a cada novo recorde na validação.
