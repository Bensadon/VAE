# Variational Autoencoder (VAE) - Fashion-MNIST

Implementação modular em PyTorch de um **Variational Autoencoder (VAE)** para o dataset **Fashion-MNIST** (`28×28`, 10 classes de vestuário no formato binário IDX — e **não** o MNIST padrão de dígitos manuscritos), com restrições arquiteturais rígidas e suporte a experimentação sistemática.

---

## 📌 Especificações do Modelo

| Componente | Especificação |
| :--- | :--- |
| **Dataset** | **Fashion-MNIST** (10 classes de artigos de moda, `28×28` em tons de cinza, 48k treino / 12k val / 10k teste) |
| **Entrada / Saída** | `784` neurônios (Fashion-MNIST achatado) |
| **Ativação de Saída** | `Sigmoid` fixa (escala `[0.0, 1.0]`) |
| **Espelhamento** | O Decoder espelha estritamente a ordem invertida das camadas do Encoder |
| **Ativação Interna** | Idêntica em todas as camadas ocultas (`relu`, `sigmoid` ou `tanh`) |
| **Tamanho do Lote** | `64` exemplos (fixo) |
| **Otimizador** | `Adam` com hiperparâmetros configuráveis |
| **Função de Custo** | `L_total = L_recon + L_KL` (Soma bruta dos erros quadráticos sobre os 784 pixels + Divergência KL analítica) |
| **Pipeline de Dados** | Dataset 100% pré-carregado em **VRAM** (~210 MB alocados na GPU), eliminando transferências PCIe e gargalos de CPU |
| **Execução na GPU** | Passo de treino capturado em **CUDA Graph**, Adam `fused=True, capturable=True`, matmuls em **TF32** (`set_float32_matmul_precision("high")`) |

---

## 🎯 Objetivos do Projeto ([plans/planejamento.txt](file:///home/bensa/Personal/Projects/RN/ProjVAE/plans/planejamento.txt))

1. **Arquiteturas**: Comparar variações de camadas ocultas e funções de ativação.
2. **Navegabilidade**: Avaliar transições contínuas via interpolação linear no espaço latente.
3. **Clusterização**: Analisar agrupamento e separabilidade das 10 classes do Fashion-MNIST no espaço latente.
4. **Geração Sintética**: Gerar novas amostras decodificando pontos de `z ~ N(0, I)`.
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
├── dataset/               # Dados binários IDX do Fashion-MNIST particionados (48k / 12k / 10k)
│   └── raw/               # Arquivos brutos originais do Fashion-MNIST
├── experiments/           # Saídas dos experimentos (ex: experiments/round0/<run>)
├── plans/                 # Planejamento científico e rodadas de experimentos
│   ├── planejamento.txt   # Objetivos e hipóteses fundamentais
│   └── round1.txt         # Especificações de arquiteturas e parâmetros para o Round 0
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

# Treino controlando por passos de gradiente (ex: 200k passos com saída em experiments/round0)
python train.py --config configs/default_config.json --max_steps 200000 --output_dir experiments/round0

# Treino com ativação Sigmoid ou Tanh
python train.py --config configs/config_sigmoid.json
python train.py --config configs/config_tanh.json

# Treino com latente 8 e override de épocas e dispositivo
python train.py --config configs/config_latent8.json --epochs 50 --device cuda

# Estender/retomar um experimento a partir do último estado (last.pt)
python train.py --resume experiments/round0/<pasta_do_experimento> --max_steps 500000
```

#### Duração do treino: `epochs` vs `max_steps`

- Com batch 64 e 48.000 amostras de treino, **1 época = 750 passos** de gradiente.
- `max_steps` pode ser definido no JSON ou via `--max_steps`; o treino para exatamente nesse passo e o número de épocas é calculado automaticamente (`ceil(max_steps / 750)`).
- Se o JSON não definir `epochs` nem `max_steps`, o padrão é `epochs=267` (~200,25k passos).
- `--epochs` na CLI tem precedência sobre o cálculo a partir de `max_steps`.

#### Opções da CLI (`train.py`)

| Flag | Padrão | Descrição |
| :--- | :--- | :--- |
| `--config` | `configs/default_config.json` | JSON de configuração |
| `--epochs` | do JSON | Sobrescreve o número de épocas |
| `--max_steps` | do JSON / `None` | Limite de passos de gradiente |
| `--device` | do JSON | `cuda`, `cpu` ou `auto` |
| `--dataset_dir` | `dataset` | Diretório com os arquivos IDX |
| `--output_dir` | `experiments` | Diretório raiz das saídas (ex: `experiments/round0`) |
| `--name` | automático | Nome customizado da pasta da run |
| `--no_timestamp` | desligado | Força omitir o timestamp do nome da pasta |
| `--with_timestamp` | desligado | Força incluir o timestamp do nome da pasta (mesmo em rounds) |
| `--plot_interval` | `50` | Atualiza `loss_curve.png` a cada N épocas (e no fim) |
| `--ckpt_interval` | `50` | Salva `last.pt` a cada N épocas (e no fim) |
| `--eval_batch_size` | `16384` | Lote de avaliação; o padrão avalia val (12k) e teste (10k) em 1 forward cada |
| `--resume` | — | Pasta da run ou `last.pt` para retomar/estender |
| `--save_intermediate_recons` | desligado | Salva reconstruções intermediárias além das finais |
| `--no_cuda_graph` | desligado | Usa execução eager em vez de CUDA Graph |

---

## ⚡ Desempenho e Execução de Rodadas

Medido em uma **RTX 3050 6GB Laptop** com a arquitetura `[512, 256]`, latente 20:

| Configuração | Tempo por época |
| :--- | :--- |
| Eager (`fused=True`, perdas acumuladas na GPU) | ~4,3 s |
| **CUDA Graph (padrão)** | **~0,61 s** |

**Rode as runs em sequência, uma por vez.** Com CUDA Graph uma única run já satura a GPU; executar várias em paralelo não aumenta o throughput total (1,64 épocas/s com 1 run vs ~1,47 com 2–6 runs) e cada processo consome ~600–900 MiB de VRAM (contexto CUDA + cópia própria do dataset), limitando a ~6 processos em 6 GB.

Estimativa por run (`[512, 256]`; arquiteturas mais profundas/largas podem ser 1,5–2× mais lentas):

| Duração | Passos | Tempo aproximado |
| :--- | :--- | :--- |
| 267 épocas | 200k | ~2,7 min |
| 667 épocas | 500k | ~7 min |
| 10.000 épocas | 7,5M | ~1h40 |

> **Notas de reprodutibilidade**
> - O split treino/val é fixo em disco (gerado por `split_mnist.py` com semente própria); a `seed` do experimento afeta apenas a inicialização dos pesos e a ordem dos lotes.
> - Em `eval()` o modelo usa `z = mu` (sem amostragem), então a perda de validação é determinística e a seleção do melhor checkpoint não depende de ruído.
> - TF32 e CUDA Graph não mudam a matemática do modelo, mas resultados não são idênticos bit a bit a uma execução eager/FP32 estrita. Mantenha todas as runs de uma rodada no mesmo modo.
> - A seleção do melhor modelo usa **apenas a validação**; o teste é reportado na época do melhor val e nunca usado para escolha.
> - A perda de treino é a média ao longo da época (modelo em mudança), enquanto val/teste são medidos ao fim dela; treino um pouco acima de val no início é esperado.

---

## 📊 Artefatos Produzidos

A cada execução, uma pasta é criada em `<output_dir>/<prefixo>layers<num>_<ativação>_lat<dim>_lr<lr>_seed<s>/` (para diretórios com `round`, o timestamp é omitido automaticamente por padrão; em outros diretórios, inclui timestamp a menos que `--no_timestamp` seja passado) contendo:

- **`config.json`**: Cópia exata dos parâmetros utilizados (suporta `epochs` e `max_steps`).
- **`summary.json`**: Resumo final com status (`COMPLETED` ou `DIVERGED`), tempo médio por época, métricas da melhor época de validação e **perda de teste correspondente à época da melhor validação**.
- **`metrics.csv`**: Registro época a época de perdas (total, recon, KL), passo de gradiente, flag de melhor val, tempo por época e status (`OK`/`DIVERGED`) para treino, validação e teste.
- **`loss_curve.png`**: Gráficos das curvas de aprendizado gerados periodicamente (`--plot_interval`) e na conclusão.
- **`checkpoints/best_model.pt`**: Checkpoint do modelo e otimizador salvo no ponto de menor erro de validação.
- **`checkpoints/last.pt`**: Checkpoint do estado salvo a cada `--ckpt_interval` épocas e no fim para extensão/retomada via `--resume`.
- **`reconstructions/`**: Imagens comparativas (Original vs Reconstruído) geradas no melhor modelo de validação.
