# Planejamento Experimental — Round 0 (Exploração Inicial)

**Objetivo:** Avaliar o impacto da profundidade, afunilamento das camadas ocultas e funções de ativação em representações latentes de baixas dimensões (latente = 2 e latente = 8) no dataset **Fashion-MNIST**.

---

## 1. Arquiteturas das Redes

> **Convenção:** A entrada (784) e a saída (784 com Sigmoid) são omitidas. A lista define as camadas do Encoder. O Decoder espelha rigidamente a ordem inversa.

### Grupo A: Espaço Latente = 2 (Navegação e Visualização Direta 2D)
- **Rede 1:** `[512, 256, 128, 64, 32, 16, 8]` (7 camadas ocultas) — *Usa Adam beta2 = 0.99*
- **Rede 2:** `[392, 196, 98, 46, 22, 8]` (6 camadas ocultas) — *Usa Adam beta2 = 0.99*
- **Rede 3:** `[648, 512, 384, 256, 192, 128, 98, 64, 32, 16, 8]` (11 camadas ocultas) — *Usa Adam beta2 = 0.99*
- **Rede 4:** `[512, 256, 98, 32]` (4 camadas ocultas) — *Usa Adam beta2 = 0.999*

### Grupo B: Espaço Latente = 8 (Representação Compacta)
- **Rede 5:** `[512, 256, 128, 64, 32, 16]` (6 camadas ocultas) — *Usa Adam beta2 = 0.99*
- **Rede 6:** `[648, 512, 384, 256, 192, 128, 98, 64, 32, 16]` (10 camadas ocultas) — *Usa Adam beta2 = 0.99*
- **Rede 7:** `[256, 98, 32]` (3 camadas ocultas) — *Usa Adam beta2 = 0.999*
- **Rede 8:** `[384, 192, 64, 16]` (4 camadas ocultas) — *Usa Adam beta2 = 0.999*

---

## 2. Hiperparâmetros e Configurações

### Parâmetros Fixos
- **Dataset:** Fashion-MNIST (`28x28`, normalizado em `[0.0, 1.0]`)
- **Tamanho do Lote (Batch Size):** 64
- **Duração do Treinamento:** 250.000 passos de gradiente (~334 épocas de 750 passos, ~4 a 5 min por run)
- **Taxa de Aprendizado (LR):** `1e-4` (0.0001) fixa para todas as runs
- **Otimizador:** Adam (`eps = 1e-8`, `weight_decay = 0.0`)
  - **$\beta_2 = 0.999$** para arquiteturas com **menos de 5 camadas ocultas** (Redes 4, 7 e 8: `betas = [0.9, 0.999]`)
  - **$\beta_2 = 0.99$** para arquiteturas com **5 ou mais camadas ocultas** (Redes 1, 2, 3, 5 e 6: `betas = [0.9, 0.99]`)
- **Diretório Raiz de Saída:** `experiments/round0/`

### Variáveis de Exploração
- **Funções de Ativação Interna:**
  - Grupo A (Latente = 2): `relu`, `sigmoid`, `tanh`
  - Grupo B (Latente = 8): `relu` *(Sigmoid e Tanh excluídas; apenas ReLU)*
- **Sementes Aleatórias (Seeds):** `42` e `24`

---

## 3. Estrutura de Execução

Todas as redes são treinadas integralmente em `LR = 10^-4` com seus respectivos $\beta_2$ do Adam em ambas as seeds:
- Grupo A (Latente 2): 4 redes × 3 ativações × 2 seeds = **24 runs**
- Grupo B (Latente 8): 4 redes × 1 ativação (ReLU) × 2 seeds = **8 runs**
- **Total do Round 0:** **32 runs** (250k passos / ~334 épocas cada).

Ao final das 40 runs, o ranqueamento automático calcula a média da melhor perda de validação entre as seeds 42 e 24 e seleciona:
- **Top 3 modelos** do Grupo Latente = 2
- **Top 3 modelos** do Grupo Latente = 8
Gerando o relatório consolidado final em `experiments/round0/round0_report.md`.
