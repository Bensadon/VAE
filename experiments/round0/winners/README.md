# Campeões do Round 0 (Fashion-MNIST VAE)

Diretório: `experiments/round0/winners/`

Critério de seleção: As duas melhores redes de arquiteturas distintas em cada espaço latente, avaliadas pela média das sementes 42 e 24 ao longo de 250.000 passos com LR = $10^{-4}$.

---

## 1. Grupo A — Espaço Latente = 2

### 🥇 1º Lugar Latente 2: `rede4` (`winner1_lat2_rede4`)
- **Arquitetura Encoder:** `[512, 256, 98, 32]` (4 camadas)
- **Ativação Interna:** `ReLU`
- **Otimizador:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\text{LR} = 10^{-4}$)
- **Média Val Loss:** **23.74** (Reconstrução MSE = 18.20, KL = 5.53)
  - Seed 42: Val Loss = 23.80 (MSE = 18.18, KL = 5.62)
  - Seed 24: Val Loss = 23.67 (MSE = 18.22, KL = 5.45)
- **Melhor Checkpoint Geral:** `winner1_lat2_rede4/best_overall_model.pt` (da Seed 24)

### 🥈 2º Lugar Latente 2: `rede1` (`winner2_lat2_rede1`)
- **Arquitetura Encoder:** `[512, 256, 128, 64, 32, 16, 8]` (7 camadas)
- **Ativação Interna:** `ReLU`
- **Otimizador:** Adam ($\beta_1 = 0.9, \beta_2 = 0.990$, $\text{LR} = 10^{-4}$)
- **Média Val Loss:** **24.05** (Reconstrução MSE = 18.66, KL = 5.39)
  - Seed 42: Val Loss = 24.06 (MSE = 18.67, KL = 5.39)
  - Seed 24: Val Loss = 24.03 (MSE = 18.65, KL = 5.38)
- **Melhor Checkpoint Geral:** `winner2_lat2_rede1/best_overall_model.pt` (da Seed 24)

---

## 2. Grupo B — Espaço Latente = 8

### 🥇 1º Lugar Latente 8: `rede7` (`winner1_lat8_rede7`) — *Campeã Geral Round 0*
- **Arquitetura Encoder:** `[256, 98, 32]` (3 camadas)
- **Ativação Interna:** `ReLU`
- **Otimizador:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\text{LR} = 10^{-4}$)
- **Média Val Loss:** **21.19** (Reconstrução MSE = 14.17, KL = 7.02)
  - Seed 42: Val Loss = 20.96 (MSE = 13.78, KL = 7.18)
  - Seed 24: Val Loss = 21.41 (MSE = 14.55, KL = 6.87)
- **Melhor Checkpoint Geral:** `winner1_lat8_rede7/best_overall_model.pt` (da Seed 42)

### 🥈 2º Lugar Latente 8: `rede8` (`winner2_lat8_rede8`)
- **Arquitetura Encoder:** `[384, 192, 64, 16]` (4 camadas)
- **Ativação Interna:** `ReLU`
- **Otimizador:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\text{LR} = 10^{-4}$)
- **Média Val Loss:** **21.81** (Reconstrução MSE = 14.86, KL = 6.95)
  - Seed 42: Val Loss = 22.35 (MSE = 15.35, KL = 7.00)
  - Seed 24: Val Loss = 21.27 (MSE = 14.36, KL = 6.91)
- **Melhor Checkpoint Geral:** `winner2_lat8_rede8/best_overall_model.pt` (da Seed 24)

---

## 3. Resumo Comparativo dos 4 Campeões

| Categoria | Pasta | Rede | Latente | Camadas | Arquitetura Encoder | Média Val | Média MSE | Média KL |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Top 1 (Lat 8)** | `winner1_lat8_rede7` | `rede7` | 8 | 3 | `[256, 98, 32]` | **21.19** | **14.17** | 7.02 |
| **Top 2 (Lat 8)** | `winner2_lat8_rede8` | `rede8` | 8 | 4 | `[384, 192, 64, 16]` | **21.81** | **14.86** | 6.95 |
| **Top 1 (Lat 2)** | `winner1_lat2_rede4` | `rede4` | 2 | 4 | `[512, 256, 98, 32]` | **23.74** | **18.20** | 5.53 |
| **Top 2 (Lat 2)** | `winner2_lat2_rede1` | `rede1` | 2 | 7 | `[512, 256, 128, 64, 32, 16, 8]` | **24.05** | **18.66** | 5.39 |
