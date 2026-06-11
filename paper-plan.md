# Plano de Paper: Chameleon-IDS com PSO-GWO Híbrido

## Título Proposto

> **A Hybrid PSO-GWO Approach for Mixed-Type Feature Selection in Network Intrusion Detection with Autoencoder-Based Anomaly Detection**

---

## 1. Resumo da Contribuição Científica

### O que é original

1. **PSO-GWO Híbrido em nível de equação de velocidade**
   - A maioria dos trabalhos usa PSO ou GWO de forma isolada, ou em cascata (um depois do outro)
   - Nossa proposta **reformula a velocidade do PSO** para incluir a influência dos 3 lobos líderes do GWO (alpha, beta, delta), mantendo a inércia e o personal best do PSO
   - Isso é uma **hibridização em nível de equação**, não uma cascata de algoritmos

2. **Espaço de busca misto com discretização adaptativa**
   - A partícula codifica simultaneamente tipos distintos:
     - `test_size`: contínuo [0.1, 0.4]
     - `features`: binário {0, 1}
     - `n_estimators`: inteiro [50, 1000]
     - `learning_rate`: contínuo [0.1, 0.3]
   - A discretização por threshold (`> 0.5 → 1`) é uma heurística adaptativa que não é nem BPSO clássico (sigmoid) nem binário forçado

3. **Função de fitness bi-objetivo explicitada**
   ```python
   fitness = w * f1 + (1 - w) * (1 - n_features / n_total)
   ```
   - Maximiza F1-score do ensemble
   - Minimiza o número de features selecionadas
   - A maioria dos trabalhos usa apenas F1 ou recorre a NSGA-II/PSO multi-objetivo complexo

4. **Pipeline em cascata: Ensemble → Autoencoder**
   - O PSO-GWO usa o **ensemble (GradientBoosting/RandomForest)** como função de fitness
   - A solução ótima é então usada para treinar o **Autoencoder** (apenas em tráfego benigno)
   - Detecção final por anomalia (MSE > threshold)

---

## 2. Estrutura do Paper

### Abstract (5 partes)
1. **Problema:** Seleção de features em IDS é crítica para reduzir overhead e melhorar precisão
2. **Gap:** Meta-heurísticas puras (PSO, GWO) sofrem de convergência prematura ou exploração insuficiente em espaços mistos contínuo-binários
3. **Proposta:** Hibridização PSO-GWO que combina inércia do PSO com hierarquia de líderes do GWO
4. **Experimento:** NSL-KDD, CICIDS2017; comparação com PSO, GWO, BPSO, sem seleção
5. **Resultado:** Melhor F1 com menos features, convergência mais rápida

### 1. Introduction
- Contexto de IDS e volume crescente de tráfego de rede
- Problema da dimensionalidade (NSL-KDD tem 41+ features)
- Seleção de features como pré-processamento essencial
- Autoencoder para detecção de anomalias (unsupervised)
- **Gap identificado:** Ninguém aplicou PSO-GWO híbrido para seleção de features em IDS com pipeline ensemble → autoencoder

### 2. Related Work
- PSO para feature selection (Kennedy & Eberhart, 1995; BPSO, 1997)
- GWO para feature selection (Mirjalili et al., 2014)
- Híbridos existentes (PSO-GA, GWO-SA, PSO-Firefly)
- **Diferença do nosso trabalho:** Híbrido em nível de equação de velocidade, não cascata de algoritmos

### 3. Proposed Method: Chameleon-PSO-GWO

#### 3.1 Particle Representation

| Dimensão | Tipo | Intervalo | Descrição |
|----------|------|-----------|-----------|
| `position[0]` | float | [0.1, 0.4] | Fração train/test split |
| `position[1..n]` | binário | {0, 1} | Máscara de seleção de features |
| `position[-2]` (gb) | inteiro | [50, 1000] | n_estimators do GradientBoosting |
| `position[-1]` (gb) | float | [0.1, 0.3] | learning_rate do GradientBoosting |
| `position[-1]` (rf) | inteiro | [50, 600] | n_estimators do RandomForest |

#### 3.2 PSO-GWO Velocity Equation

A velocidade híbrida é definida como:

```
v(t+1) = w * v(t) + (c/2) * r * (x1 - x) + (c/3) * r * (x2 - x) + (c/4) * r * (x3 - x)
```

Onde:
- `w`: inércia do PSO
- `x1, x2, x3`: posições influenciadas pelos lobos alpha, beta, delta
- `c`: fator de convergência adaptativo
- `r`: número aleatório [0, 1]

#### 3.3 Fitness Function

```
fitness = w * f1_score + (1 - w) * (1 - n_selected / n_total)
```

- `w` = 0.8 (peso para F1)
- `1 - w` = 0.2 (peso para minimização de features)

#### 3.4 Pipeline

```
Dataset (NSL-KDD / CICIDS2017)
       │
       ▼
┌─────────────────────────────────────────┐
│    Chameleon-PSO-GWO (Enxame)           │
│                                         │
│  • Seleção de features (binário)        │
│  • Hiperparâmetros do ensemble          │
│  • Fitness = F1 ponderado + features    │
│                                         │
│  Saída: melhor subconjunto de features  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Ensemble (GradientBoosting)          │
│    Avaliação do fitness durante PSO     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Autoencoder (PyTorch)                │
│                                         │
│  • Treinado APENAS em tráfego benigno   │
│  • Detecção por erro de reconstrução     │
└─────────────────────────────────────────┘
```

### 4. Experimental Setup

#### 4.1 Datasets
- **NSL-KDD** (principal): 41 features, classes normal/ataque
- **CICIDS2017** (validação): 80+ features, classe BENIGN/ataques

#### 4.2 Baselines

| Baseline | Descrição | Motivação |
|----------|-----------|-----------|
| **PSO** | PSO clássico com inércia fixa | Comparar com meta-heurística tradicional |
| **GWO** | GWO puro (sem inércia do PSO) | Comparar com meta-heurística de hierarquia |
| **BPSO** | PSO binário com sigmoid | Comparar com discretização clássica |
| **No-Selection** | Todas as features, apenas hiperparâmetros | Mostrar ganho da seleção |
| **Proposta** | PSO-GWO híbrido | Nosso método |

#### 4.3 Métricas
- **Accuracy**, **Precision**, **Recall**, **F1-score**
- **TPR** (True Positive Rate), **FPR** (False Positive Rate)
- **Número de features selecionadas**
- **Tempo de convergência** (iterações até estabilizar)

#### 4.4 Configuração Experimental
- **30 execuções independentes** (seeds fixas para reprodutibilidade)
- **Enxame:** 15 partículas
- **Iterações:** 30
- **Teste estatístico:** Wilcoxon rank-sum (baseline vs proposta)
- **Significância:** α = 0.05

### 5. Results and Discussion

#### 5.1 Tabela Comparativa

| Método | Dataset | F1 | Precision | Recall | Features | TPR | FPR |
|--------|---------|----|-----------|--------|----------|-----|-----|
| No-Selection | NSL-KDD | - | - | - | 41 | - | - |
| PSO | NSL-KDD | - | - | - | - | - | - |
| GWO | NSL-KDD | - | - | - | - | - | - |
| BPSO | NSL-KDD | - | - | - | - | - | - |
| **Proposta** | NSL-KDD | - | - | - | - | - | - |

#### 5.2 Gráficos Esperados
- **Gráfico de convergência:** Fitness médio vs iteração (comparando PSO, GWO, Proposta)
- **Boxplot:** Distribuição do F1 nas 30 execuções
- **Boxplot:** Distribuição do número de features
- **Scatter plot:** F1 vs n_features (trade-off)

### 6. Conclusion
- Recapitular a contribuição: hibridização PSO-GWO em nível de equação
- Destacar os ganhos: melhor F1 com menos features
- Trabalhos futuros:
  - MOPSO real (fronteira de Pareto)
  - Outros datasets (UNSW-NB15, CSE-CIC-IDS2018)
  - Hibridização com outros algoritmos (ABC, FA)

---

## 3. Roadmap de Implementação

### Semana 1: Infrastructure (Reprodutibilidade)
- [ ] Criar `experiments/run_benchmark.py`
  - Script que roda N vezes cada configuração
  - Salva resultados em `experiments/results/<timestamp>/`
  - Formato: CSV com colunas [run, method, dataset, iteration, f1, precision, recall, accuracy, n_features, tpr, fpr, time]
- [ ] Implementar seeds fixas para reprodutibilidade
  ```python
  random.seed(42 + run_id)
  np.random.seed(42 + run_id)
  torch.manual_seed(42 + run_id)
  ```
- [ ] Criar `experiments/config.py` com todas as configurações centralizadas

### Semana 2: Baselines
- [ ] **Baseline 1: PSO puro**
  - Usar `optimizer_type = "pso"` no `main.py`
  - Garantir que funciona isolado
- [ ] **Baseline 2: GWO puro**
  - Implementar `chameleon/pso/gwo_pure.py`
  - GWO sem a parte do PSO (sem inércia, sem personal best)
  - Apenas hierarquia alpha/beta/delta
- [ ] **Baseline 3: BPSO**
  - Implementar `chameleon/pso/bpso.py`
  - Usar sigmoid sobre a velocidade para probabilidade de flip
  - Comparar com nosso threshold
- [ ] **Baseline 4: No-Selection**
  - Todas as features = 1
  - Apenas otimizar hiperparâmetros do ensemble

### Semana 3: Experimentos
- [ ] Rodar 30 execuções no NSL-KDD para cada baseline + proposta
- [ ] Rodar 30 execuções no CICIDS2017 para cada baseline + proposta
- [ ] Coletar todas as métricas
- [ ] Gerar gráficos:
  - `experiments/plots/convergence_comparison.png`
  - `experiments/plots/boxplot_f1.png`
  - `experiments/plots/boxplot_features.png`
  - `experiments/plots/f1_vs_features.png`

### Semana 4: Análise Estatística e Escrita
- [ ] Teste de Wilcoxon rank-sum (baseline vs proposta)
- [ ] Teste de Friedman + Nemenyi post-hoc (comparação múltipla)
- [ ] Gerar tabelas LaTeX com resultados
- [ ] Escrever o paper (LaTeX no Overleaf)
  - Template: IEEEtran ou Springer LNCS
- [ ] Revisão e submissão

---

## 4. Onde Publicar

### Opções por nível de competitividade

| Veículo | Tipo | Impacto | Requisitos |
|---------|------|---------|------------|
| **SBRC / ENIAC** | Evento nacional (Brasil) | Baixo | Em português, menos rigoroso |
| **SBSI** | Evento nacional | Baixo | Em português |
| **IEEE Access** | Journal Open Access | Médio | Em inglês, revisão rápida |
| **Applied Soft Computing** | Journal (Elsevier) | Alto | Em inglês, rigoroso |
| **Information Sciences** | Journal (Elsevier) | Alto | Em inglês, rigoroso |
| **Neurocomputing** | Journal (Elsevier) | Médio-Alto | Em inglês |
| **IEEE SMC** | Conference | Médio-Alto | Em inglês, deadline anual |
| **GECCO / CEC** | Conference | Alto | Em inglês, top tier |

### Recomendação
1. **Primeiro:** Submeter para **SBRC** ou **SBSI** (2025/2026)
   - Validação do trabalho em português
   - Feedback da comunidade brasileira
2. **Segundo:** Expandir para inglês e submeter para **IEEE Access** ou **Applied Soft Computing**

---

## 5. Arquivos Necessários

```
experiments/
├── run_benchmark.py          # Script principal de benchmark
├── config.py                 # Configurações centralizadas
├── results/
│   ├── 2024-01-15-pso/
│   ├── 2024-01-15-gwo/
│   ├── 2024-01-15-bpso/
│   ├── 2024-01-15-proposta/
│   └── 2024-01-15-no-selection/
├── plots/
│   ├── convergence_comparison.png
│   ├── boxplot_f1.png
│   ├── boxplot_features.png
│   └── f1_vs_features.png
└── statistical_tests.py      # Wilcoxon, Friedman

chameleon/pso/
├── gwo_pure.py               # GWO puro (baseline)
├── bpso.py                   # BPSO com sigmoid (baseline)
└── gwo_optimizer.py          # Já existe (nosso híbrido)

paper/
├── main.tex
├── figures/
└── tables/
```

---

## 6. Checklist Pré-Submissão

- [ ] 30 execuções com seeds para cada método
- [ ] Teste estatístico com p-values
- [ ] Gráficos gerados em alta resolução (300 DPI)
- [ ] Código disponível no GitHub (reprodutibilidade)
- [ ] README com instruções claras
- [ ] Requisitos.txt / requirements.txt atualizado
- [ ] Paper revisado por colega/orientador
- [ ] Inglês revisado (se for journal/conferência internacional)

---

*Documento criado em: 2024-01-15*
*Última atualização: 2024-01-15*
