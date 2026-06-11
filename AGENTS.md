# Chameleon-IDS - Contexto para Agentes

> **Branch atual:** `dev`  
> **Status:** GWO implementado na estrutura refatorada. Paper plan criado.  
> **Próximo passo:** Implementar baselines e experimentos para o paper.

---

## O que é este projeto

Sistema de Detecção de Intrusão (IDS) que combina:
1. **PSO-GWO híbrido** para seleção automática de features + hiperparâmetros do ensemble
2. **Ensemble** (GradientBoosting/RandomForest) como função de fitness
3. **Autoencoder** (PyTorch) treinado apenas em tráfego benigno para detecção de anomalias

**O diferencial:** O PSO-GWO é uma hibridização em nível de equação de velocidade — não uma cascata de dois algoritmos. A velocidade do PSO inclui a influência dos 3 lobos líderes do GWO (alpha, beta, delta).

---

## Estrutura de Arquivos

```
Chameleon-IDS/
├── main.py                          # Ponto de entrada (suporta PSO e GWO)
├── paper-plan.md                    # Plano completo para paper
├── chameleon/
│   ├── config.py                    # PSOConfig, ModelBounds
│   ├── pso/
│   │   ├── optimizer.py            # PSO clássico (search_space, evaluate_fitness, check_velocity, update_particle, update_pb)
│   │   ├── gwo_optimizer.py        # PSO-GWO híbrido (convergence_factors, calculate_distances, leadership, check_velocity, find_leaders)
│   │   └── particle.py             # Classe Particle (position, velocity, personal_best, pb_val, etc.)
│   ├── models/
│   │   ├── autoencoder.py          # Autoencoder PyTorch + EarlyStopping + get_overall_metrics
│   │   ├── gradient_boost.py       # Wrapper sklearn GradientBoostingClassifier
│   │   └── random_forest.py        # Wrapper sklearn RandomForestClassifier
│   └── data/
│       └── preprocessing.py        # preprocess_nslkdd, preprocess_cicids, particle_choices, get_optimal_subsets, transform_min_max_scaler
└── tests/
    ├── test_preprocessing.py
    ├── test_pso.py
    └── test_autoencoder.py

# Arquivos legados (não usar, manter para referência)
├── newPSO.py                        # Código antigo da apso com GWO
├── main_newPSO.py                   # Main antigo da apso
```

---

## Convenções de Código

### Estilo
- **Snake_case** para tudo (funções, variáveis, arquivos)
- **Docstrings** em todos os módulos e funções públicas (Google style)
- **Type hints** obrigatórios
- **Sem typos** nos nomes canônicos (ex: `search_space`, não `Search_Space`)
- Aliases legados existem para compatibilidade, mas os nomes canônicos são os snake_case corretos

### PSOConfig (chameleon/config.py)
```python
@dataclass
class PSOConfig:
    swarm_size: int = 15
    max_iterations: int = 30
    inertia: float = 0.5
    cognitive_param: float = 1.0   # c1
    social_param: float = 2.0      # c2
```

### ModelBounds
```python
@dataclass
class ModelBounds:
    test_size: tuple = (0.1, 0.4)
    n_estimators_gb: tuple = (50, 1000)
    n_estimators_rf: tuple = (50, 600)
    learning_rate: tuple = (0.1, 0.3)
```

### Estrutura da Partícula
```
position = [test_size, feat_0, feat_1, ..., feat_n, n_estimators, learning_rate]  # gb
position = [test_size, feat_0, feat_1, ..., feat_n, n_estimators]                  # rf
```

---

## Como usar o GWO

No `main.py`:
```python
optimizer_type = "gwo"  # "pso" = clássico, "gwo" = PSO-GWO híbrido
```

O GWO usa:
- `gwo_optimizer.check_velocity()` — velocidade com alpha/beta/delta
- `gwo_optimizer.find_leaders()` — encontra os 3 líderes
- `c = (m/MAX_ITERATIONS)**(2/3) + 1` — fator de convergência adaptativo
- `m` — contador de estagnação (reseta se globalbest melhorar)

---

## O que está planejado (Paper)

Documentado em `paper-plan.md`. Resumo:

### Baselines a implementar
1. **PSO puro** — já existe (`optimizer_type = "pso"`)
2. **GWO puro** — precisa criar `chameleon/pso/gwo_pure.py`
3. **BPSO** — precisa criar `chameleon/pso/bpso.py` (sigmoid)
4. **No-Selection** — todas as features = 1

### Experimentos
- 30 execuções com seeds diferentes
- Datasets: NSL-KDD, CICIDS2017
- Métricas: F1, precision, recall, accuracy, TPR, FPR, n_features, tempo
- Testes estatísticos: Wilcoxon rank-sum

### Scripts a criar
```
experiments/
├── run_benchmark.py          # Roda 30x cada config
├── config.py                 # Configs centralizadas
├── results/                  # CSVs com resultados
├── plots/                    # Gráficos gerados
└── statistical_tests.py      # Wilcoxon, Friedman
```

---

## Dependências

```
pandas, numpy, scikit-learn, torch, joblib, tqdm, pytest
```

---

## Datasets

- **NSL-KDD**: `csv_result-KDDTrain+_20Percent.csv` (não está no repo)
- **CICIDS2017**: `CICIDS2017.csv` (não está no repo)
- Os datasets NÃO estão no repositório (excedem limite do GitHub)

---

## Notas Importantes

- `gradient_boost.py` usa `sklearn.ensemble.GradientBoostingClassifier` — NÃO é xgboost
- Autoencoder é treinado **somente em tráfego benigno** (classe 0)
- `globalbest` considera tanto F1 quanto número de features (menos features desempata)
- O GWO foi cherry-picked do commit `8c5eb99` da branch `apso`
- A branch `dev` foi criada a partir de `main` + merge de `refactor`

---

## Checklist para Próxima Sessão

- [ ] Implementar `chameleon/pso/gwo_pure.py` (GWO puro, sem PSO)
- [ ] Implementar `chameleon/pso/bpso.py` (BPSO com sigmoid)
- [ ] Criar `experiments/run_benchmark.py` (30 runs com seeds)
- [ ] Criar `experiments/config.py` (configs centralizadas)
- [ ] Testar se `main.py` funciona com `optimizer_type = "gwo"`
- [ ] Rodar baseline PSO vs GWO vs Proposta (1 execução rápida)
- [ ] Atualizar `paper-plan.md` se necessário

---

## Contato / Autores

- Guilherme Pereira (guilp03)
- João Regis (Jvcregis)

---

*Última atualização: 2024-01-15*
*Branch: dev*
*Commit: 084af50*
