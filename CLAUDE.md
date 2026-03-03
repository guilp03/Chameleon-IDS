# Chameleon-IDS

Sistema de Detecção de Intrusão (IDS) baseado em PSO para seleção de features + Autoencoder para detecção de anomalias.

## Visão Geral

Pipeline principal:
1. **PSO** (`chameleon/pso/optimizer.py`) busca o subconjunto ótimo de features e hiperparâmetros do ensemble usando F1-score como fitness
2. **Ensemble** (GradientBoosting/Random Forest) serve como função de fitness durante o PSO
3. **Autoencoder** (`chameleon/models/autoencoder.py`, PyTorch) treinado apenas no tráfego benigno; anomalias detectadas por erro de reconstrução (MSE > threshold)

## Estrutura de Arquivos

```
main.py                             # Ponto de entrada principal — NSL-KDD com PSO + Autoencoder
chameleon/
├── __init__.py
├── config.py                       # PSOConfig e ModelBounds (centraliza magic numbers)
├── pso/
│   ├── particle.py                 # Classe Particle (schema da partícula PSO)
│   └── optimizer.py               # Lógica PSO: search_space, evaluate_fitness, check_velocity, update_particle, update_pb
├── models/
│   ├── autoencoder.py             # Autoencoder PyTorch + EarlyStopping + get_overall_metrics
│   ├── gradient_boost.py          # Wrapper GradientBoostingClassifier (sklearn, não lib xgboost)
│   └── random_forest.py           # Wrapper RandomForestClassifier (sklearn)
└── data/
    └── preprocessing.py           # Pré-processamento: NSL-KDD (preprocess_nslkdd) e CICIDS2017 (preprocess_cicids)
tests/
├── test_preprocessing.py
├── test_pso.py
└── test_autoencoder.py
```

## Datasets Suportados

- **NSL-KDD** — `csv_result-KDDTrain+_20Percent.csv` (coluna label: `class`, valores: `normal` / ataques)
- **CICIDS2017** — coluna label: `Label`, valor benigno: `BENIGN`

Os datasets **não estão no repositório** (excedem o limite do GitHub). Ver README.md para links de download.

## Estrutura da Partícula (PSO)

```
position = [test_size, feat_0, feat_1, ..., feat_n, n_estimators, learning_rate]  # para "gb"
position = [test_size, feat_0, feat_1, ..., feat_n, n_estimators]                  # para "rf"
```

- `position[0]`: fração para split train/val (0.1–0.4)
- `position[1..n_features]`: seleção de features (0 ou 1)
- `position[-2]` (gb): n_estimators (50–1000)
- `position[-1]` (gb): learning_rate (0.1–0.3); (rf): n_estimators (50–600)

## Configuração (chameleon/config.py)

```python
@dataclass
class PSOConfig:
    swarm_size: int = 15
    max_iterations: int = 30
    inertia: float = 0.5
    cognitive_param: float = 1.0   # c1
    social_param: float = 2.0      # c2

@dataclass
class ModelBounds:
    test_size: tuple = (0.1, 0.4)
    n_estimators_gb: tuple = (50, 1000)
    n_estimators_rf: tuple = (50, 600)
    learning_rate: tuple = (0.1, 0.3)
```

## Como Executar

```bash
python3 main.py
```

O arquivo CSV do dataset deve estar no mesmo diretório. Ajuste o caminho em `csv_path` se necessário.

## Como Testar

```bash
python3 -m pytest tests/ -v
```

Requer `pytest` instalado: `pip install pytest`.

## Dependências

```
pandas, numpy, scikit-learn, torch, joblib, tqdm
```

## Notas Importantes

- `gradient_boost.py` usa `sklearn.GradientBoostingClassifier` — NÃO a lib `xgboost`
- O Autoencoder é treinado **somente em tráfego benigno** (classe 0) — detecção por anomalia
- `globalbest` considera tanto F1-score quanto número de features (menos features desempata)
- Aliases legados (`Search_Space`, `Evaluate_fitness`, `checkvelocity`, `inteiro`, `preprocessing`, `get_optimal_subesets`) existem nos módulos para compatibilidade; os nomes canônicos são os snake_case sem typos
