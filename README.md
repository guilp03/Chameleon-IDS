# Chameleon-IDS

> Sistema de Detecção de Intrusão baseado em **PSO** para seleção automática de features + **Autoencoder** para detecção de anomalias de rede.

---

## Como funciona

O Chameleon-IDS combina dois estágios de inteligência para detectar tráfego malicioso:

```
Dataset NSL-KDD
      │
      ▼
┌─────────────────────────────────────────┐
│              PSO (Enxame)               │
│                                         │
│  Cada partícula codifica:               │
│  • Quais features usar                  │
│  • Hiperparâmetros do ensemble          │
│                                         │
│  Fitness = F1-score do ensemble         │
│  (GradientBoosting ou RandomForest)     │
└────────────────┬────────────────────────┘
                 │  solução ótima
                 ▼
┌─────────────────────────────────────────┐
│           Autoencoder (PyTorch)         │
│                                         │
│  • Treinado APENAS em tráfego benigno   │
│  • Detecta anomalias por erro de        │
│    reconstrução (MSE > threshold)       │
└─────────────────────────────────────────┘
```

O PSO encontra o subconjunto de features e os hiperparâmetros que maximizam o F1-score do ensemble. O Autoencoder, treinado nessa seleção ótima, aprende o padrão do tráfego normal e rejeita o que foge dele.

---

## Requisitos

- Python 3.10+
- As dependências listadas no arquivo requirements.txt

```bash
pip install -r requirements.txt
```

---

## Instalação

```bash
git clone https://github.com/guilp03/Chameleon-IDS.git
cd Chameleon-IDS
pip install -r requirements.txt
```

---

## Datasets

Os datasets não estão incluídos no repositório por excederem o limite de tamanho do GitHub. Baixe e coloque o arquivo CSV na raiz do projeto.

| Dataset | Link | Arquivo esperado |
|---------|------|-----------------|
| **NSL-KDD** *(principal)* | [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd) | `csv_result-KDDTrain+_20Percent.csv` |
| CICIDS2017 | [Kaggle](https://www.kaggle.com/datasets/cicdataset/cicids2017) | — |
| UNSW-NB15 | [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/code) | — |

---

## Executando

```bash
python3 main.py
```

O caminho do CSV é configurado na variável `csv_path` dentro de `main.py`. Os hiperparâmetros do PSO ficam em `chameleon/config.py`:

```python
@dataclass
class PSOConfig:
    swarm_size: int = 15       # tamanho do enxame
    max_iterations: int = 30   # iterações do PSO
    inertia: float = 0.5
    cognitive_param: float = 1.0
    social_param: float = 2.0
```

Para alternar entre os ensembles:

```python
# em main.py
ensemble_type = "gb"   # GradientBoosting (padrão)
ensemble_type = "rf"   # RandomForest
```

---

## Testes

```bash
python3 -m pytest tests/ -v
```

Os testes cobrem pré-processamento, lógica do PSO e o Autoencoder — sem necessidade de dataset.

---

## Estrutura do projeto

```
Chameleon-IDS/
├── main.py                       # Ponto de entrada
├── chameleon/
│   ├── config.py                 # Hiperparâmetros centralizados
│   ├── pso/
│   │   ├── particle.py           # Representação de uma partícula
│   │   └── optimizer.py          # Lógica do PSO
│   ├── models/
│   │   ├── autoencoder.py        # Autoencoder + EarlyStopping
│   │   ├── gradient_boost.py     # Wrapper GradientBoostingClassifier
│   │   └── random_forest.py      # Wrapper RandomForestClassifier
│   └── data/
│       └── preprocessing.py      # Limpeza e normalização dos datasets
└── tests/
    ├── test_preprocessing.py
    ├── test_pso.py
    └── test_autoencoder.py
```

---

## Referências

- Kennedy, J. & Eberhart, R. (1995). *Particle Swarm Optimization*. ICNN.
- Hinton, G. & Salakhutdinov, R. (2006). *Reducing the Dimensionality of Data with Neural Networks*. Science.
- Dataset NSL-KDD: Tavallaee et al. (2009). *A Detailed Analysis of the KDD CUP 99 Data Set*. IEEE CISDA.
