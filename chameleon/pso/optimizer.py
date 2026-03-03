"""Lógica do algoritmo PSO (Particle Swarm Optimization).

Funções para inicializar o espaço de busca, avaliar fitness, atualizar
velocidade/posição e gerenciar o personal best de cada partícula.
"""
import random

import numpy as np
import pandas as pd

from chameleon.config import ModelBounds
from chameleon.data.preprocessing import particle_choices, split_train_test
from chameleon.models.gradient_boost import GradientBoost, get_metrics as gb_get_metrics
from chameleon.models.random_forest import RandomForest, get_metrics as rf_get_metrics
from chameleon.pso.particle import Particle

_bounds = ModelBounds()


def search_space(ensemble_type: str, n_features: int) -> list:
    """Define a posição inicial aleatória de uma partícula.

    Args:
        ensemble_type: ``"rf"`` para RandomForest ou ``"gb"`` para GradientBoosting.
        n_features: Número de features disponíveis no dataset.

    Returns:
        Posição inicial da forma ``[test_size, feat_0, ..., feat_n, hiperparâmetros]``.
    """
    initial_position: list = []
    initial_position.append(random.uniform(*_bounds.test_size))
    for _ in range(n_features):
        initial_position.append(random.choice((0, 1)))
    if ensemble_type == "gb":
        initial_position.append(random.randint(*_bounds.n_estimators_gb))
        initial_position.append(random.uniform(*_bounds.learning_rate))
    if ensemble_type == "rf":
        initial_position.append(random.randint(*_bounds.n_estimators_rf))
    return initial_position


# Alias em maiúsculas mantido para compatibilidade com chamadas legadas (remover quando possível)
Search_Space = search_space


def evaluate_fitness(
    ensemble_type: str,
    particle: Particle,
    column_names: list[str],
    df: pd.DataFrame,
    y: pd.Series,
    i: int,
    n_features: int | None = None,
) -> float:
    """Calcula o F1-score de uma partícula usando o ensemble especificado.

    Args:
        ensemble_type: ``"rf"`` para RandomForest ou ``"gb"`` para GradientBoosting.
        particle: Partícula cujo fitness será calculado.
        column_names: Lista de colunas disponíveis no dataset.
        df: Base de dados completa (normalizada).
        y: Labels do dataset.
        i: Índice da partícula (usado no log).
        n_features: Número de features do dataset (inferido de ``column_names`` se omitido).

    Returns:
        F1-score arredondado em 4 casas decimais.
    """
    if n_features is None:
        n_features = len(column_names)

    selected_columns = particle_choices(particle.position, column_names, n_features)
    x_train, y_train, x_val, y_val = split_train_test(df, column_names, y, particle.position[0])
    x_train_sel = x_train[selected_columns]
    x_val_sel = x_val[selected_columns]

    if ensemble_type == "gb":
        model = GradientBoost(x_train_sel, y_train, particle.position[-2], particle.position[-1])
        accuracy, f1, precision, recall = gb_get_metrics(model, x_val_sel, y_val)
        f1 = round(f1, 4)
        print(i, "accuracy:", accuracy, "f1_score:", f1, "precision:", precision, "recall:", recall)
        return f1
    else:
        model = RandomForest(n_features, x_train_sel, y_train, particle.position[-1])
        accuracy, f1, precision, recall = rf_get_metrics(model, x_val_sel, y_val)
        f1 = round(f1, 4)
        print(i, "accuracy:", accuracy, "f1_score:", f1, "precision:", precision, "recall:", recall)
        return f1


# Alias mantido para compatibilidade
Evaluate_fitness = evaluate_fitness


def check_velocity(
    globalbest: list,
    particle: Particle,
    inertia: float,
    c1: float,
    c2: float,
) -> list:
    """Calcula o vetor de velocidade de uma partícula segundo a equação do PSO.

    v(t+1) = w*v(t) + c1*r1*(pb - x) + c2*r2*(gb - x)

    Args:
        globalbest: Melhor posição global encontrada no enxame.
        particle: Partícula cujo vetor de velocidade será atualizado.
        inertia: Peso de inércia (w).
        c1: Coeficiente cognitivo.
        c2: Coeficiente social.

    Returns:
        Novo vetor de velocidade com o mesmo comprimento de ``particle.position``.
    """
    velocity = list(
        np.array(particle.velocity) * inertia
        + c1 * random.random() * (np.array(particle.personal_best) - np.array(particle.position))
        + c2 * random.random() * (np.array(globalbest) - np.array(particle.position))
    )
    return velocity


# Alias mantido para compatibilidade
checkvelocity = check_velocity


def clip_to_bounds(particle: Particle, ensemble_type: str, n_features: int) -> list:
    """Clipa cada dimensão de ``particle.position`` para seu intervalo válido.

    Garante que:
    - ``position[0]`` (test_size) ∈ [0.1, 0.4]
    - ``position[1..n_features]`` (features) ∈ {0, 1}
    - ``position[-1]`` (rf: n_estimators) ∈ [50, 600]
    - ``position[-2]`` (gb: n_estimators) ∈ [50, 1000]
    - ``position[-1]`` (gb: learning_rate) ∈ [0.1, 0.3]

    Args:
        particle: Partícula a ser clipada (modificada in-place).
        ensemble_type: ``"rf"`` ou ``"gb"``.
        n_features: Número de features do dataset.

    Returns:
        ``particle.position`` com os valores clipados.
    """
    particle.position[0] = np.clip(particle.position[0], *_bounds.test_size)

    if ensemble_type == "rf":
        particle.position[-1] = int(np.clip(particle.position[-1], *_bounds.n_estimators_rf))
    if ensemble_type == "gb":
        particle.position[-2] = int(np.clip(particle.position[-2], *_bounds.n_estimators_gb))
        particle.position[-1] = round(float(np.clip(particle.position[-1], *_bounds.learning_rate)), 4)

    for m in range(1, n_features + 1):
        particle.position[m] = 1 if particle.position[m] > 0.5 else 0

    return particle.position


# Alias mantido para compatibilidade
inteiro = clip_to_bounds


def update_particle(particle: Particle, ensemble_type: str, n_features: int) -> list:
    """Atualiza ``particle.position`` somando a velocidade e clipando os limites.

    Args:
        particle: Partícula a ser atualizada.
        ensemble_type: ``"rf"`` ou ``"gb"``.
        n_features: Número de features do dataset.

    Returns:
        Nova posição clipada da partícula.
    """
    for i in range(len(particle.position)):
        particle.position[i] += particle.velocity[i]
    particle.position = clip_to_bounds(particle, ensemble_type, n_features)
    return particle.position


def update_pb(particle: Particle) -> Particle:
    """Atualiza o personal best se a posição atual for melhor.

    Args:
        particle: Partícula a ser avaliada.

    Returns:
        A mesma partícula, com ``pb_val`` e ``personal_best`` atualizados se necessário.
    """
    if particle.pb_val < particle.pos_val:
        particle.pb_val = particle.pos_val
        particle.personal_best = particle.position
    return particle
