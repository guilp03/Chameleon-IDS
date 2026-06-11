"""Lógica do PSO híbrido com GWO (Grey Wolf Optimizer).

Adaptação que combina a inércia do PSO com a hierarquia de líderes
(alpha, beta, delta) do GWO para atualização de velocidade.
"""
import random

import numpy as np

from chameleon.pso.particle import Particle


def convergence_factors(a: float, c: float) -> tuple[float, float, float]:
    """Calcula os fatores de convergência A1, A2, A3 do GWO.

    Args:
        a: Parâmetro de decaimento linear (2 → 0).
        c: Fator de convergência adaptativo.

    Returns:
        Tupla ``(a1, a2, a3)``.
    """
    r1 = random.random()
    r2 = random.random()
    r3 = random.random()
    a1 = (2**c) * a * r1 - a
    a2 = (2**c) * a * r2 - a
    a3 = (2**c) * a * r3 - a
    return a1, a2, a3


def calculate_distances(
    X_alpha: Particle,
    X_beta: Particle,
    X_delta: Particle,
    X_i: Particle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula as distâncias D_α, D_β, D_δ até os 3 lobos líderes.

    Args:
        X_alpha: Líder alpha (melhor solução).
        X_beta: Líder beta (segunda melhor).
        X_delta: Líder delta (terceira melhor).
        X_i: Partícula atual.

    Returns:
        Tupla ``(D_alpha, D_beta, D_delta)``.
    """
    c1 = random.uniform(0, 2)
    c2 = random.uniform(0, 2)
    c3 = random.uniform(0, 2)
    D_alpha = np.abs(c1 * (np.array(X_alpha.position) - np.array(X_i.position)))
    D_beta = np.abs(c2 * (np.array(X_beta.position) - np.array(X_i.position)))
    D_delta = np.abs(c3 * (np.array(X_delta.position) - np.array(X_i.position)))
    return D_alpha, D_beta, D_delta


def leadership(
    curr_particle: Particle,
    X_alpha: Particle,
    X_beta: Particle,
    X_delta: Particle,
    curr_iteration: int,
    num_iterations: int,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula as posições influenciadas pelos 3 lobos líderes.

    Args:
        curr_particle: Partícula atual.
        X_alpha: Líder alpha.
        X_beta: Líder beta.
        X_delta: Líder delta.
        curr_iteration: Iteração atual.
        num_iterations: Total de iterações.
        c: Fator de convergência.

    Returns:
        Tupla ``(x1, x2, x3)`` com as posições influenciadas.
    """
    a = 2 * (1 - curr_iteration / num_iterations)
    a1, a2, a3 = convergence_factors(a, c)
    D_alpha, D_beta, D_delta = calculate_distances(X_alpha, X_beta, X_delta, curr_particle)
    x1 = np.array(X_alpha.position) - a1 * D_alpha
    x2 = np.array(X_beta.position) - a2 * D_beta
    x3 = np.array(X_delta.position) - a3 * D_delta
    return x1, x2, x3


def check_velocity(
    globalbest: list,
    particle: Particle,
    inertia: float,
    c: float,
    X_alpha: Particle,
    X_beta: Particle,
    X_delta: Particle,
    curr_iteration: int,
    num_iterations: int,
) -> list:
    """Calcula a velocidade híbrida PSO + GWO.

    Combina:
    - Inércia do PSO clássico
    - Influência dos 3 lobos líderes do GWO (alpha, beta, delta)

    v(t+1) = w*v(t) + (c/2)*r*(x1 - x) + (c/3)*r*(x2 - x) + (c/4)*r*(x3 - x)

    Args:
        globalbest: Melhor posição global (mantido para compatibilidade, não usado).
        particle: Partícula atual.
        inertia: Peso de inércia (w).
        c: Fator de convergência do GWO.
        X_alpha: Líder alpha.
        X_beta: Líder beta.
        X_delta: Líder delta.
        curr_iteration: Iteração atual.
        num_iterations: Total de iterações.

    Returns:
        Novo vetor de velocidade.
    """
    inertia_array = np.array([inertia])
    x1, x2, x3 = leadership(particle, X_alpha, X_beta, X_delta, curr_iteration, num_iterations, c)
    r4 = random.random()

    velocity = (
        inertia_array * np.array(particle.velocity)
        + (c / 2) * r4 * (x1 - np.array(particle.position))
        + (c / 3) * r4 * (x2 - np.array(particle.position))
        + (c / 4) * r4 * (x3 - np.array(particle.position))
    )
    return list(velocity)


def find_leaders(
    swarm: list[Particle],
    globalbest_val: float,
    globalbest_feat_number: int,
) -> tuple[Particle, Particle, Particle]:
    """Encontra os 3 lobos líderes (alpha, beta, delta) do enxame.

    Args:
        swarm: Lista de partículas.
        globalbest_val: Valor de fitness do globalbest (para compatibilidade).
        globalbest_feat_number: Número de features do globalbest.

    Returns:
        Tupla ``(X_alpha, X_beta, X_delta)``.
    """
    # Ordena por fitness (descendente) e número de features (ascendente)
    sorted_swarm = sorted(
        swarm,
        key=lambda p: (p.pb_val, -p.pb_feat_number),
        reverse=True,
    )
    X_alpha = sorted_swarm[0]
    X_beta = sorted_swarm[1] if len(sorted_swarm) > 1 else X_alpha
    X_delta = sorted_swarm[2] if len(sorted_swarm) > 2 else X_beta
    return X_alpha, X_beta, X_delta


# Aliases para compatibilidade
Search_Space = None  # Não implementado aqui — usa optimizer.search_space
Evaluate_fitness = None  # Não implementado aqui — usa optimizer.evaluate_fitness
checkvelocity = check_velocity
