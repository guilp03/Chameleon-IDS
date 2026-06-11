"""Configurações centralizadas do Chameleon-IDS.
"""
from dataclasses import dataclass, field


@dataclass
class PSOConfig:
    """Hiperparâmetros do algoritmo PSO (Particle Swarm Optimization).

    Attributes:
        swarm_size: Número de partículas no enxame.
        max_iterations: Número máximo de iterações do PSO.
        inertia: Peso de inércia (w) — controla a influência da velocidade anterior.
        cognitive_param: Coeficiente cognitivo (c1) — atração pelo melhor pessoal.
        social_param: Coeficiente social (c2) — atração pelo melhor global.
    """
    swarm_size: int = 15
    max_iterations: int = 30
    inertia: float = 0.5
    cognitive_param: float = 1.0
    social_param: float = 2.0


@dataclass
class ModelBounds:
    """Intervalos válidos para os hiperparâmetros do ensemble.

    Usados pelo PSO para inicializar e clipar posições de partículas.

    Attributes:
        test_size: Fração mínima/máxima para split train/val.
        n_estimators_gb: Intervalo de n_estimators para GradientBoosting.
        n_estimators_rf: Intervalo de n_estimators para RandomForest.
        learning_rate: Intervalo de learning_rate para GradientBoosting.
    """
    test_size: tuple = field(default=(0.1, 0.4))
    n_estimators_gb: tuple = field(default=(50, 1000))
    n_estimators_rf: tuple = field(default=(50, 600))
    learning_rate: tuple = field(default=(0.1, 0.3))
