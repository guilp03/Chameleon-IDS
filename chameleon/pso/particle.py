"""Schema da partícula usada pelo algoritmo PSO."""


class Particle:
    """Representa uma partícula no espaço de busca do PSO.

    Cada partícula codifica uma configuração candidata: fração de split
    treino/validação, máscara binária de seleção de features e
    hiperparâmetros do ensemble (n_estimators e, opcionalmente, learning_rate).

    Estrutura de ``position``:
        - ``position[0]``: test_size (float, 0.1–0.4)
        - ``position[1..n_features]``: seleção de features (0 ou 1)
        - ``position[-2]`` (gb): n_estimators (int, 50–1000)
        - ``position[-1]`` (gb): learning_rate (float, 0.1–0.3)
        - ``position[-1]`` (rf): n_estimators (int, 50–600)

    Attributes:
        velocity: Vetor de velocidade de cada dimensão de ``position``.
        position: Configuração atual da partícula.
        personal_best: Melhor configuração pessoal encontrada.
        pb_val: F1-score da melhor configuração pessoal.
        pos_val: F1-score da configuração atual (após atualização de velocidade).
        index: Identificador único da partícula no enxame.
        pb_feat_number: Número de features selecionadas em ``personal_best``.
    """

    velocity: list
    position: list
    personal_best: list
    pb_val: float
    pos_val: float
    index: int
    pb_feat_number: int

    def __init__(
        self,
        index: int,
        initial_position: list,
        ensemble_type: str,
        column_names: list,
    ) -> None:
        """Inicializa a partícula.

        Args:
            index: Identificador único da partícula.
            initial_position: Posição inicial gerada por ``Search_Space``.
            ensemble_type: ``"rf"`` para RandomForest ou ``"gb"`` para GradientBoosting.
            column_names: Lista de nomes de colunas do dataset.
        """
        velocity_size = len(column_names) + 2 if ensemble_type == "rf" else len(column_names) + 3
        self.velocity = [0] * velocity_size
        self.position = initial_position
        self.personal_best = initial_position
        self.pb_val = 0
        self.pos_val = 0
        self.pb_feat_number = 0
        self.index = index
