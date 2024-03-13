class Particle:
    velocity: list # Array que representa a velocidade de cada índice de position
    position: list # Configuração atual da partícula
    personal_best: list # Melhor configuração de partícula pessoal
    pb_val: float # Melhor f1_score pessoal
    pos_val: float # Novo melhor f1_score após novo cálculo de velocidade
    index: int # Identificador único da partícula

    def __init__(self, index: int, initial_position: list, funct: str, columnsName: list):
        self.velocity = [0] * (len(columnsName)+2) if funct == "rf" else [0] * (len(columnsName)+3)
        self.position = initial_position
        self.personal_best = initial_position
        self.pb_val = 0
        self.pos_val = 0
        self.index = index

class Particle2:
    velocity: list # Array que representa a velocidade de cada índice de position
    position: list # Configuração atual da partícula
    personal_best: list # Melhor configuração de partícula pessoal
    pb_val: float # Melhor f1_score pessoal
    pos_val: float # Novo melhor f1_score após novo cálculo de velocidade
    index: int # Identificador único da partícula

    def __init__(self, index: int, initial_position: list, funct: str):
        self.velocity = [0] * 49 if funct == "rf" else [0] * 50
        self.position = initial_position
        self.personal_best = initial_position
        self.pb_val = 0
        self.pos_val = 0
        self.index = index