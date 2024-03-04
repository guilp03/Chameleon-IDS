import pandas as pd
import dataset
import pso
import time

funct = "gb"

df = pd.read_csv("csv_result-KDDTrain+_20Percent.csv")

columnsName = df.drop(labels= 'class', axis= 1).columns.values.tolist()

df = dataset.preprocessing(df)

SWARM_SIZE = 15
MAX_ITERATIONS = 30
C1 = 2
C2 = 2
INERTIA = 0.5
globalbest = []
globalbes_val = 0

start_time = time.time()
swarm = []
n_features = len(columnsName)

# Inicialização
for i in range(SWARM_SIZE):
    inicial_position = pso.Search_Space(funct)
    swarm.append(pso.Particle(i,inicial_position))
    particle_choices = dataset.particle_choices(swarm[i].position, columnsName, n_features)
    swarm[i].pb_val = pso.Evaluate_fitness(funct,particle_choices, df, swarm[i].number, n_features= n_features)
    swarm[i].pos_val = swarm[i].pb_val

# Iterative Search
for particle in swarm:
    if particle.pb_val > globalbes_val:
        globalbest = particle.position.copy()
        globalbes_val = particle.pb_val

