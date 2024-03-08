import pandas as pd
import dataset
import ParticleSwarmOptimization as pso
import particle_schema as part
import time
import torch
import Autoencoder as ae
from joblib import Parallel, delayed

funct = "gb"
df = pd.read_csv("csv_result-KDDTrain+_20Percent.csv")
df.columns = df.columns.str.replace("'", "")
df, columnsName, y = dataset.preprocessing(df)
SWARM_SIZE = 15
MAX_ITERATIONS = 30
component_1 = 2
component_2 = 2
INERTIA = 0.5
globalbest = []
globalbest_val = 0

start_time = time.time()
swarm = []

def process_particle(i, funct, columnsName, df, y):
    inicial_position = pso.Search_Space(funct, n_features=42)
    particle = part.Particle(i, inicial_position, funct=funct)
    particle.pb_val = pso.Evaluate_fitness(funct, particle, columnsName, df, y,particle.index, n_features=42)
    particle.pos_val = particle.pb_val
    
    return particle

swarm.extend(Parallel(n_jobs=-1)(delayed(process_particle)(i, funct, columnsName, df,y) for i in range(SWARM_SIZE)))
print(swarm[0].position)
print(swarm[0].personal_best)

globalbest_val = max(p.pb_val for p in swarm)
globalbest = max(swarm, key=lambda p: p.pb_val).position
print(globalbest)


def apply_pso(funct, particle, df, y):
    particle.velocity = pso.checkvelocity(globalbest=globalbest, particle=particle, inertia=INERTIA, c1 = component_1, c2 = component_2 )
    particle.position = pso.update_particle(particle, funct, n_features=42)
    particle.pos_val = pso.Evaluate_fitness(funct,particle, columnsName, df, y,particle.index, n_features=42)
    particle = pso.update_pb(particle) # Atualiza valor de personal best (ignorar return)
    
    return particle
    
itter = 5
for i in range(itter):
    print("Iteração:", i)
    #for particle in swarm:
    #    particle.velocity = pso.checkvelocity(globalbest=globalbest, particle=particle, inertia=INERTIA, c1=component_1, c2=component_2)
    #    particle.position = pso.update_particle(particle, funct, n_features=42)
    #    particle.pb_val = pso.Evaluate_fitness(funct,particle,columnsName ,df, y, particle.index, n_features=42)
    #    particle = pso.update_pb(particle) # Atualiza valor de personal best (ignorar return)
    swarm = Parallel(n_jobs=-1)(delayed(apply_pso)(funct, particle, df, y) for particle in swarm)
    print(swarm[0].position)
    print(swarm[0].personal_best)
    print(swarm[0].pb_val)


    if max(p.pb_val for p in swarm) > globalbest_val:
        globalbest_val = max(p.pb_val for p in swarm)
        globalbest = max(swarm, key=lambda p: p.pb_val).position
    print(globalbest)

optimal_solution = globalbest

optimal_x_train, optimal_x_val, optimal_x_test, optimal_y_train, optimal_y_val, optimal_y_test = dataset.get_optimal_subesets(df, optimal_solution, columnsName, y, optimal_solution[0], n_features=42)   
     
end_time = time.time()
dataset.get_time(start_time, end_time)
print(dataset.particle_choices(globalbest, columnsName, n_features=42))
print(len(dataset.particle_choices(globalbest,columnsName, n_features=42)))

benign_x_val_optimal = optimal_x_val[optimal_y_test == 1]
benign_x_val_optimal = torch.FloatTensor(benign_x_val_optimal)
optimal_x_train_tensor = torch.FloatTensor(optimal_x_train)

BATCH_SIZE = 32
ALPHA = 5e-4
PATIENCE = 7
DELTA = 0.001
NUM_EPOCHS = 1000
IN_FEATURES = optimal_x_train.shape[1]

start_time = time.time()
ae_model = ae.Autoencoder(IN_FEATURES)
ae_model.compile(learning_rate= ALPHA)
train_avg_losses, val_avg_losses = ae_model.fit(optimal_x_train_tensor,
                                                NUM_EPOCHS,
                                                BATCH_SIZE,
                                                X_val = benign_x_val_optimal,
                                                patience = PATIENCE,
                                                delta = DELTA)

end_time = time.time()
dataset.get_time(start_time, end_time)



