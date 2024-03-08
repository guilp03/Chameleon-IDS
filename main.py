import pandas as pd
import dataset
import ParticleSwarmOptimization as pso
import time
import torch
import Autoencoder as ae
from joblib import Parallel, delayed

funct = "gb"
df = pd.read_csv("csv_result-KDDTrain+_20Percent.csv")
df.columns = df.columns.str.replace("'", "")
columnsName = df.drop(labels= 'class', axis= 1).columns.values.tolist()
x_train, y_train, x_val, y_val, x_test, y_test = dataset.preprocessing(df)
SWARM_SIZE = 15
MAX_ITERATIONS = 30
C1 = 2
C2 = 2
INERTIA = 0.5
globalbest = []
globalbest_val = 0

start_time = time.time()
swarm = []

def process_particle(i, funct, columnsName, x_train, y_train, x_val, y_val, x_test, y_test):
    inicial_position = pso.Search_Space(funct)
    particle = pso.Particle(i, inicial_position, funct=funct)
    particle.pb_val = pso.Evaluate_fitness(funct, particle, columnsName, x_train, y_train, x_val, y_val, x_test, y_test, particle.number, n_features=42)
    print(particle.pb_val)
    particle.pos_val = particle.pb_val
    return particle

swarm.extend(Parallel(n_jobs=-1)(delayed(process_particle)(i, funct, columnsName, x_train, y_train, x_val, y_val, x_test, y_test) for i in range(SWARM_SIZE)))

globalbest_val = max(p.pb_val for p in swarm)
globalbest = max(swarm, key=lambda p: p.pb_val).position

def apply_pso(funct, particle, x_train, y_train, x_val, y_val, x_test, y_test):
    particle.velocity = pso.checkvelocity(globalbest=globalbest, particle=particle, inertia=INERTIA)
    particle.position = pso.update_particle(particle, funct)
    particle.pos_val = pso.Evaluate_fitness(funct,particle, columnsName ,x_train, y_train, x_val, y_val, x_test, y_test, particle.number, n_features=42)
    particle = pso.update_pb(particle) # Atualiza valor de personal best (ignorar return)
    print(particle.pb_val)
    return particle
    
itter = 5
for i in range(itter):
    print("Iteração:", i)
    #for particle in swarm:
    #    particle.velocity = pso.checkvelocity(globalbest=globalbest, particle=particle, inertia=INERTIA)
    #    particle.position = pso.update_particle(particle, funct)
    #    particle.pb_val = pso.Evaluate_fitness(funct,particle, x_train, y_train, x_val, y_val, x_test, y_test, particle.number, n_features=42)
    #    particle = pso.update_pb(particle) # Atualiza valor de personal best (ignorar return)
    swarm = Parallel(n_jobs=-1)(delayed(apply_pso)(funct, particle, x_train, y_train, x_val, y_val, x_test, y_test) for particle in swarm)
    print(swarm[0].pb_val)
    print(swarm[1].pb_val)
    print(swarm[2].pb_val)


    if max(p.pb_val for p in swarm) > globalbest_val:
        globalbest_val = max(p.pb_val for p in swarm)
        globalbest = max(swarm, key=lambda p: p.pb_val).position
    print(globalbest)
    print(globalbest_val)
optimal_solution = globalbest   
optimal_x_train, optimal_x_val, optimal_x_test = dataset.get_optimal_subesets(x_train, x_val, x_test, optimal_solution, columnsName)   
     
end_time = time.time()
dataset.get_time(start_time, end_time)
print(dataset.particle_choices(globalbest))
print(len(dataset.particle_choices(globalbest)))

benign_x_val_optimal = optimal_x_val[y_val == 1]
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



