import pandas as pd
import dataset
import ParticleSwarmOptimization as pso
import particle_schema as part
import time
import torch
import os
import Autoencoder as ae
from joblib import Parallel, delayed

funct = "gb"
df = pd.read_csv('csv_result-KDDTrain+_20Percent.csv')
#df_list = []
#for file in os.listdir('MachineLearningCVE'):
#  df_aux = pd.read_csv(f'MachineLearningCVE/{file}')
#  df_list.append(df_aux)
#df = pd.concat(df_list, ignore_index=True)

#df_list = []
#df_benign = df.query(' Label == "BENIGN"').sample(frac=0.02, random_state=42)
#df_list.append(df_benign)

#df_anomaly = df.query(' Label != "BENIGN"').sample(frac=0.02, random_state=42)
#df_list.append(df_anomaly)

#df = pd.concat(df_list, ignore_index=True)

df.columns = df.columns.str.replace("'", "")
df = df.drop(label = 'id', axis = 1)
df, columnsName, y = dataset.preprocessing(df)
SWARM_SIZE = 15
MAX_ITERATIONS = 5
component_1 = 2
component_2 = 2
INERTIA = 0.5
globalbest = []
globalbest_val = 0

start_time = time.time()
swarm = []

def process_particle(i, funct, columnsName, df, y):
    inicial_position = pso.Search_Space(funct, n_features=len(columnsName))
    particle = part.Particle(i, inicial_position, funct=funct)
    particle.pb_val = pso.Evaluate_fitness(funct, particle, columnsName, df, y,particle.index, n_features=len(columnsName))
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
    particle.position = pso.update_particle(particle, funct, n_features=len(columnsName))
    particle.pos_val = pso.Evaluate_fitness(funct,particle, columnsName, df, y,particle.index, n_features=len(columnsName))
    particle = pso.update_pb(particle) # Atualiza valor de personal best (ignorar return)
    
    return particle
    
itter = 30
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

optimal_x_train, optimal_x_val, optimal_x_test, optimal_y_train, optimal_y_val, optimal_y_test = dataset.get_optimal_subesets(df, optimal_solution, columnsName, y, optimal_solution[0], n_features=len(columnsName))   
     
end_time = time.time()
dataset.get_time(start_time, end_time)
print(dataset.particle_choices(globalbest, columnsName, n_features=len(columnsName)))
print(len(dataset.particle_choices(globalbest,columnsName, n_features=len(columnsName))))
swarm[0].pos = globalbest

print(pso.Evaluate_fitness(funct,swarm[0], columnsName, df, y,swarm[0].index, n_features=len(columnsName)))
optimal_x_train['class'] = optimal_y_train
optimal_x_train = optimal_x_train.query('`class` == 0')
optimal_x_train = optimal_x_train.drop(labels = 'class', axis = 1)
optimal_x_val['class'] = optimal_y_val
benign_x_val_optimal = optimal_x_val[optimal_x_val['class']== 1]
benign_x_val_optimal = benign_x_val_optimal.drop(labels = 'class', axis = 1)
optimal_x_val = optimal_x_val.drop(labels = 'class', axis = 1)

optimal_x_train, optimal_x_val, optimal_x_test, benign_x_val_optimal = dataset.transform_MinMaxScaler(optimal_x_train, optimal_x_val, optimal_x_test,benign_x_val_optimal)

benign_x_val_optimal_tensor = torch.FloatTensor(benign_x_val_optimal)
optimal_x_train_tensor = torch.FloatTensor(optimal_x_train)

#BATCH_SIZE = 32
#ALPHA = 5e-4
#PATIENCE = 7
#DELTA = 0.001
#NUM_EPOCHS = 500
#IN_FEATURES = optimal_x_train.shape[1]
#DROPOUT_RATE = 0.5
#REGULARIZER = 0.001

BATCH_SIZE = 16
ALPHA = 0.096
PATIENCE = 10
DELTA = 0.0001
NUM_EPOCHS = 1000
IN_FEATURES = optimal_x_train.shape[1]
DROPOUT_RATE = 0.5
REGULARIZER = 0.00075

start_time = time.time()
ae_model = ae.Autoencoder(IN_FEATURES, DROPOUT_RATE)
ae_model.compile(learning_rate= ALPHA, weight_decay=REGULARIZER)
train_avg_losses, val_avg_losses = ae_model.fit(optimal_x_train_tensor,
                                                NUM_EPOCHS,
                                                BATCH_SIZE,
                                                X_val = benign_x_val_optimal_tensor,
                                                patience = PATIENCE,
                                                delta = DELTA)

end_time = time.time()
dataset.get_time(start_time, end_time)

#ae.plot_train_val_losses(train_avg_losses, val_avg_losses)
val_anomaly_scores = ae.get_autoencoder_anomaly_scores(ae_model, optimal_x_val)

lista_arrays = [float(i) for i in range(1, 91)]  # Gerar números inteiros de 1 a 90
lista_arrays = [x / 1000 for x in lista_arrays]
lista_f1 =[]
best_f1 = 0.0
best_thresh = 0.0
for i in lista_arrays:
    metrics = ae.get_overall_metrics(optimal_y_val, val_anomaly_scores > i)
    f1_score = metrics['f1-score']
    lista_f1.append(f1_score)
    if f1_score > best_f1:
        best_f1 = f1_score
        precision = metrics['precision']
        tpr = metrics['tpr']
        accuracy = metrics['acc']
        fpr = metrics['fpr']
        best_thresh = i

print("Melhor Threshold:", best_thresh)
print("Melhor F1-score:", best_f1)

print("metricas:", "f1_score:", best_f1, "precision:", precision, "accuracy:", accuracy, "tpr:", tpr, "fpr:", fpr)