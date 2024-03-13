import pandas as pd
import UNSWNB15_dataset
import ParticleSwarmOptimization as pso
import particle_schema as part
import time
import torch
import Autoencoder_UNSWNB15 as ae
from joblib import Parallel, delayed
import random
import dataset
import XGBoost as gb
#torch.manual_seed(42)
#random.seed(42)

funct = "gb"
frac = 0.001
df, columnsName, y = UNSWNB15_dataset.preprocessing(frac)
SWARM_SIZE = 15
MAX_ITERATIONS = 15
component_1 = 2
component_2 = 2
INERTIA = 0.5
globalbest = []
globalbest_val = 0

start_time = time.time()
swarm = []

def process_particle(i, funct, columnsName, df, y):
    inicial_position = pso.Search_Space(funct, n_features=len(columnsName))
    particle = part.Particle(i, inicial_position, funct=funct, columnsName=columnsName)
    particle.pb_val = pso.Evaluate_fitness(funct, particle, columnsName, df, y,particle.index, n_features=len(columnsName))
    particle.pos_val = particle.pb_val
    
    return particle

#process_particle(0, funct, columnsName, df, y)
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
    
for i in range(MAX_ITERATIONS):
    print("Iteração:", i)
    #for particle in swarm:
    #    particle.velocity = pso.checkvelocity(globalbest=globalbest, particle=particle, inertia=INERTIA, c1=component_1, c2=component_2)
    #    particle.position = pso.update_particle(particle, funct, n_features=len(columnsName))
    #    particle.pb_val = pso.Evaluate_fitness(funct,particle,columnsName ,df, y, particle.index, n_features=len(columnsName))
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

swarm[0].position = globalbest

optimal_x_train = optimal_x_train.reset_index(drop = True)
optimal_y_train = optimal_y_train.reset_index(drop = True)
optimal_x_train['class'] = optimal_y_train
benign_x_train_optimal = optimal_x_train[optimal_x_train['class'] == 0]
benign_x_train_optimal = benign_x_train_optimal.drop(labels= 'class', axis = 1)
optimal_x_train = optimal_x_train.drop(labels='class', axis = 1)

optimal_x_val = optimal_x_val.reset_index(drop=True)
optimal_y_val = optimal_y_val.reset_index(drop=True)
optimal_x_val['class'] = optimal_y_val
benign_x_val_optimal = optimal_x_val[optimal_x_val['class'] == 0]
benign_x_val_optimal = benign_x_val_optimal.drop(labels = 'class', axis = 1)
optimal_x_val = optimal_x_val.drop(labels = 'class', axis = 1)

benign_x_train_optimal, optimal_x_val, optimal_x_test, benign_x_val_optimal = dataset.transform_MinMaxScaler(benign_x_train_optimal, optimal_x_val, optimal_x_test,benign_x_val_optimal)

benign_x_val_optimal_tensor = torch.FloatTensor(benign_x_val_optimal)
optimal_x_train_tensor = torch.FloatTensor(benign_x_train_optimal)

# Função para otimizar
def optimize_autoencoder_hyperparameters(IN_FEATURES):
    # Definir intervalos para os hiperparâmetros
    hyperparameter_ranges = {
        'BATCH_SIZE': [64],
        'ALPHA': [1e-4,1e-3,1e-2],
        'PATIENCE': [10],
        'DELTA': [0.0001],
        'NUM_EPOCHS': [1000],
        'DROPOUT_RATE': [0.5],
        'REGULARIZER': [1e-4,1e-3,1e-2]
    }

    # Definir o número de iterações da busca aleatória
    num_iterations = 10
    # Melhores hiperparâmetros e seu desempenho
    best_hyperparameters = {}
    best_f1_score = float('-inf')

    for m in range(num_iterations):
        print("iteração:", m)
        # Amostrar hiperparâmetros aleatoriamente dentro dos intervalos especificados
        hyperparameters = {param: random.choice(values) for param, values in hyperparameter_ranges.items()}

        # Configurar e treinar o modelo com os hiperparâmetros amostrados
        ae_model = ae.Autoencoder(IN_FEATURES, hyperparameters['DROPOUT_RATE'])
        ae_model.compile(learning_rate=hyperparameters['ALPHA'], weight_decay=hyperparameters['REGULARIZER'])

        train_avg_losses, val_avg_losses = ae_model.fit(optimal_x_train_tensor,
                                                        hyperparameters['NUM_EPOCHS'],
                                                        hyperparameters['BATCH_SIZE'],
                                                        X_val=benign_x_val_optimal_tensor,
                                                        patience=hyperparameters['PATIENCE'],
                                                        delta=hyperparameters['DELTA'])

        # Avaliar o desempenho do modelo usando o F1-score
        val_anomaly_scores = ae.get_autoencoder_anomaly_scores(ae_model, optimal_x_val)

        lista_arrays = [float(i) for i in range(1, 5000)]  # Gerar números inteiros de 1 a 90
        lista_arrays = [x / 1000 for x in lista_arrays]
        lista_f1 = []
        best_f1 = 0.0
        best_thresh = 0.0   
        for i in lista_arrays:
            metrics = ae.get_overall_metrics(optimal_y_val, val_anomaly_scores > i)
            f1_score = metrics['f1-score']
            precision = metrics['precision']
            accuracy = metrics['acc']
            tpr = metrics['tpr']
            fpr = metrics['fpr']
            recall = metrics['recall']
            lista_f1.append(f1_score)
            if f1_score > best_f1 and tpr > 0.0 and precision > 0.0:
                best_f1 = f1_score
                best_thresh = i
        print("Melhor Threshold:", best_thresh)
        print("Melhor F1-score:", best_f1)


        # Atualizar os melhores hiperparâmetros se o desempenho atual for melhor
        if best_f1 > best_f1_score:
            best_threshold = best_thresh
            best_precision = precision
            best_accuracy = accuracy
            best_tpr = tpr
            best_fpr = fpr
            best_f1_score = best_f1
            best_hyperparameters = hyperparameters
            best_recall = recall

    return best_threshold,best_hyperparameters, best_f1_score,best_fpr,best_tpr,best_accuracy,best_precision, best_recall

# Executar a otimização dos hiperparâmetros
start_time = time.time()
threshold ,best_hyperparameters, best_f1_score,best_fpr,best_tpr,best_accuracy,best_precision, best_recall = optimize_autoencoder_hyperparameters(optimal_x_train.shape[1],)

# Exibir os melhores hiperparâmetros encontrados e o melhor F1-score
print("Melhores hiperparâmetros:", best_hyperparameters)
print("Melhor F1-score:", best_f1_score)

print("metricas:", "f1_score:", best_f1_score, "precision:", best_precision, "accuracy:", best_accuracy, "tpr:", best_tpr, "fpr:", best_fpr, "recall:", best_recall)


end_time = time.time()
print("Tempo da tunagem")
dataset.get_time(start_time, end_time)

start_time = time.time()
ae_model = ae.Autoencoder(optimal_x_train.shape[1], best_hyperparameters['DROPOUT_RATE'])

ae_model.compile(learning_rate=best_hyperparameters['ALPHA'], weight_decay=best_hyperparameters['REGULARIZER'])

train_avg_losses, val_avg_losses = ae_model.fit(optimal_x_train_tensor,
                                                best_hyperparameters['NUM_EPOCHS'],
                                                best_hyperparameters['BATCH_SIZE'],
                                                X_val=benign_x_val_optimal_tensor,
                                                patience=best_hyperparameters['PATIENCE'],
                                                delta=best_hyperparameters['DELTA'])

test_anomaly_scores = ae.get_autoencoder_anomaly_scores(ae_model, optimal_x_test)
metrics = ae.get_overall_metrics(optimal_y_test, test_anomaly_scores > threshold)

print("Features Esolhidas: ",dataset.particle_choices(globalbest, columnsName, n_features=len(columnsName)))
print(len(dataset.particle_choices(globalbest,columnsName, n_features=len(columnsName))), "no total")
#print('Métricas do ensemble: ')
#particle_choices = dataset.particle_choices(swarm[0].position, columnsName, len(columnsName))
#x_train, y_train, x_val, y_val = dataset.split_train_test(df, columnsName, y, swarm[0].position[0])
#x_train_selected = x_train[particle_choices]
#x_val_selected = x_val[particle_choices]
#gb_model = gb.GradientBoost(x_train_selected, y_train, swarm[0].position[-2], swarm[0].position[-1])
#accuracy_gb, f1_gb, precision_gb, recall_gb = gb.get_metrics(gb_model, x_val_selected, y_val)

#print('accuracy:', accuracy_gb,'f1_score:',f1_gb, 'precision:', precision_gb, 'recall:', recall_gb)

print("Métricas da melhor particula no ensemble",pso.Evaluate_fitness(funct,swarm[0], columnsName, df, y,swarm[0].index, n_features=len(columnsName)))
print("Foi separado", swarm[0].position[0], "do dataset para teste e validação e os valores de learning rate e número de árvores foi de", swarm[0].position[-2], swarm[0].position[-1])
print("metricas:", "f1_score:", metrics['f1-score'], "precision:", metrics['precision'], "accuracy:", metrics['acc'], "tpr:", metrics['tpr'], "fpr:", metrics['fpr'], "recall:", metrics['recall'])

end_time = time.time()
print("Tempo de treinamento do autoencoder com melhores hiperparâmetros")
dataset.get_time(start_time, end_time)
