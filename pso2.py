from sklearn import preprocessing
preprocessing.LabelEncoder()
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.max_columns", None)
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import RandomForest as rf
import XGBoost as gb
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import time
import funct
import Autoencoder as ae
import torch
import torch.nn as nn
import torch.optim as optim


# Definindo seed de aleatoriedade
random.seed(10)
# Função para remover espaços iniciais e finais
def remove_initial_and_ending_spaces(name):
    regex = r'^(?:\s+)?(?P<gp>.+?)(?:\s+)?$'
    mo = re.search(regex, name)
    if mo is not None:
        return mo['gp']
    else:
        print(f'Deu erro em: {name}')
        return name
    
#FUNÇÃO QUE CALCULA O F1 DO RANDOM FOREST DADO UM CONJUNTO DE FEATURES
def f1_score_calc_rf(particle_choices, x_train, y_train, x_val, y_val, x_test, y_test, i, n_estimators):
    # Selecionar as colunas apropriadas
    x_train_selected = x_train[particle_choices]
    x_val_selected = x_val[particle_choices]
    x_test_selected = x_test[particle_choices]
    rf_model = rf.RandomForest(42, x_train_selected, y_train,n_estimators)
    accuracy_rf, f1_rf, precision_rf, recall_rf = rf.get_metrics(rf_model, x_val_selected, y_val)
    print(i,'accuracy:', accuracy_rf,'f1_score:',f1_rf, 'precision:', precision_rf, 'recall:', recall_rf)
    return f1_rf

def f1_score_calc_gb(particle_choices, x_train, y_train, x_val, y_val, x_test, y_test, i, n_estimators, leanrning_rate):
    # Selecionar as colunas apropriadas
    x_train_selected = x_train[particle_choices]
    x_val_selected = x_val[particle_choices]
    x_test_selected = x_test[particle_choices]
    gb_model = gb.GradientBoost(x_train_selected, y_train, n_estimators, leanrning_rate)
    accuracy_gb, f1_gb, precision_gb, recall_gb = gb.get_metrics(gb_model, x_val_selected, y_val)
    print(i,'accuracy:', accuracy_gb,'f1_score:',f1_gb, 'precision:', precision_gb, 'recall:', recall_gb)
    return f1_gb

def normalize_data(subset):
    std_scaler = StandardScaler()
    colunas_numericas = subset.select_dtypes(include=['number'])
    colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
    subset = colunas_numericas_scaler
    return subset

# TRATANDO O DATASET
df = pd.read_csv("csv_result-KDDTrain+_20Percent.csv")
#df = pd.read_csv("KDDTrain+.csv")
for col in df.columns:
    df= df.rename({col:remove_initial_and_ending_spaces(col)}, axis='columns')

# O DATASET CONTÉM ASPAS SIMPLES NOS NOMES DAS COLUNAS, RETIRAR PARA SIMPLIFICAR A ESCRITA
df.columns = df.columns.str.replace("'", "")

initial_len = df.shape[0]
df = df.dropna()

# Separando labels
columnsName = df.drop(labels= 'class', axis= 1).columns.values.tolist()
y = df['class']
y = y.apply(lambda c: 0 if c == 'normal' else 1)

# Transformando tipos categóricos em numéricos (Random Forest não trabalha com valores categóricos)
df_not_numeric = df.select_dtypes(exclude=[np.number])
not_numeric_columns = df_not_numeric.columns
encoder = LabelEncoder()
for column in not_numeric_columns:
    df[column] = encoder.fit_transform(df[column])

#normalize_data(df)

# Divisão do conjunto de treino validação e teste
# Dividindo a database em % para treinamento e % para validacao e testes
x_train, x_val_test, y_train, y_val_test =  train_test_split(df[columnsName], y, test_size=0.1, random_state=42, stratify=df['class'])

# Reset dos índices dos subsets
x_train = x_train.reset_index(drop=True)
x_val_test = x_val_test.reset_index(drop=True)

# Dividindo o subset de validação + teste em subset de validação e subset de testes
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.65, stratify=y_val_test, random_state=33)
  
# Reset dos índices dos subsets
x_val, x_test = x_val.reset_index(drop=True), x_test.reset_index(drop=True)
y_val, y_test =  y_val.reset_index(drop=True), y_test.reset_index(drop=True)

del x_val_test

# PSO
# Gerando 20 partículas da forma [0 0 1 0 ... 0 1] de tamanho 42 (número de features da database)
columnsName1=[0,1]
particles=[] # Array de partículas
for i in range(15):
    part1=[]
    for i in range(42):
        item = random.choice(tuple(columnsName1))
        part1.append(item)
    if funct.funct == "rf":
        part1.append(random.randint(50, 600))
        
    if funct.funct == "gb":
        part1.append(random.randint(50, 1000))
        part1.append(random.uniform(0.1, 0.3))
            
    particles.append(part1)
    
# Retorna as colunas escolhidas da partícula
def particle_choices(particle):
    particle_columns=[]
    for i in range(42):
        if particle[i]==1:
                particle_columns.append(columnsName[i])
    #print(particle_choice)
    return particle_columns

# Personal best array initialization
pb=[]
#for i in range(len(particles)):
    #print(particles[i])
    #chosen_columns = particle_choices(particles[i])
start_time = time.time()
if funct.funct == "rf":
        #pb.append(f1_score_calc_rf(chosen_columns, x_train, y_train, x_val, y_val, x_test, y_test, i, particles[i][42]))
    pb.extend(Parallel(n_jobs=-1)(delayed(f1_score_calc_rf)(particle_choices(p), x_train, y_train, x_val, y_val, x_test, y_test, i, p[42]) for i, p in enumerate(particles)))

if funct.funct == "gb":
        #pb.append(f1_score_calc_gb(chosen_columns, x_train, y_train, x_val, y_val, x_test, y_test, i, particles[i][42], particles[i][43]))
    pb.extend(Parallel(n_jobs=-1)(delayed(f1_score_calc_gb)(particle_choices(p), x_train, y_train, x_val, y_val, x_test, y_test, i, p[42], p[43]) for i, p in enumerate(particles)))
def checkvelocity(globalbest, pb_particles, prev_velocity, inertia, prev_particles):
    inertia_array = np.array([inertia])
    velocity=[]
    for j in range(len(particles)):
        velocity.append(list((prev_velocity[j] * inertia_array) + 2 * random.random() * (np.array(pb_particles[j]) - np.array(prev_particles[j])) + 2 * random.random() * (np.array(globalbest) - np.array(prev_particles[j]))))
    #print(velocity)
    return velocity

def update_particles(velocity, particles):
    particles_updated=[]
    for i in range(len(velocity)):
        #print(particles[i])
        nextparticle=[]
        for j in range(len(velocity[i])):
            nextparticle.append(particles[i][j]+velocity[i][j])
    
        particles_updated.append(nextparticle)
        #print(particles_updated[i])

    return particles_updated

def inteiro(particles2):
    for l in range(len(particles2)):
        if funct.funct == "rf":
            if particles2[l][-1] > 600:
                particles2[l][-1] = 600
            elif particles2[l][-1] < 50:
                particles2[l][-1] = 50
            particles2[l][-1] = int(particles2[l][-1])
        if funct.funct == "gb":
            if particles2[l][-2] > 1000:
                particles2[l][-2] = 1000
            elif particles2[l][-2] < 50:
                particles2[l][-2] = 50
            if particles2[l][-1] > 0.3:
                particles2[l][-1] = 0.3
            if particles2[l][-1] < 0.1:
                particles2[l][-1] = 0.1
            particles2[l][-2] = int(particles2[l][-2])


        for m in range(42):
            if particles2[l][m]>0.5:
                particles2[l][m]=1
            else:
                particles2[l][m]=0
    return particles2

def update_pb(particles2, particles, pb):
    personal=[]
    #for i in range(len(particles2)):
    if funct.funct == "rf":
            #personal.append(f1_score_calc_rf(particle_choices(particles2[i]), x_train, y_train, x_val, y_val, x_test, y_test, i, int(particles2[i][42])))
        personal.extend(Parallel(n_jobs=-1)(delayed(f1_score_calc_rf)(particle_choices(p), x_train, y_train, x_val, y_val, x_test, y_test, i, p[42]) for i, p in enumerate(particles2)))

    if funct.funct == "gb":
            #personal.append(f1_score_calc_gb(particle_choices(particles2[i]), x_train, y_train, x_val, y_val, x_test, y_test, i, int(particles2[i][42]), particles2[i][43]))
        personal.extend(Parallel(n_jobs=-1)(delayed(f1_score_calc_gb)(particle_choices(p), x_train, y_train, x_val, y_val, x_test, y_test, i, int(p[42]), p[43]) for i, p in enumerate(particles2)))
        
    for j in range(len(personal)):
        if(personal[j]>pb[j]):
            particles[j]=particles2[j]
            pb[j]=personal[j]
    return personal

# Inicialização
#pb = pb.tolist()
ind = pb.index(max(pb))
print(particles[ind])
globalbest=particles[ind]
globalbest_f1 = 0.5
if funct.funct == 'gb':
    velocity = [0] * 44
elif funct.funct == 'rf':
    velocity = [0] * 43
particles2 = particles
itter = 5
for i in range(itter):
    #inertia = 0.9 - ((0.5 / itter) * (i))
    inertia = 0.5
    personal=[]
    velocity=checkvelocity(globalbest=globalbest, pb_particles=particles, prev_velocity=velocity, inertia=inertia, prev_particles=particles2)
    particles2=update_particles(velocity, particles)
    particles2=inteiro(particles2)
    personal=update_pb(particles2, particles, pb) # Atualiza valor de personal best (ignorar return)
    ind = pb.index(max(pb))
    print(max(pb))
    if max(pb) > globalbest_f1:   
     globalbest=particles[ind]
     globalbest_f1 = max(pb)
    print("Iteração:", i)
    print(globalbest)
    print(globalbest_f1)
                
end_time = time.time()

print(particle_choices(globalbest))
print(len(particle_choices(globalbest)))

execution_time = end_time - start_time
mins = execution_time // 60
hours = mins // 60
mins = execution_time // 60 - (hours *60)
segs = execution_time % 60
print("Tempo de execução:", int(hours), "horas,", int(mins), "minutos e", int(segs),"segundos")
del df
#PSO FUNCIONA E A FÓRMULA FOI APRIMORADA, PRÓXIMO PASSO É AVALIAR E COMPARAR O RESULTADO OBTIDO NO PSO COM OS OBTIDOS QUANDO SE UTILIZA TODAS AS FEATURES
minmax_scaler = MinMaxScaler()
colunas_selecionadas = particle_choices(globalbest)
x_train_optimal = x_train[colunas_selecionadas]
x_val_optimal = x_val[colunas_selecionadas]
x_train_optimal['class'] = y_train
#x_train_optimal = x_train_optimal[x_train_optimal['class'] == 0]
x_train_optimal = x_train_optimal.drop(labels= 'class', axis= 1)

minmax_scaler = minmax_scaler.fit(x_train_optimal)

x_train_optimal = minmax_scaler.transform(x_train_optimal)
x_val_optimal = minmax_scaler.transform(x_val_optimal)

#x_train_optimal = normalize_data(x_train_optimal)
del x_train, x_val
benign_x_val_optimal = x_val_optimal[y_val == 1]

benign_x_val_optimal = torch.FloatTensor(benign_x_val_optimal)

BATCH_SIZE = 32
ALPHA = 5e-4
PATIENCE = 7
DELTA = 0.001
NUM_EPOCHS = 1000
IN_FEATURES = x_train_optimal.shape[1]
start_time = time.time()
ae_model = ae.Autoencoder(IN_FEATURES)
ae_model.compile(learning_rate= ALPHA)
train_avg_losses, val_avg_losses = ae_model.fit(torch.FloatTensor(x_train_optimal),
                                                NUM_EPOCHS,
                                                BATCH_SIZE,
                                                X_val = benign_x_val_optimal,
                                                patience = PATIENCE,
                                                delta = DELTA)

end_time = time.time()
execution_time = end_time - start_time
mins = execution_time // 60
hours = mins // 60
mins = execution_time // 60 - (hours *60)
segs = execution_time % 60
print("Tempo de execução:", int(hours), "horas,", int(mins), "minutos e", int(segs),"segundos")

def plot_train_val_losses(train_avg_losses, val_avg_losses):
  epochs = list(range(1, len(train_avg_losses)+1))
  plt.plot(epochs, train_avg_losses, color='blue', label='Loss do treino')
  plt.plot(epochs, val_avg_losses, color='orange', label='Loss da validação')
  plt.title('Losses de treino e validação por época de treinamento')
  plt.legend()
  plt.show()

plot_train_val_losses(train_avg_losses, val_avg_losses)

def get_autoencoder_anomaly_scores(ae_model, X):
  X = torch.FloatTensor(X)
  reconstructed_X = ae_model(X)
  anomaly_scores = torch.mean(torch.pow(X - reconstructed_X, 2), axis=1).detach().numpy() # MSELoss
  return anomaly_scores

val_anomaly_scores = get_autoencoder_anomaly_scores(ae_model, x_val_optimal)
#ALTERAR A ARQUITETURA DO AUTOENCODER, ADICIONAR UM REGULARIZADOR E TESTAR

def get_overall_metrics(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  acc = (tp+tn)/(tp+tn+fp+fn)
  tpr = tp/(tp+fn)
  fpr = fp/(fp+tn)
  precision = tp/(tp+fp)
  f1 = (2*tpr*precision)/(tpr+precision)
  return {'acc':acc,'tpr':tpr,'fpr':fpr,'precision':precision,'f1-score':f1}

#BEST_VALIDATION_THRESHOLD = 0.02
#print(get_overall_metrics(y_val, val_anomaly_scores > BEST_VALIDATION_THRESHOLD))

lista_arrays = [float(i) for i in range(1, 91)]  # Gerar números inteiros de 1 a 90
lista_arrays = [x / 100 for x in lista_arrays]
lista_f1 =[]
best_f1 = 0.0
best_thresh = 0.0
for i in lista_arrays:
    metrics = get_overall_metrics(y_val, val_anomaly_scores > i)
    print(metrics)
    f1_score = metrics['f1-score']
    lista_f1.append(f1_score)
    if f1_score > best_f1:
        best_f1 = f1_score
        best_thresh = i

print("Melhor Threshold:", best_thresh)
print("Melhor F1-score:", best_f1)

plt.plot(lista_arrays, lista_f1, color='blue', label='f1s')
plt.title('f1')
plt.legend()
plt.show()
