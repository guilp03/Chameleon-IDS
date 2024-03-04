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

# Olá, seja bem vindo ao código. Quando começamos a codar, só eu e Deus sabíamos como funcionava. Agora, só Deus.
# Registre aqui o número de horas desperdiçadas tentando otimizar essa rotina: 2h

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
def f1_score_calc_rf(particle_choices, x_train, y_train, x_val, y_val, i, n_estimators):
    # Selecionar as colunas apropriadas
    x_train_selected = x_train[particle_choices]
    x_val_selected = x_val[particle_choices]
    #x_test_selected = x_test[particle_choices]
    rf_model = rf.RandomForest(42, x_train_selected, y_train,n_estimators)
    accuracy_rf, f1_rf, precision_rf, recall_rf = rf.get_metrics(rf_model, x_val_selected, y_val)
    print(i,'accuracy:', accuracy_rf,'f1_score:',f1_rf, 'precision:', precision_rf, 'recall:', recall_rf)
    return f1_rf

def f1_score_calc_gb(particle_choices, x_train, y_train, x_val, y_val, i, n_estimators, leanrning_rate):
    # Selecionar as colunas apropriadas
    x_train_selected = x_train[particle_choices]
    x_val_selected = x_val[particle_choices]
    #x_test_selected = x_test[particle_choices]
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
def split_train_test(df, columnsName, y, test_size):
    # Divisão do conjunto de treino validação e teste
    # Dividindo a database em % para treinamento e % para validacao e testes
    x_train, x_val, y_train, y_val =  train_test_split(df[columnsName], y, test_size=test_size, random_state=42, stratify=df['class'])

    # Reset dos índices dos subsets
    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    
    return x_train, y_train, x_val, y_val

# Retorna as colunas escolhidas da partícula
def particle_choices(particle):
    particle_columns=[]
    for i in range(1, len(df.columns)): # Verifica no índice 1 até o índice 43 (para cada partícula)
        if particle[i]==1:
                particle_columns.append(columnsName[i - 1])
    #print(particle_choice)
    return particle_columns

def calculate_f1_for_particle(p, df, columnsName, y, funct, i):
    # Chamada para split_train_test com o test_size da partícula
    x_train, y_train, x_val, y_val = split_train_test(df, columnsName, y, p[0])
    
    # Executa a função correspondente ao método escolhido
    if funct == "rf":
        return f1_score_calc_rf(particle_choices(p), x_train, y_train, x_val, y_val, i, p[-1])
    elif funct == "gb":
        return f1_score_calc_gb(particle_choices(p), x_train, y_train, x_val, y_val, i, p[-2], p[-1])
    else:
        raise ValueError("Função não reconhecida.")


# PSO
# Gerando 20 partículas da forma [0 0 1 0 ... 0 1] de tamanho 42 (número de features da database)
columnsName1=[0,1]
particles=[] # Array de partículas
for i in range(15):
    part1=[]
    part1.append(random.uniform(0.1, 0.4)) # Train-test-split
    for i in range(len(df.columns) - 1):
        item = random.choice(tuple(columnsName1))
        part1.append(item)
    if funct.funct == "rf":
        part1.append(random.randint(50, 1000))
        
    if funct.funct == "gb":
        part1.append(random.randint(50, 1000))
        part1.append(random.uniform(0.1, 0.3))
            
    particles.append(part1)

# Personal best array initialization
pb=[]
#for i in range(len(particles)):
    #print(particles[i])
    #chosen_columns = particle_choices(particles[i])
start_time = time.time()
results = Parallel(n_jobs=-1)(delayed(calculate_f1_for_particle)(p, df, columnsName, y, funct.funct, i) for i, p in enumerate(particles))
pb.append(results)

def checkvelocity(globalbest, particles, prev_velocity, inertia, prev_particles):
    inertia_array = np.array([inertia])
    velocity=[]
    for j in range(len(particles)):
        velocity.append(list((prev_velocity[j] * inertia_array) + (np.random.random(1)[0]) * (np.array(particles[j]) - np.array(prev_particles[j])) + 2 * (np.random.random(1)[0]) * (np.array(globalbest) - np.array(particles[j]))))
    #print(velocity)
    return velocity

def update_particles(velocity, particles):
    particles_updated=[]
    for i in range(len(velocity)):
        nextparticle=[]
        for j in range(len(velocity[i])):
            nextparticle.append(particles[i][j]+velocity[i][j])
        particles_updated.append(nextparticle)
    return particles_updated

def inteiro(particles2):
    for l in range(len(particles2)):
        if funct.funct == "rf":
            if particles2[l][-1] > 1000:
                particles2[l][-1] = 1000
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
        
        # Transformando train-test-split no range esperado
        if particles2[l][0] > 0.4:
            particles2[l][0] = 0.4
        elif particles2[l][0] < 0.1:
            particles2[l][0] = 0.1
            


        for m in range(1, len(df.columns)):
            if particles2[l][m]>0.5:
                particles2[l][m]=1
            else:
                particles2[l][m]=0
    return particles2

def update_pb(particles2, particles):
    personal=[]
    #for i in range(len(particles2)):
    results = Parallel(n_jobs=-1)(delayed(calculate_f1_for_particle)(p, df, columnsName, y, funct.funct, i) for i, p in enumerate(particles2))
    personal.append(results)

    for j in range(len(personal)):
        if(personal[j]>pb[j]):
            particles[j]=particles2[j]
            pb[j]=personal[j]
    return personal

# Inicialização
max(pb)
ind = pb.index(max(pb))
globalbest=particles[ind]
if funct.funct == 'rf':
    aux = 44
else:
    aux = 45
velocity = [0] * aux
particles2=[0] * aux # Novo valor de particles depois da iteração

print(particles)
print(particles2)
itter = 30
for i in range(itter):
    #inertia = 0.9 - ((0.5 / itter) * (i))
    inertia = 0.5
    personal=[]
    velocity=checkvelocity(globalbest=globalbest, particles=particles, prev_velocity=velocity, inertia=inertia, prev_particles=particles2)
    particles2=update_particles(velocity, particles)
    particles2=inteiro(particles2)
    print(particles2)
    personal=update_pb(particles2, particles) # Atualiza valor de personal best (ignorar return)
    particles = particles2
    globalbest=[]
    ind = pb.index(max(pb))
    globalbest=particles[ind]
    print(particles[ind])
    print("Iteração:", i)
                
end_time = time.time()

ind = pb.index(max(pb))
globalbest=particles[ind]

print(particle_choices(globalbest))
print(len(particle_choices(globalbest)))

execution_time = end_time - start_time
mins = execution_time // 60
hours = mins // 60
segs = execution_time % 60
print("Tempo de execução:", int(hours), "horas,", int(mins), "minutos e", int(segs),"segundos")
del df