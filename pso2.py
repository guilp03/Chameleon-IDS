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
from sklearn.preprocessing import StandardScaler
import RandomForest as rf

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
def f1_score_calc(x):
  x_train, y_train, x_val, y_val, x_test, y_test = conjuntos(df[x])
  rf_model = rf.RandomForest(42, x_train, y_train)
  f1_rf, precision_rf, recall_rf = rf.get_metrics(rf_model, x_val, y_val)
  print('f1_score:',f1_rf, 'precision:', precision_rf, 'recall:', recall_rf)
  return f1_rf

def conjuntos(x):
  
  # Dividindo a database em % para treinamento e % para validacao e testes
  x_train, x_val_test, y_train, y_val_test =  train_test_split(x, y, test_size=0.3, random_state=42, stratify=df['class'])

  # Reset dos índices dos subsets
  x_train = x_train.reset_index(drop=True)
  x_val_test = x_val_test.reset_index(drop=True)

  #y_train = classes_train.apply(lambda c: 0 if c == 'normal' else 1)
  #print(y_train)

  # Dividindo o subset de validação + teste em subset de validação e subset de testes
  x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.65, stratify=y_val_test, random_state=33)
  
  # Reset dos índices dos subsets
  x_val, x_test = x_val.reset_index(drop=True), x_test.reset_index(drop=True)
  y_val, y_test =  y_val.reset_index(drop=True), y_test.reset_index(drop=True)
  #y_val, y_test = classes_val.apply(lambda c: 0 if c == 'normal' else 1), classes_test.apply(lambda c: 0 if c == 'normal' else 1)

  del x_val_test
  #print(x_train)
  
  # NORMALIZANDO DADOS
  #train
  x_train = normalize_data(x_train)
  #val
  x_val = normalize_data(x_val)
  #test
  x_test = normalize_data(x_test)

  return x_train, y_train, x_val, y_val, x_test, y_test

def normalize_data(subset):
    std_scaler = StandardScaler()
    colunas_numericas = subset.select_dtypes(include=['number'])
    colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
    subset = colunas_numericas_scaler
    return subset

# MAIN
df = pd.read_csv("csv_result-KDDTrain+_20Percent.csv")
#print(df.info())
for col in df.columns:
    df= df.rename({col:remove_initial_and_ending_spaces(col)}, axis='columns')

# O DATASET CONTÉM ASPAS SIMPLES NOS NOMES DAS COLUNAS, RETIRAR PARA SIMPLIFICAR A ESCRITA
df.columns = df.columns.str.replace("'", "")

# VENDO A PROPORÇÃO BENIGNO X MALICIOSO
#df['class_category'] = df['class'].apply(lambda label: 'Malicious' if label != 'normal' else 'Benign')
#sns.countplot(data=df, x='class_category')
#plt.show()

initial_len = df.shape[0]
df = df.dropna()
#print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros com valores NA')
#DIVISÃO DO CONJUNTO DE TREINO, VALIDAÇÃO E TESTE
columnsName = df.drop(labels= 'class', axis= 1).columns.values.tolist()
y = df['class']
y = y.apply(lambda c: 0 if c == 'normal' else 1)

# Gerando 20 partículas da forma [0 0 1 0 ... 0 1] de tamanho 42 (número de features da database)
columnsName1=[0,1]
particles=[] # Array de partículas
for i in range(20):
    part1=[]
    for i in range(42):
        item = random.choice(tuple(columnsName1))
        part1.append(item)
    particles.append(part1)
#print(particles[0])
    
# Retorna as colunas escolhidas da partícula
def particle_choices(particle):
    particle_collumns=[]
    for i in range(len(particle)):
        if particle[i]!=1:
                particle_collumns.append(columnsName[i])
    #print(particle_choice)
    return particle_collumns

# Personal best array initialization
pb=[]
for i in range(len(particles)):
    pb.append(f1_score_calc(particle_choices(particles[i])))

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
        for m in range(len(particles2[l])):
            if particles2[l][m]>0.5:
                particles2[l][m]=1
            else:
                particles2[l][m]=0
    return particles2

def update_pb(particles2, particles):
    personal=[]
    for i in range(len(particles2)):
        personal.append(f1_score_calc(particle_choices(particles2[i])))
        #print(particles[i])
    for j in range(len(personal)):
        if(personal[j]>pb[j]):
            particles[j]=particles2[j]
            pb[j]=personal[j]
    return personal

# Inicialização
max(pb)
ind = pb.index(max(pb))
globalbest=particles[ind]
velocity = [0] * 42
particles2=[0] * 42 # Novo valor de particles depois da iteração
itter = 10
for i in range(itter):
    #inertia = 0.9 - ((0.5 / itter) * (i))
    inertia = 0.5
    personal=[]
    velocity=checkvelocity(globalbest=globalbest, particles=particles, prev_velocity=velocity, inertia=inertia, prev_particles=particles2)
    particles2=update_particles(velocity, particles)
    particles2=inteiro(particles2)
    personal=update_pb(particles2, particles) # Atualiza valor de personal best (ignorar return)
    particles = particles2
    globalbest=[]
    ind = pb.index(max(pb))
    globalbest=particles[ind]
                
    

ind = pb.index(max(pb))
globalbest=particles[ind]

print(particle_choices(globalbest))
print(len(particle_choices(globalbest)))

#PSO FUNCIONA E A FÓRMULA FOI APRIMORADA, PRÓXIMO PASSO É AVALIAR E COMPARAR O RESULTADO OBTIDO NO PSO COM OS OBTIDOS QUANDO SE UTILIZA TODAS AS FEATURES