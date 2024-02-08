import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pso2 import f1_score_calc_rf,f1_score_calc_gb
import random
from joblib import Parallel, delayed
import time

def normalize_data(subset):
    std_scaler = StandardScaler()
    colunas_numericas = subset.select_dtypes(include=['number'])
    colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
    subset = colunas_numericas_scaler
    return subset

# Features do dataset
features = pd.read_csv(r'csv_NUSW-NB15\NUSW-NB15_features.csv', encoding='ISO-8859-1')
print(features)

# Mapeamento dos tipos de dados
type_mapping = {
    'integer': 'int',
    'Integer': 'int',
    'Float': 'float',
    'nominal': 'str',
    'binary': 'int',  # Ou 'object' se representar strings de bits
    'Binary': 'int',  # Ou 'object' se representar strings de bits
    'Timestamp': 'float'
}

# Criando um dicionário para usar com 'dtype' em 'read_csv'
dtype_dict = {row['Name']: type_mapping[row['Type ']] for index, row in features.iterrows() if row['Type '] in type_mapping}

print(dtype_dict)

df_list = []
for number in range(1, 5):
    df_aux = pd.read_csv(f'csv_NUSW-NB15\\UNSW-NB15_{number}.csv', dtype=dtype_dict, low_memory=False)
    df_aux.columns = features['Name']
    print(len(df_aux.columns))
    df_list.append(df_aux)
df = pd.concat(df_list, ignore_index=True)

NAcolumns = df.columns[df.isna().any(axis=0)]
# ['ct_flw_http_mthd', 'is_ftp_login', 'attack_cat'] são colunas com valores NA
# No caso de 'attack_cat' pode-se dizer que é o tipo do ataque. Quando ocorre NA significa que não foi ataque.
# Dito isso, essa coluna não pode ser usada dito que ela literalmente diz a categoria do ataque.

df = df.drop('attack_cat', axis=1)
df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].fillna(-1) # Só ocorre se foi requisição http
df['is_ftp_login'] = df['is_ftp_login'].fillna(-1) # Só ocorre se foi requisição ftp

# Não há colunas com valores infinitos.

# Transformando tipos categóricos em numéricos (Random Forest não trabalha com valores categóricos)
df_not_numeric = df.select_dtypes(exclude=[np.number])
not_numeric_columns = df_not_numeric.columns
encoder = LabelEncoder()
for column in not_numeric_columns:
    # Garantindo que todos os valores da coluna sejam do tipo string
    df[column] = df[column].astype(str)
    df[column] = encoder.fit_transform(df[column])

# Separando labels
columnsName = df.drop(labels= 'Label', axis= 1).columns.values.tolist()
y = df['Label']
y = y.apply(lambda c: 0 if c == 'normal' else 1)

# Divisão do conjunto de treino validação e teste
# Dividindo a database em % para treinamento e % para validacao e testes
x_train, x_val_test, y_train, y_val_test =  train_test_split(df[columnsName], y, test_size=0.3, random_state=42, stratify=df['Label'])

# Reset dos índices dos subsets
x_train = x_train.reset_index(drop=True)
x_val_test = x_val_test.reset_index(drop=True)

# Dividindo o subset de validação + teste em subset de validação e subset de testes
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.65, stratify=y_val_test, random_state=33)
  
# Reset dos índices dos subsets
x_val, x_test = x_val.reset_index(drop=True), x_test.reset_index(drop=True)
y_val, y_test =  y_val.reset_index(drop=True), y_test.reset_index(drop=True)

del x_val_test

# Normalizando dados
x_train = normalize_data(x_train)
x_val = normalize_data(x_val)
x_test = normalize_data(x_test)

# PSO
# Gerando 20 partículas da forma [0 0 1 0 ... 0 1] de tamanho 42 (número de features da database)
columnsName1=[0,1]
funct = "rf"
particles=[] # Array de partículas
for i in range(15):
    part1=[]
    for i in range(42):
        item = random.choice(tuple(columnsName1))
        part1.append(item)
    if funct == "rf":
        part1.append(random.randint(50, 1000))
        
    if funct == "gb":
        part1.append(random.randint(50, 1000))
        part1.append(random.uniform(0.1, 0.3))
            
    particles.append(part1)
    
# Retorna as colunas escolhidas da partícula
def particle_choices(particle):
    particle_columns=[]
    for i in range(42):
        if particle[i]!=1:
                particle_columns.append(columnsName[i])
    #print(particle_choice)
    return particle_columns

# Personal best array initialization
pb=[]
#for i in range(len(particles)):
    #print(particles[i])
    #chosen_columns = particle_choices(particles[i])
start_time = time.time()
if funct == "rf":
        #pb.append(f1_score_calc_rf(chosen_columns, x_train, y_train, x_val, y_val, x_test, y_test, i, particles[i][42]))
    pb.append(Parallel(n_jobs=-1)(delayed(f1_score_calc_rf)(particle_choices(p), x_train, y_train, x_val, y_val, x_test, y_test, i, p[42]) for i, p in enumerate(particles)))

if funct == "gb":
        #pb.append(f1_score_calc_gb(chosen_columns, x_train, y_train, x_val, y_val, x_test, y_test, i, particles[i][42], particles[i][43]))
    pb.append(Parallel(n_jobs=-1)(delayed(f1_score_calc_gb)(particle_choices(p), x_train, y_train, x_val, y_val, x_test, y_test, i, p[42], p[43]) for i, p in enumerate(particles)))

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
        if funct == "rf":
            if particles2[l][42] > 1000:
                particles2[l][42] = 1000
            elif particles2[l][42] < 50:
                particles2[l][42] = 50
            particles2[l][42] = int(particles2[l][42])
        if funct == "gb":
            if particles2[l][42] > 1000:
                particles2[l][42] = 1000
            elif particles2[l][42] < 50:
                particles2[l][42] = 50
            if particles2[l][43] > 0.3:
                particles2[l][43] = 0.3
            if particles2[l][43] < 0.1:
                particles2[l][43] = 0.1
            particles2[l][42] = int(particles2[l][42])


        for m in range(42):
            if particles2[l][m]>0.5:
                particles2[l][m]=1
            else:
                particles2[l][m]=0
    return particles2

def update_pb(particles2, particles):
    personal=[]
    #for i in range(len(particles2)):
    if funct == "rf":
            #personal.append(f1_score_calc_rf(particle_choices(particles2[i]), x_train, y_train, x_val, y_val, x_test, y_test, i, int(particles2[i][42])))
        personal.append(Parallel(n_jobs=-1)(delayed(f1_score_calc_rf)(particle_choices(p), x_train, y_train, x_val, y_val, x_test, y_test, i, p[42]) for i, p in enumerate(particles2)))

    if funct == "gb":
            #personal.append(f1_score_calc_gb(particle_choices(particles2[i]), x_train, y_train, x_val, y_val, x_test, y_test, i, int(particles2[i][42]), particles2[i][43]))
        personal.append(Parallel(n_jobs=-1)(delayed(f1_score_calc_gb)(particle_choices(p), x_train, y_train, x_val, y_val, x_test, y_test, i, int(p[42]), p[43]) for i, p in enumerate(particles2)))

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
itter = 30
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
    print(particles[ind])
                
end_time = time.time()

ind = pb.index(max(pb))
globalbest=particles[ind]

print(particle_choices(globalbest))
print(len(particle_choices(globalbest)))
execution_time = end_time - start_time
mins = execution_time // 60
segs = execution_time % 60
print("Tempo de execução:", mins, "minutos e", segs,"segundos")
#PSO FUNCIONA E A FÓRMULA FOI APRIMORADA, PRÓXIMO PASSO É AVALIAR E COMPARAR O RESULTADO OBTIDO NO PSO COM OS OBTIDOS QUANDO SE UTILIZA TODAS AS FEATURES