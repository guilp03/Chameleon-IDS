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
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import RandomForest as rf



df = pd.read_csv("csv_result-KDDTrain+_20Percent.csv")
#print(df.info())
# REMOVENDO ESPAÇOS NO FINAL E NO COMEÇO
def remove_initial_and_ending_spaces(name):
    regex = r'^(?:\s+)?(?P<gp>.+?)(?:\s+)?$'
    mo = re.search(regex, name)
    if mo is not None:
        return mo['gp']
    else:
        print(f'Deu erro em: {name}')
        return name
for col in df.columns:
    df= df.rename({col:remove_initial_and_ending_spaces(col)}, axis='columns')
if 'class' not in df.columns:
    print("Warning: 'class' column not found after renaming.")

 #O DATASET CONTÉM ASPAS SIMPLES NOS NOMES DAS COLUNAS, RETIRAR PARA SIMPLIFICAR A ESCRITA
df.columns = df.columns.str.replace("'", "")
 #VENDO A PROPORÇÃO BENIGNO X MALICIOSO
#df['class_category'] = df['class'].apply(lambda label: 'Malicious' if label != 'normal' else 'Benign')
#sns.countplot(data=df, x='class_category')
#plt.show()

initial_len = df.shape[0]
df = df.dropna()
print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros com valores NA')
#DIVISÃO DO CONJUNTO DE TREINO, VALIDAÇÃO E TESTE
columnsName = df.drop(labels= 'class', axis= 1).columns.values.tolist()
#FUNÇÃO QUE CALCULA O F1 DO RANDOM FOREST DADO UM CONJUNTO DE FEATURES
def accuracy_calc(x):
  x_train, y_train, x_val, y_val = conjuntos(df[x])
  rf_model = rf.RandomForest(42, x_train, y_train)
  f1_rf = rf.get_metrics(rf_model, x_val, y_val)
  print('Accuracy:',f1_rf)
  return f1_rf

def conjuntos(df):
  print(df.columns)
  df_train = df.sample(frac = 0.6, random_state = 33)
  df_val_test = df.drop(df_train.index)
  print(df_train.columns)

  df_train = df_train.reset_index(drop=True)
  df_val_test = df_val_test.reset_index(drop=True)
  classes_train = df_train['class']
  x_train = df_train.drop('class', axis='columns')
  y_train = classes_train.apply(lambda c: 0 if c == 'normal' else 1)
  #print(y_train)
  x_val, x_test, classes_val, classes_test = train_test_split(df_val_test.drop('class', axis='columns'), df_val_test['class'], test_size=0.65, stratify=df_val_test['class'], random_state=33)
  x_val, x_test = x_val.reset_index(drop=True), x_test.reset_index(drop=True)
  classes_val, classes_test =  classes_val.reset_index(drop=True), classes_test.reset_index(drop=True)

  y_val, y_test = classes_val.apply(lambda c: 0 if c == 'normal' else 1), classes_test.apply(lambda c: 0 if c == 'normal' else 1)

  del df_train, df_val_test 
  
  #print(x_train)
  #NORMALIZANDO DADOS
  #train
  std_scaler = StandardScaler()
  colunas_numericas = x_train.select_dtypes(include=['number'])
  colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
  x_train = colunas_numericas_scaler
  #print(x_train)
  #val
  colunas_numericas = x_val.select_dtypes(include=['number'])
  colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
  x_val = colunas_numericas_scaler
  #test
  colunas_numericas = x_test.select_dtypes(include=['number'])
  colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
  x_test = colunas_numericas_scaler
  
  return x_train, y_train, x_val, y_val


columnsName1=[0,1]
particles=[]
for i in range(10):
    part1=[]
    for i in range(41):
        item = random.choice(tuple(columnsName1))
        part1.append(item)
    part1.append(0)
    particles.append(part1)
    
    
def data(particles1):
    particles2=[]
    for i in range(len(particles1)):
        if particles1[i]!=1:
                particles2.append(columnsName[i])
    return particles2

pb=[]
def checkpersonalnest(particles):
    for i in range(len(particles)):
         pb.append(accuracy_calc(data(particles[i])))
checkpersonalnest(particles)

def checkvelocity(globalbest, particles):
    velocity=[]
    for j in range(len(particles)):
        velocity.append(list(0+1*(np.random.random(1)[0])*(np.array(particles[j])-np.array(particles[j]))+1*(np.random.random(1)[0])*(np.array(globalbest)-np.array(particles[j]))))
    #print(velocity)
    return velocity

def addingparticles(velocity, particles):
    particles2=[]
    for i in range(len(velocity)):
        nextparticle=[]
        for j in range(len(velocity[i])):
            nextparticle.append(particles[i][j]+velocity[i][j])
        particles2.append(nextparticle)
    return particles2

def inteiro(particles2):
    for l in range(len(particles2)):
        for m in range(len(particles2[l])):
            if particles2[l][m]>0.5:
                particles2[l][m]=1
            else:
                particles2[l][m]=0
    return particles2

def checkpd(particles2, particles):
    personal=[]
    for i in range(len(particles2)):
        personal.append(accuracy_calc(data(particles2[i])))
    for j in range(len(personal)):
        if(personal[j]>pb[j]):
            particles[j]=particles2[j]
            pb[j]=personal[j]
    return personal

max(pb)
ind = pb.index(max(pb))
globalbest=particles[ind]
for i in range(10):
    particles2=[]
    personal=[]
    velocity=checkvelocity(globalbest, particles)
    particles2=addingparticles(velocity, particles)
    particles2=inteiro(particles2)
    personal=checkpd(particles2, particles)
    particles = particles2
    globalbest=[]
    max(pb)
    ind = pb.index(max(pb))
    globalbest=particles[ind]
                
    
max(pb)

ind = pb.index(max(pb))
globalbest=particles[ind]

print(data(globalbest))

#FALTA CHECAR A ATUALIZAÇÃO DE VALORES DE PARTICLES