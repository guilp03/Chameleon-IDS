import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import RandomForest as rf
import pso2 as pso



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

df_train = df.sample(frac = 0.6, random_state = 33)
df_val_test = df.drop(df_train.index)

df_train = df_train.reset_index(drop=True)
df_val_test = df_val_test.reset_index(drop=True)

x_train = df_train.drop('class', axis='columns')
classes_train = df_train['class']
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

rf_model = rf.RandomForest(42, x_train, y_train)
f1_rf = rf.get_metrics(rf_model, x_val, y_val)
print('Accuracy:',f1_rf)



#PROXIMAS AÇÕES: TESTAR O DATASET NOS ALGORITMOS, TESTAR A JUNÇÃO COM O PSO