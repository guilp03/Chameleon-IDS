import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



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
df_train = df.sample(frac = 0.6, random_state = 33)
df_val_test = df.drop(df_train.index)

df_train = df_train.reset_index(drop=True)
df_val_test = df_val_test.reset_index(drop=True)

x_train = df_train.drop('class', axis='columns')

x_val, x_test, classes_val, classes_test = train_test_split(df_val_test.drop('class', axis='columns'), df_val_test['class'], test_size=0.65, stratify=df_val_test['class'], random_state=33)
x_val, x_test = x_val.reset_index(drop=True), x_test.reset_index(drop=True)
classes_val, classes_test =  classes_val.reset_index(drop=True), classes_test.reset_index(drop=True)

y_val, y_test = classes_val.apply(lambda c: 0 if c == 'normal' else 1), classes_test.apply(lambda c: 0 if c == 'normal' else 1)

del df_train, df_val_test

#NORMALIZANDO DADOS

std_scaler = StandardScaler()
std_scaler = std_scaler.fit(x_train)

norm_X_train = std_scaler.transform(x_train)
norm_X_val = std_scaler.transform(x_val)
norm_X_test = std_scaler.transform(x_test)

del x_train, x_val, x_test

#PROXIMAS AÇÕES: SEPARAR AS STRINGS PARA NORMALIZAR OS VALORES, TESTAR O DATASET NOS ALGORITMOS, TESTAR A JUNÇÃO COM O PSO