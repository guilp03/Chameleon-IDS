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
from sklearn.preprocessing import LabelEncoder
from particle_schema import Particle



# Função para remover espaços iniciais e finais
def remove_initial_and_ending_spaces(name):
    regex = r'^(?:\s+)?(?P<gp>.+?)(?:\s+)?$'
    mo = re.search(regex, name)
    if mo is not None:
        return mo['gp']
    else:
        print(f'Deu erro em: {name}')
        return name
      
def normalize_data(subset):
    std_scaler = StandardScaler()
    colunas_numericas = subset.select_dtypes(include=['number'])
    colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
    subset = colunas_numericas_scaler
    return subset

def split_train_test(df, columnsName, y, test_size):
    """ Divisão do conjunto em treino e teste 
    
        Args:
            df (pd.DataFrame): base de dados
            columnsName (list[str]): lista de colunas do dataset
            y (pd.DataFrame): labels
            test_size (float): tamanho das samples de testes
        
        Returns:
            x_train (pd.DataFrame): subset de treinamento do df
            y_train (pd.DataFrame): labels do subset de treinamento do df
            x_val (pd.DataFrame): subset de validação do df
            y_val (pd.DataFrame): labels do subset de validação do df

    """
    # Divisão do conjunto de treino validação e teste
    # Dividindo a database em % para treinamento e % para validacao e testes
    x_train, x_val, y_train, y_val =  train_test_split(df[columnsName], y, test_size=test_size, random_state=42, stratify=df['class'])

    # Reset dos índices dos subsets
    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    
    return x_train, y_train, x_val, y_val

def preprocessing(df):
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

    df = normalize_data(df)

    return df

def particle_choices(particle: Particle, columnsName: list[str], n_features: int):
    """ Retorna um array de strings com o nome das colunas selecionadas pela partícula

        Args:
            particle (Particle): partícula em questão
            columnsName (list[str]): nome das colunas do dataset
            n_features (int): número de features do dataset

        Returns:
            particle_columns (list[str]): lista das colunas escolhidas pela partícula
    """
    particle_columns=[]
    for i in range(1, n_features + 1): # Verifica do índice 1 até o índice n_features (range é exclusivo com limite superior)
        if particle.position[i] == 1:
                particle_columns.append(columnsName[i - 1])

    return particle_columns

def normalize_data(subset):
    std_scaler = StandardScaler()
    colunas_numericas = subset.select_dtypes(include=['number'])
    colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
    subset = colunas_numericas_scaler
    return subset