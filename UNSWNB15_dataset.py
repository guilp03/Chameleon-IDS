import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from joblib import Parallel, delayed
import time
import RandomForest as rf
import XGBoost as gb

def normalize_data(subset):
    std_scaler = StandardScaler()
    colunas_numericas = subset.select_dtypes(include=['number'])
    colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
    subset = colunas_numericas_scaler
    return subset

# Features do dataset
def preprocessing(frac: float = 1.0):
    """ Recebe um parâmetro frac e retorna uma fração do dataset UNSW-NB15 com dados já processados """
    features = pd.read_csv(r'archive\NUSW-NB15_features.csv', encoding='ISO-8859-1')
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
        df_aux = pd.read_csv(f'archive\\UNSW-NB15_{number}.csv', dtype=dtype_dict, low_memory=False)
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

    df_subset = df.sample(frac=frac, random_state=1)

    # Separando labels
    columnsName = df_subset.drop(labels= 'Label', axis= 1).columns.values.tolist()
    y = df_subset['Label']

    df_subset = normalize_data(df_subset)

    print('AAAAAAAAAAAAAA')
    print(len(columnsName))

    return df_subset, columnsName, y