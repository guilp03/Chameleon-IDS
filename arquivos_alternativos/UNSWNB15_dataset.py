import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

def normalize_data(subset):
    std_scaler = StandardScaler()
    colunas_numericas = subset.select_dtypes(include=['number'])
    colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
    subset = colunas_numericas_scaler
    return subset

def UNSW_NB15_preprocessing(df_frac = 1.0, random_seed = 33):
    """ Recebe uma fração e uma random_seed e faz o pré-processamento de dados do dataset
        Args:
            df_frac (float): fração do dataset a ser retornada.

            random_seed (int): definição de seed de aleatoriedade para tornar os resultados mais previsíveis.

        Returns:

            df_subset (pandas.Dataframe): subset desejado do dataset UNSW-NB15
    
    """
    features = pd.read_csv('./archive/NUSW-NB15_features.csv', encoding='ISO-8859-1')

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

    # Lendo e concatenando arquivos
    df_list = []
    for number in range(1, 5):
        df_aux = pd.read_csv(f'./archive/UNSW-NB15_{number}.csv', dtype=dtype_dict, low_memory=False)
        df_aux.columns = features['Name'] # Adicionando o tipo da feature em uma coluna separada
        df_list.append(df_aux)
    df = pd.concat(df_list, ignore_index=True)

    """ 
    Trabalhando com colunas de valores NA ou NaN
        As colunas 'ct_flw_http_mthd', 'is_ftp_login', 'attack_cat' são colunas com valores NA:

            'ct_flw_http_mhd' só ocorre se a requisição foi HTTP. Quando não foi é NA.

            'is_ftp_long' é a mesma lógica, só ocorre se a requisição foi FTP.

            'attack_cat' é uma coluna categórica com o tipo do ataque. Quando ocorre NA significa que não foi ataque. Dito isso, essa coluna não pode ser usada pois revela se é ataque ou não. Apesar disso vamos mantê-la por enquanto para fazer uma análise exploratória dos dados.
        
    """
    df['attack_cat'] = df['attack_cat'].fillna('BENIGN')
    df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].fillna(-1) # Só ocorre se foi requisição http
    df['is_ftp_login'] = df['is_ftp_login'].fillna(-1) # Só ocorre se foi requisição ftp

    # Não há colunas com valores infinitos

    # Definindo subset
    df_subset = df.sample(frac=df_frac, random_state=random_seed)

    # Separando labels
    columnsName = df_subset.drop(labels= 'Label', axis= 1).columns.values.tolist()
    y = df_subset['Label']

    # Drop da classe label do df_subset
    df_subset = df_subset.drop(labels= 'Label', axis = 1)

    # Normalização
    # Transformando tipos categóricos em numéricos (Random Forest não trabalha com valores categóricos)
    df_subset_not_numeric = df_subset.select_dtypes(exclude=[np.number])
    not_numeric_columns = df_subset_not_numeric.columns
    encoder = LabelEncoder()
    for column in not_numeric_columns:
        # Garantindo que todos os valores da coluna sejam do tipo string
        df_subset[column] = df_subset[column].astype(str)
        df_subset[column] = encoder.fit_transform(df_subset[column])

    df_subset = normalize_data(df_subset)

    return df_subset
