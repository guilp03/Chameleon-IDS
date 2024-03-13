import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from particle_schema import Particle
      
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
    x_train, x_val, y_train, y_val =  train_test_split(df[columnsName], y, test_size=test_size, random_state=42, stratify=y)

    # Reset dos índices dos subsets
    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    
    return x_train, y_train, x_val, y_val

def split_train_val_test(df, columnsName,y ,test_size):
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
    x_train, x_val_test, y_train, y_val_test =  train_test_split(df[columnsName], y, test_size=test_size, random_state=42, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test)

    # Reset dos índices dos subsets
    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    return x_train, y_train, x_val, y_val, x_test, y_test
    
def remove_initial_and_ending_spaces(name):
    # Define a expressão regular para encontrar espaços em branco no início e no final da string
    regex = r'^(?:\s+)?(?P<gp>.+?)(?:\s+)?$'
    
    # Procura por correspondências na string usando a expressão regular
    mo = re.search(regex, name)
    
    # Se houver uma correspondência
    if mo is not None:
        # Retorna o grupo de caracteres capturado pela expressão regular
        return mo['gp']
    else:
        # Se ocorrer um erro, imprime uma mensagem de erro e retorna o nome original
        print(f'Deu erro em: {name}')
        return name

    
def preprocessing_CICS(df):
    # Renomeia as colunas removendo espaços em branco no início e no final de cada nome de coluna
    for col in df.columns:
        df = df.rename({col: remove_initial_and_ending_spaces(col)}, axis='columns')
    
    # Conta o número de linhas inicial
    initial_len = df.shape[0]
    
    # Remove duplicatas do dataframe
    df = df.drop_duplicates()
    print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartadas {initial_len - df.shape[0]} duplicadas')
    
    # Descarta registros com valores NaN/Null/NA
    initial_len = df.shape[0]
    df = df.dropna()
    print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros com valores NA')
    
    # Redefine os índices após as operações de limpeza
    df = df.reset_index(drop=True)
    
    # Verifica se há valores infinitos nas colunas numéricas
    df_columns_isfinite = np.isfinite(df.drop(['Label'], axis='columns')).all(axis=0)
    df_columns_isfinite[df_columns_isfinite == False] 
    df_rows_isfinite = np.isfinite(df.drop(['Label'], axis='columns')).all(axis=1)
    inf_indexes = df_rows_isfinite[df_rows_isfinite == False].index
    
    # Encontra os valores máximos finitos das colunas 'Flow Packets/s' e 'Flow Bytes/s'
    max_finite_flow_packets_per_sec = df[np.isfinite(df['Flow Packets/s'])]['Flow Packets/s'].max()
    max_finite_flow_bytes_per_sec = df[np.isfinite(df['Flow Bytes/s'])]['Flow Bytes/s'].max()

    # Substitui os valores infinitos nas colunas 'Flow Packets/s' e 'Flow Bytes/s' pelos valores máximos finitos
    df.loc[df['Flow Packets/s'] == np.inf, 'Flow Packets/s'] = max_finite_flow_packets_per_sec
    df.loc[df['Flow Bytes/s'] == np.inf, 'Flow Bytes/s'] = max_finite_flow_bytes_per_sec
    
    # Seleciona colunas não numéricas
    df_not_numeric = df.select_dtypes(exclude=[np.number])
    not_numeric_columns = df_not_numeric.columns
    
    # Codifica as colunas não numéricas usando LabelEncoder
    y = df['Label']
    encoder = LabelEncoder()
    for column in not_numeric_columns:
        df[column] = encoder.fit_transform(df[column])
    
    # Obtém os nomes das colunas após a remoção da coluna 'Label'
    columnsName = df.drop(labels='Label', axis=1).columns.values.tolist()
    
    # Converte os rótulos do alvo ('Label') para valores numéricos (0 para 'BENIGN' e 1 para outras classes)
    y = y.apply(lambda c: 0 if c == 'BENIGN' else 1)
    
    df = df.drop(labels='Label', axis=1)
    
    df = normalize_data(df)
    
    return df, columnsName, y


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
    df = df.drop(labels= 'class', axis=1)

    return df, columnsName, y

def particle_choices(pos, columnsName: list[str], n_features: int):
    """ Retorna um array de strings com o nome das colunas selecionadas pela partícula

        Args:
            pos (list): a posição da partícula
            columnsName (list[str]): nome das colunas do dataset
            n_features (int): número de features do dataset

        Returns:
            particle_columns (list[str]): lista das colunas escolhidas pela partícula
    """
    particle_columns=[]
    for i in range(1, n_features + 1): # Verifica do índice 1 até o índice n_features (range é exclusivo com limite superior)
        if pos[i] == 1:
                particle_columns.append(columnsName[i - 1])

    return particle_columns

def normalize_data(subset):
    # Inicializa o StandardScaler
    std_scaler = StandardScaler()
    
    # Seleciona apenas as colunas numéricas do conjunto de dados
    colunas_numericas = subset.select_dtypes(include=['number'])
    
    # Aplica o StandardScaler nas colunas numéricas e cria um DataFrame com os dados normalizados
    colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
    subset = colunas_numericas_scaler
    
    return subset


def get_optimal_subesets(df, optimal_solution, columnsName, y, test_size, n_features):
    # Divida os dados em conjuntos de treinamento, validação e teste
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(df, columnsName, y, test_size)
    
    # Escolha as colunas relevantes conforme a solução ótima encontrada pelo PSO
    chosen_columns = particle_choices(optimal_solution, columnsName, n_features)
    
    # Selecione apenas as colunas relevantes nos conjuntos de dados
    x_train_selected = x_train[chosen_columns]
    x_val_selected = x_val[chosen_columns]
    x_test_selected = x_test[chosen_columns]
    
    return x_train_selected, x_val_selected, x_test_selected, y_train, y_val, y_test


def transform_MinMaxScaler(x_train, x_val, x_test, benign_x_train):
    # Cria uma instância do MinMaxScaler
    minmax_scaler = MinMaxScaler()
    
    # Ajusta o MinMaxScaler aos dados de treinamento
    minmax_scaler = minmax_scaler.fit(x_train)
    
    # Aplica o escalonamento Min-Max aos dados 
    x_train = minmax_scaler.transform(x_train)
    x_val = minmax_scaler.transform(x_val)
    x_test = minmax_scaler.transform(x_test)
    benign_x_train = minmax_scaler.transform(benign_x_train)
    
    # Retorna os dados escalonados
    return x_train, x_val, x_test, benign_x_train


def get_time(start_time, end_time):
    
    execution_time = end_time - start_time
    mins = execution_time // 60
    hours = mins // 60
    mins = execution_time // 60 - (hours * 60)
    segs = execution_time % 60
    
    # Imprime o tempo de execução formatado
    print("Tempo de execução:", int(hours), "horas,", int(mins), "minutos e", int(segs), "segundos")
