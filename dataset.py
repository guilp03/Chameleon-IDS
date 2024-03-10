import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from particle_schema import Particle
np.random.seed(42)


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
    std_scaler = StandardScaler()
    colunas_numericas = subset.select_dtypes(include=['number'])
    colunas_numericas_scaler = pd.DataFrame(std_scaler.fit_transform(colunas_numericas), columns=colunas_numericas.columns)
    subset = colunas_numericas_scaler
    return subset

def get_optimal_subesets(df, optimal_solution, columnsName, y , test_size, n_features):
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(df, columnsName,y ,test_size)
    print(x_train)
    print(x_val)
    print(x_test)
    chosen_columns = particle_choices(optimal_solution, columnsName, n_features)
    x_train_selected = x_train[chosen_columns]
    x_val_selected = x_val[chosen_columns]
    x_test_selected = x_test[chosen_columns]
    
    return x_train_selected, x_val_selected, x_test_selected, y_train, y_val, y_test

def transform_MinMaxScaler(x_train, x_val, x_test, benign_x_train):
    minmax_scaler = MinMaxScaler()
    minmax_scaler = minmax_scaler.fit(x_train)
    
    x_train = minmax_scaler.transform(x_train)
    x_val = minmax_scaler.transform(x_val)
    x_test = minmax_scaler.transform(x_test)
    benign_x_train = minmax_scaler.transform(benign_x_train)
    
    return x_train, x_val, x_test, benign_x_train

def get_time(start_time, end_time):
    execution_time = end_time - start_time
    mins = execution_time // 60
    hours = mins // 60
    mins = execution_time // 60 - (hours *60)
    segs = execution_time % 60
    print("Tempo de execução:", int(hours), "horas,", int(mins), "minutos e", int(segs),"segundos")