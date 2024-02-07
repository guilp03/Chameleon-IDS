import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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