import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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