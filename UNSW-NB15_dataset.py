import pandas as pd

# Features do dataset
features = pd.read_csv(r'csv_NUSW-NB15\NUSW-NB15_features.csv', encoding='ISO-8859-1')
print(features)

# Mapeamento dos tipos de dados
type_mapping = {
    'integer': 'int',
    'Float': 'float',
    'nominal': 'object',
    'binary': 'bool',  # Ou 'object' se representar strings de bits
    'Binary': 'bool',  # Ou 'object' se representar strings de bits
    'Timestamp': 'datetime64'
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