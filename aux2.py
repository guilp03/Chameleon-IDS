import pandas as pd
from sklearn.model_selection import train_test_split


df1 = pd.read_csv("TOW_test.csv")



# Concatenar os datasets verticalmente (empilhando as linhas)
df1['class'] = pd.read_csv("TOW_y_test.csv")['class']
df1['class'] = df1['class'].apply(lambda c: 0 if c == 'Normal' else 1)

print(df1)
# Salvar o dataset combinado
df1.to_csv('TOW_test.csv', index=False)