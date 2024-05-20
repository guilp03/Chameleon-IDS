import pandas as pd
from sklearn.model_selection import train_test_split


df1 = pd.read_csv("TOW_train.csv")
df2 = pd.read_csv('TOW_test.csv')


# Concatenar os datasets verticalmente (empilhando as linhas)
df_concatenado = pd.concat([df1, df2], axis=0)
fracao = 0.02

df_reduzido, _ = train_test_split(df_concatenado, stratify=df_concatenado['class'], test_size=(1 - fracao), random_state=42)
df_reduzido.to_csv('TOW_2_percent.csv', index=False)