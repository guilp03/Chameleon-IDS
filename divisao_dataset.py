from sklearn.preprocessing import train_test_split

def divisao_dataset(df, label_class, frac_train, frac_test): # O que sobrar das frações é dos dados de validação
    # Divisão do conjunto de treino validação e teste
    # Dividindo a database em % para treinamento e % para validacao e testes
    columnsName = df.drop(labels= label_class, axis= 1).columns.values.tolist()
    y = df[label_class] # Labels

    x_train, x_val_test, y_train, y_val_test =  train_test_split(df[columnsName], y, test_size=(1-frac_train), random_state=42, stratify=df['Label'])

    # Reset dos índices dos subsets
    x_train = x_train.reset_index(drop=True)
    x_val_test = x_val_test.reset_index(drop=True)

    # Dividindo o subset de validação + teste em subset de validação e subset de testes
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=frac_test, stratify=y_val_test, random_state=33)
  
    # Reset dos índices dos subsets
    x_val, x_test = x_val.reset_index(drop=True), x_test.reset_index(drop=True)
    y_val, y_test =  y_val.reset_index(drop=True), y_test.reset_index(drop=True)

    del x_val_test

    return x_train, y_train, x_val, y_val, x_test, y_test