import pandas as pd
import dataset
import ParticleSwarmOptimization as pso
import particle_schema as part
import time
import torch
import Autoencoder as ae
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)

funct = "gb"
df = pd.read_csv("KDD-all.csv")
df.columns = df.columns.str.replace("'", "")
df, columnsName, y = dataset.preprocessing(df)

x_train, x_val_test, y_train, y_val_test =  train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test)

# Reset dos índices dos subsets
x_train = x_train.reset_index(drop=True)
x_val = x_val.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
x_val['class'] = y_val
benign_x_val= x_val[x_val['class']== 1]
benign_x_val = benign_x_val.drop(labels = 'class', axis = 1)
x_val = x_val.drop(labels = 'class', axis = 1)

minmax_scaler = MinMaxScaler()
minmax_scaler = minmax_scaler.fit(x_train)

x_train = minmax_scaler.transform(x_train)
x_val = minmax_scaler.transform(x_val)
x_test = minmax_scaler.transform(x_test)
benign_x_val = minmax_scaler.transform(benign_x_val)

benign_x_val_optimal_tensor = torch.FloatTensor(benign_x_val)
optimal_x_train_tensor = torch.FloatTensor(x_train)

BATCH_SIZE = 16
ALPHA = 0.096
PATIENCE = 10
DELTA = 0.001
NUM_EPOCHS = 1000
IN_FEATURES = x_train.shape[1]
DROPOUT_RATE = 0.5
REGULARIZER = 0.00075

start_time = time.time()
ae_model = ae.Autoencoder(IN_FEATURES, DROPOUT_RATE)
ae_model.compile(learning_rate= ALPHA, weight_decay=REGULARIZER)
train_avg_losses, val_avg_losses = ae_model.fit(optimal_x_train_tensor,
                                                NUM_EPOCHS,
                                                BATCH_SIZE,
                                                X_val = benign_x_val_optimal_tensor,
                                                patience = PATIENCE,
                                                delta = DELTA)

end_time = time.time()
dataset.get_time(start_time, end_time)

#ae.plot_train_val_losses(train_avg_losses, val_avg_losses)
val_anomaly_scores = ae.get_autoencoder_anomaly_scores(ae_model, x_val)

lista_arrays = [float(i) for i in range(1, 91)]  # Gerar números inteiros de 1 a 90
lista_arrays = [x / 100 for x in lista_arrays]
lista_f1 =[]
best_f1 = 0.0
best_thresh = 0.0
for i in lista_arrays:
    metrics = ae.get_overall_metrics(y_val, val_anomaly_scores > i)
    f1_score = metrics['f1-score']
    lista_f1.append(f1_score)
    if f1_score > best_f1:
        best_f1 = f1_score
        precision = metrics['precision']
        tpr = metrics['tpr']
        accuracy = metrics['acc']
        fpr = metrics['fpr']
        best_thresh = i

print("Melhor Threshold:", best_thresh)
print("Melhor F1-score:", best_f1)

print("metricas:", "f1_score:", best_f1, "precision:", precision, "accuracy:", accuracy, "tpr:", tpr, "fpr:", fpr)
#Tempo de execução: 0 horas, 11 minutos e 20 segundos
#Melhor Threshold: 0.08
#metricas: f1_score: 0.878349738614421 precision: 0.8869970512698564 accuracy: 0.8840560193913277 tpr: 0.8698694029850746 fpr: 0.10278594912614639