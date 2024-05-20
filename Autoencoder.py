import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import random

#CLASSE QUE DEFINE O EARLY STOP, BASEADA NO CÓDIGO APRESENTADO NA AULA DE AUTOENCODERS
class EarlyStopping:
  def __init__(self, patience=7, delta=0, verbose=True, path='checkpoint.pt'):
      self.patience = patience
      self.delta = delta
      self.verbose = verbose
      self.counter = 0
      self.early_stop = False
      self.val_min_loss = np.Inf
      self.path = path

  def __call__(self, val_loss, model):
    if val_loss < self.val_min_loss - self.delta:   # Caso a loss da validação reduza, vamos salvar o modelo e nova loss mínima
      self.save_checkpoint(val_loss, model)
      self.counter = 0
    else:                                           # Caso a loss da validação NÃO reduza, vamos incrementar o contador da paciencia
      self.counter += 1
      print(f'EarlyStopping counter: {self.counter} out of {self.patience}. Current validation loss: {val_loss:.5f}')
      if self.counter >= self.patience:
          self.early_stop = True

  def save_checkpoint(self, val_loss, model):
    if self.verbose:
        print(f'Validation loss decreased ({self.val_min_loss:.5f} --> {val_loss:.5f}).  Saving model ...')
    torch.save(model, self.path)
    self.val_min_loss = val_loss
#CLASSE DO AUTOENCODER EM USA ARQUITETURA UTILIZADA NO NSL-KDD E CICS_IDS-2017 
class Autoencoder(nn.Module):
  
  def __init__(self, in_features, dropout_rate=0.5):
    super().__init__()

    self.in_features = in_features
    self.dropout_rate = dropout_rate
    self.early_stopping = None
    self.encoder = nn.Sequential(
        # Camada 1:
      nn.Linear(in_features, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      # Camada 2:
      nn.Linear(128, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      # Camade 3:
      nn.Linear(64, 32),
      nn.BatchNorm1d(32),
      nn.ReLU()
    )

    self.decoder = nn.Sequential(
         # Camada 4: 
        nn.Linear(32, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        #Camada 5:
        nn.Linear(64,128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        # Camada 6 de decoding:
        nn.Linear(128, in_features),
        nn.BatchNorm1d(in_features),
        nn.Sigmoid()
    )

  def forward(self, X):
    # Processar a entrada sequencialmente
    encoded = self.encoder(X)
    decoded = self.decoder(encoded)
    return decoded

  def compile(self, learning_rate, weight_decay = 0.001):
    #MSELOSS FOI USADA EM TODOS OS AUTOENCODERS COM REGULARIZAÇÃO L2
    self.criterion = nn.MSELoss()
    self.optimizer = optim.Adam(self.parameters(), lr = learning_rate, weight_decay=weight_decay)

  def fit(self, X_train, num_epochs, batch_size, X_val = None, patience = None, delta = None):
    if X_val is not None and patience is not None and delta is not None:
      print(f'Using early stopping with patience={patience} and delta={delta}')
      self.early_stopping = EarlyStopping(patience, delta)

    val_avg_losses = []
    train_avg_losses = []

    for epoch in range(num_epochs):
      # Calibrando os pesos do modelo
      train_losses = []
      self.train()
      for batch in tqdm(range(0, len(X_train), batch_size)):
        batch_X = X_train[batch:(batch+batch_size)]
        batch_reconstruction = self.forward(batch_X)

        train_loss = self.criterion(batch_reconstruction, batch_X)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        train_losses.append(train_loss.item())
      train_avg_loss = np.mean(train_losses)
      train_avg_losses.append(train_avg_loss)
      print(f'Epoch#{epoch+1}: Train Average Loss = {train_avg_loss:.5f}')

      # Mecanismo de early stopping
      if self.early_stopping is not None:
        val_losses = []
        self.eval()
        with torch.no_grad():
          for batch in range(0, len(X_val), batch_size):
            batch_X = X_val[batch:(batch+batch_size)]
            batch_reconstruction = self.forward(batch_X)
            val_loss = self.criterion(batch_reconstruction, batch_X)
            val_losses.append(val_loss.item())
        val_avg_loss = np.mean(val_losses)
        val_avg_losses.append(val_avg_loss)
        self.early_stopping(val_avg_loss, self)
        if self.early_stopping.early_stop:
          print(f'Stopped by early stopping at epoch {epoch+1}')
          break

    if self.early_stopping is not None:
     # self = torch.load('checkpoint.pt')
     pass
    self.eval()
    return train_avg_losses, val_avg_losses
  
def get_autoencoder_anomaly_scores(ae_model, X):
    X_tensor = torch.FloatTensor(X)
    reconstructed_X = ae_model(X_tensor)
    
    # Converter o tensor reconstruído para um array NumPy
    reconstructed_X_np = reconstructed_X.detach().numpy()
    
    # Calcular os escores de anomalia usando MSELoss
    anomaly_scores = np.mean(np.power(X - reconstructed_X_np, 2), axis=1)
    
    return anomaly_scores


def get_overall_metrics(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  tpr = tp/(tp+fn)
  fpr = fp/(fp+tn)
  precision = tp/(tp+fp)
  f1 = (2*tpr*precision)/(tpr+precision)
  recall = tp/(tp+fn)
  return {'accuracy':accuracy,'tpr':tpr,'fpr':fpr,'precision':precision,'f1-score':f1, 'recall':recall}

def optimize_autoencoder_hyperparameters(IN_FEATURES, x_train_tensor, x_val_tensor_benign, x_val, y_val ):
    # Definir intervalos para os hiperparâmetros
    hyperparameter_ranges = {
        'BATCH_SIZE': [16],
        'ALPHA': [1e-3,1e-2, 1e-1],
        'PATIENCE': [10],
        'DELTA': [0.0001],
        'NUM_EPOCHS': [1000],
        'DROPOUT_RATE': [0.5],
        'REGULARIZER': [1e-4, 1e-3, 1e-2]
    }

    # Definir o número de iterações da busca aleatória
    num_iterations = 20

    # Melhores hiperparâmetros e seu desempenho
    best_hyperparameters = {}
    best_f1_score = float('-inf')

    for m in range(num_iterations):
        print("iteração:", m)
        # Amostrar hiperparâmetros aleatoriamente dentro dos intervalos especificados
        hyperparameters = {param: random.choice(values) for param, values in hyperparameter_ranges.items()}

        # Configurar e treinar o modelo com os hiperparâmetros amostrados
        ae_model = Autoencoder(IN_FEATURES, hyperparameters['DROPOUT_RATE'])
        ae_model.compile(learning_rate=hyperparameters['ALPHA'], weight_decay=hyperparameters['REGULARIZER'])

        train_avg_losses, val_avg_losses = ae_model.fit(x_train_tensor,
                                                        hyperparameters['NUM_EPOCHS'],
                                                        hyperparameters['BATCH_SIZE'],
                                                        X_val=x_val_tensor_benign,
                                                        patience=hyperparameters['PATIENCE'],
                                                        delta=hyperparameters['DELTA'])

        # Avaliar o desempenho do modelo usando o F1-score
        val_anomaly_scores = get_autoencoder_anomaly_scores(ae_model, x_val)

        lista_arrays = [float(i) for i in range(1, 5000)]  # Gerar números inteiros de 1 a 90
        lista_arrays = [x / 1000 for x in lista_arrays]
        lista_f1 = []
        best_f1 = 0.0
        best_thresh = 0.0   
        for i in lista_arrays:
            metrics = get_overall_metrics(y_val, val_anomaly_scores > i)
            f1_score = metrics['f1-score']
            precision = metrics['precision']
            accuracy = metrics['accuracy']
            tpr = metrics['tpr']
            fpr = metrics['fpr']
            recall = metrics['recall']
            lista_f1.append(f1_score)
            if f1_score > best_f1 and tpr > 0.0:
                best_f1 = f1_score
                best_thresh = i
        print("Melhor Threshold:", best_thresh)
        print("Melhor F1-score:", best_f1)


        # Atualizar os melhores hiperparâmetros se o desempenho atual for melhor
        if best_f1 > best_f1_score:
            best_threshold = best_thresh
            best_precision = precision
            best_accuracy = accuracy
            best_tpr = tpr
            best_fpr = fpr
            best_recall = recall
            best_f1_score = best_f1
            best_hyperparameters = hyperparameters

    return best_threshold,best_hyperparameters, best_f1_score,best_fpr,best_tpr,best_accuracy,best_precision, best_recall