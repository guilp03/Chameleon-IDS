import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import random

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
        if val_loss < self.val_min_loss - self.delta:
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}. Current validation loss: {val_loss:.5f}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_min_loss:.5f} --> {val_loss:.5f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_min_loss = val_loss

class Autoencoder(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5):
        super().__init__()
        self.in_features = in_features
        self.dropout_rate = dropout_rate
        self.early_stopping = None

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, in_features),
            nn.BatchNorm1d(in_features),
            nn.Sigmoid()
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded

    def compile(self, learning_rate, weight_decay=0.001):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def fit(self, X_train, num_epochs, batch_size, X_val=None, patience=None, delta=None):
        if X_val is not None and patience is not None and delta is not None:
            print(f'Using early stopping with patience={patience} and delta={delta}')
            self.early_stopping = EarlyStopping(patience, delta)

        val_avg_losses = []
        train_avg_losses = []

        for epoch in range(num_epochs):
            train_losses = []
            self.train()
            for batch in tqdm(range(0, len(X_train), batch_size)):
                batch_X = X_train[batch:(batch + batch_size)]
                if batch_X.shape[0] == 1:
                    continue  # Skip batches with only one sample
                batch_reconstruction = self.forward(batch_X)

                train_loss = self.criterion(batch_reconstruction, batch_X)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.item())
            train_avg_loss = np.mean(train_losses)
            train_avg_losses.append(train_avg_loss)
            print(f'Epoch#{epoch + 1}: Train Average Loss = {train_avg_loss:.5f}')

            if self.early_stopping is not None:
                val_losses = []
                self.eval()
                with torch.no_grad():
                    for batch in range(0, len(X_val), batch_size):
                        batch_X = X_val[batch:(batch + batch_size)]
                        if batch_X.shape[0] == 1:
                            continue  # Skip batches with only one sample
                        batch_reconstruction = self.forward(batch_X)
                        val_loss = self.criterion(batch_reconstruction, batch_X)
                        val_losses.append(val_loss.item())
                val_avg_loss = np.mean(val_losses)
                val_avg_losses.append(val_avg_loss)
                self.early_stopping(val_avg_loss, self)
                if self.early_stopping.early_stop:
                    print(f'Stopped by early stopping at epoch {epoch + 1}')
                    break

        if self.early_stopping is not None:
            self.load_state_dict(torch.load(self.early_stopping.path))
        self.eval()
        return train_avg_losses, val_avg_losses

def get_autoencoder_anomaly_scores(ae_model, X):
    X_tensor = torch.FloatTensor(X)
    reconstructed_X = ae_model(X_tensor)
    reconstructed_X_np = reconstructed_X.detach().numpy()
    anomaly_scores = np.mean(np.power(X - reconstructed_X_np, 2), axis=1)
    return anomaly_scores

def get_overall_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp)
    f1 = (2 * tpr * precision) / (tpr + precision)
    recall = tp / (tp + fn)
    return {'accuracy': accuracy, 'tpr': tpr, 'fpr': fpr, 'precision': precision, 'f1-score': f1, 'recall': recall}

def optimize_autoencoder_hyperparameters(IN_FEATURES, x_train_tensor, x_val_tensor_benign, x_val, y_val):
    hyperparameter_ranges = {
        'BATCH_SIZE': [16,32, 64],
        'ALPHA': [1e-2, 1e-1],
        'PATIENCE': [10],
        'DELTA': [0.0001],
        'NUM_EPOCHS': [1000],
        'DROPOUT_RATE': [0.5],
        'REGULARIZER': [1e-3, 1e-2]
    }

    num_iterations = 12
    best_hyperparameters = {}
    best_f1_score = float('-inf')

    for m in range(num_iterations):
        print("Iteration:", m)
        hyperparameters = {param: random.choice(values) for param, values in hyperparameter_ranges.items()}

        ae_model = Autoencoder(IN_FEATURES, hyperparameters['DROPOUT_RATE'])
        ae_model.compile(learning_rate=hyperparameters['ALPHA'], weight_decay=hyperparameters['REGULARIZER'])

        train_avg_losses, val_avg_losses = ae_model.fit(X_train=x_train_tensor,
                                                        num_epochs=hyperparameters['NUM_EPOCHS'],
                                                        batch_size=hyperparameters['BATCH_SIZE'],
                                                        X_val=x_val_tensor_benign,
                                                        patience=hyperparameters['PATIENCE'],
                                                        delta=hyperparameters['DELTA'])

        val_anomaly_scores = get_autoencoder_anomaly_scores(ae_model, x_val)

        thresholds = [x / 1000 for x in range(1, 5000)]
        best_f1 = 0.0
        best_thresh = 0.0
        for thresh in thresholds:
            metrics = get_overall_metrics(y_val, val_anomaly_scores > thresh)
            f1_score = metrics['f1-score']
            if f1_score > best_f1 and metrics['tpr'] > 0.0:
                best_f1 = f1_score
                best_thresh = thresh
                precision = metrics['precision']
                accuracy = metrics['accuracy']
                tpr = metrics['tpr']
                fpr = metrics['fpr']
                recall = metrics['recall']

        print("Best Threshold:", best_thresh)
        print("Best F1-score:", best_f1)

        if best_f1 > best_f1_score:
            best_threshold = best_thresh
            best_precision = precision
            best_accuracy = accuracy
            best_tpr = tpr
            best_fpr = fpr
            best_recall = recall
            best_f1_score = best_f1
            best_hyperparameters = hyperparameters

    return best_threshold, best_hyperparameters, best_f1_score, best_fpr, best_tpr, best_accuracy, best_precision, best_recall
