"""Autoencoder PyTorch para detecção de anomalias de rede.

O modelo é treinado exclusivamente no tráfego benigno (classe 0).
Anomalias são detectadas pelo erro de reconstrução (MSE > threshold).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm


class EarlyStopping:
    """Interrompe o treinamento quando a loss de validação para de melhorar.

    Salva os pesos do melhor modelo em disco e os restaura ao final.

    Attributes:
        patience: Épocas toleradas sem melhora antes de parar.
        delta: Melhora mínima considerada significativa.
        verbose: Se True, imprime mensagem a cada salvamento.
        counter: Contador de épocas consecutivas sem melhora.
        early_stop: Flag ativada quando o critério de parada é atingido.
        val_min_loss: Menor loss de validação registrada até agora.
        path: Caminho do arquivo de checkpoint.
    """

    def __init__(
        self,
        patience: int = 7,
        delta: float = 0.0,
        verbose: bool = True,
        path: str = "checkpoint.pt",
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_min_loss = np.inf
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.val_min_loss - self.delta:
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}. "
                f"Current validation loss: {val_loss:.5f}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Salva o state_dict do modelo quando a loss de validação melhora.

        Args:
            val_loss: Loss de validação atual.
            model: Modelo cujo estado será salvo.
        """
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_min_loss:.5f} --> {val_loss:.5f}). "
                "Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_min_loss = val_loss


class Autoencoder(nn.Module):
    """Autoencoder para detecção de anomalias em tráfego de rede.

    Arquitetura encoder-decoder com BatchNorm e Dropout, projetada para
    os datasets NSL-KDD e CICIDS2017.

    Attributes:
        in_features: Dimensão da entrada (número de features selecionadas).
        dropout_rate: Taxa de dropout aplicada nas camadas intermediárias.
        early_stopping: Instância de ``EarlyStopping`` (configurada em ``fit``).
        encoder: Módulo sequencial de codificação.
        decoder: Módulo sequencial de decodificação.
    """

    def __init__(self, in_features: int, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.dropout_rate = dropout_rate
        self.early_stopping: EarlyStopping | None = None

        self.encoder = nn.Sequential(
            # Camada 1
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Camada 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Camada 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # Camada 4
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Camada 5
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Camada 6
            nn.Linear(128, in_features),
            nn.BatchNorm1d(in_features),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Passa ``X`` pelo encoder e decoder.

        Args:
            X: Tensor de entrada com shape ``(batch, in_features)``.

        Returns:
            Reconstrução de ``X`` com o mesmo shape.
        """
        return self.decoder(self.encoder(X))

    def compile(self, learning_rate: float, weight_decay: float = 0.001) -> None:
        """Configura critério de loss e otimizador Adam.

        Args:
            learning_rate: Taxa de aprendizado do Adam.
            weight_decay: Regularização L2 do Adam.
        """
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def fit(
        self,
        X_train: torch.Tensor,
        num_epochs: int,
        batch_size: int,
        X_val: torch.Tensor | None = None,
        patience: int | None = None,
        delta: float | None = None,
    ) -> tuple[list[float], list[float]]:
        """Treina o Autoencoder com early stopping opcional.

        Se ``X_val``, ``patience`` e ``delta`` forem fornecidos, o early
        stopping é ativado e os pesos do melhor modelo são restaurados ao fim.

        Args:
            X_train: Tensor de treinamento (apenas tráfego benigno).
            num_epochs: Número máximo de épocas.
            batch_size: Tamanho do mini-batch.
            X_val: Tensor de validação (apenas tráfego benigno).
            patience: Épocas sem melhora antes de parar.
            delta: Melhora mínima considerada.

        Returns:
            Tupla ``(train_avg_losses, val_avg_losses)`` com as perdas médias
            por época.
        """
        if X_val is not None and patience is not None and delta is not None:
            print(f"Using early stopping with patience={patience} and delta={delta}")
            self.early_stopping = EarlyStopping(patience, delta)

        train_avg_losses: list[float] = []
        val_avg_losses: list[float] = []

        for epoch in range(num_epochs):
            train_losses: list[float] = []
            self.train()
            for batch in tqdm(range(0, len(X_train), batch_size)):
                batch_X = X_train[batch : batch + batch_size]
                reconstruction = self.forward(batch_X)
                loss = self.criterion(reconstruction, batch_X)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            train_avg_loss = float(np.mean(train_losses))
            train_avg_losses.append(train_avg_loss)
            print(f"Epoch#{epoch + 1}: Train Average Loss = {train_avg_loss:.5f}")

            if self.early_stopping is not None:
                val_losses: list[float] = []
                self.eval()
                with torch.no_grad():
                    for batch in range(0, len(X_val), batch_size):
                        batch_X = X_val[batch : batch + batch_size]
                        reconstruction = self.forward(batch_X)
                        val_losses.append(self.criterion(reconstruction, batch_X).item())
                val_avg_loss = float(np.mean(val_losses))
                val_avg_losses.append(val_avg_loss)
                self.early_stopping(val_avg_loss, self)
                if self.early_stopping.early_stop:
                    print(f"Stopped by early stopping at epoch {epoch + 1}")
                    break

        if self.early_stopping is not None:
            self.load_state_dict(torch.load(self.early_stopping.path))
        self.eval()
        return train_avg_losses, val_avg_losses


def get_autoencoder_anomaly_scores(ae_model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Calcula o score de anomalia (MSE de reconstrução) para cada amostra.

    Args:
        ae_model: Autoencoder treinado.
        X: Array de amostras com shape ``(n_samples, n_features)``.

    Returns:
        Array de scores com shape ``(n_samples,)``. Valores maiores indicam
        maior probabilidade de anomalia.
    """
    X_tensor = torch.FloatTensor(X)
    reconstructed = ae_model(X_tensor).detach().numpy()
    return np.mean(np.power(X - reconstructed, 2), axis=1)


def get_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de classificação binária a partir de y_true e y_pred.

    Args:
        y_true: Labels reais (0 ou 1).
        y_pred: Labels preditos (0 ou 1).

    Returns:
        Dicionário com as chaves ``accuracy``, ``tpr``, ``fpr``,
        ``precision``, ``f1-score`` e ``recall``.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp)
    f1 = (2 * tpr * precision) / (tpr + precision)
    recall = tp / (tp + fn)
    return {
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1-score": f1,
        "recall": recall,
    }
