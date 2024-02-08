import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers  # Import regularizers
from tensorflow.keras.layers import Dropout,Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.5),
            Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.5),
            Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001))
        ])
        self.decoder = tf.keras.Sequential([
            Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.5),
            Dense(128, activation="sigmoid", kernel_regularizer=regularizers.l2(0.001))
        ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AutoEncoder()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
model.compile(optimizer='adam', loss="mae")

def get_metrics(model, normal_test_data,anomaly_test_data):
    reconstruction = model.predict(normal_test_data)
    train_loss = tf.keras.losses.mae(reconstruction, normal_test_data)
    plt.hist(train_loss, bins=50)
    threshold = np.mean(train_loss) + 2*np.std(train_loss)
    reconstruction_a = model.predict(anomaly_test_data)
    train_loss_a = tf.keras.losses.mae(reconstruction_a, anomaly_test_data)
    preds = tf.math.less(train_loss, threshold)
    print(tf.math.count_nonzero(preds))
    preds_a = tf.math.greater(train_loss_a, threshold)
    print(tf.math.count_nonzero(preds_a))
    return