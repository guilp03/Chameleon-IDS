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
# history = model.fit(normal_train_data, normal_train_data, epochs=50, batch_size=120,
#                     validation_data=(train_data_scaled[:,1:], train_data_scaled[:, 1:]),
#                     shuffle=True,
#                     callbacks=[early_stopping]
#                     )
