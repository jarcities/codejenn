import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras import layers, callbacks, models, optimizers
from keras.models import load_model
from keras.activations import sigmoid
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import os

#parameters
FILE = 'example'
np.random.seed(1)
NUM_SAMPLES = 10000 #10,000
INPUT_DIM = 10
OUTPUT_DIM = 10
input = np.random.rand(NUM_SAMPLES, INPUT_DIM)
output = np.random.rand(NUM_SAMPLES, OUTPUT_DIM)
BATCH = 64
LEARNING_RATE = 0.001
EPOCHS = 50
NEURONS = 8

#define model
model = Sequential([
    layers.Input(shape=(INPUT_DIM,)),
    layers.Dense(NEURONS, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(NEURONS, activation='relu'),
    layers.Dense(OUTPUT_DIM, activation='linear')
])

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=tf.keras.losses.LogCosh()
              )

#train model
history = model.fit(input, 
                    output,
                    batch_size=BATCH,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_split=0.3,
                    )

#save model
model_filename = f"{FILE}.keras"
model.save(model_filename)

#predict model
model.summary()