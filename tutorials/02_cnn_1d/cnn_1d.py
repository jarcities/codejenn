import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import Sequential, Model
from keras import layers, callbacks, models, optimizers
from keras.models import load_model
from keras.activations import sigmoid
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

#parameters
FILE = 'cnn_1d'
np.random.seed(1)
NUM_SAMPLES = 10000 #10,000
INPUT_DIM = 100  
OUTPUT_DIM = 10
input = np.random.rand(NUM_SAMPLES, INPUT_DIM, 1) 
output = np.random.rand(NUM_SAMPLES, OUTPUT_DIM)
BATCH = 64
LEARNING_RATE = 0.001
EPOCHS = 10
EPS = 1e-8

#normalize data
X_min = input.min(axis=0, keepdims=True)  
X_max = input.max(axis=0, keepdims=True)  
y_min = output.min(axis=0) + EPS
y_max = output.max(axis=0) + EPS
input = (input - X_min) / (X_max - X_min)
output = (output - y_min) / (y_max - y_min)

#define model
inputs = layers.Input(shape=(INPUT_DIM, 1))
#layer 1
x = layers.Conv1D(filters=32, kernel_size=5, activation='sigmoid', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.AveragePooling1D(pool_size=2)(x)
#layer 2
x = layers.DepthwiseConv1D(kernel_size=3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
#layer 3
x = layers.Conv1D(filters=64, kernel_size=5, padding='same')(x)
x = layers.Activation('silu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
#layer 4
x = layers.SeparableConv1D(filters=96, kernel_size=3, activation='swish', padding='same')(x)
x = layers.BatchNormalization()(x)
#layer 5
x = layers.Conv1D(filters=128, kernel_size=3, activation='tanh', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
#layer 6
x = layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, activation='softmax', padding='same')(x)
# x = layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(x)
# x = layers.Activation('softmax')(x)
x = layers.BatchNormalization()(x)
#layer 7
x = layers.GlobalAveragePooling1D()(x)
#layer 8
x = layers.Dense(256)(x)
x = layers.Activation('mish')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='gelu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(OUTPUT_DIM, activation='linear')(x)
#create the model
model = Model(inputs=inputs, outputs=outputs)
"""
IT IS CONVENTION TO TYPICALLY USE THE SAME ACTIVATION FUNCTION
FOR THE ENTIRE MODEL, BUT THIS IS JUST AN EXAMPLE.
"""

#compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=keras.losses.LogCosh()
              )
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=50, 
                               restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.25, 
                              patience=25, 
                              min_lr=1e-7)

#train model
history = model.fit(input, 
                    output,
                    batch_size=BATCH,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_split=0.15,
                    callbacks=[early_stopping, reduce_lr]
                    )

#save model
model_filename = f"{FILE}.keras"
model.save(model_filename)

#save normalization
np.save(f"input_min.npy", X_min)
np.save(f"input_max.npy", X_max)
np.save(f"output_min.npy", y_min)
np.save(f"output_max.npy", y_max)

# #predict model
model.summary()
# trained_model = load_model(model_filename)
# prediction = trained_model.predict(input)
# plt.figure()
# plt.plot(output[:, 0], '-k', label=f'actual Output')
# plt.plot(prediction[:, 0], '--r', label=f'predicted Output')
# plt.legend()
# plt.show()