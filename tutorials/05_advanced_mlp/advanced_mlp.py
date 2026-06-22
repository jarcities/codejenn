import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import Sequential
from keras import layers, callbacks, models, optimizers
from keras.models import load_model
from keras.activations import sigmoid
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2


#parameters
FILE = 'advanced_mlp'
np.random.seed(1)
NUM_SAMPLES = 10000 #10,000
INPUT_DIM = 500
OUTPUT_DIM = 100
input = np.random.rand(NUM_SAMPLES, INPUT_DIM)
output = np.random.rand(NUM_SAMPLES, OUTPUT_DIM)
BATCH = 64
LEARNING_RATE = 0.001
EPOCHS = 10
NEURONS = 12

# #normalize data
# X_mean = input.mean(axis=0)
# X_std = input.std(axis=0)
# y_mean = output.mean(axis=0)
# y_std = output.std(axis=0)
# input = (input - X_mean) / X_std
# output = (output - y_mean) / y_std

#custom activation
#https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
def custom_activation(x):
    return (keras.activations.sigmoid(x) * 5) - 1

#define model
model = Sequential([
    layers.Input(shape=(INPUT_DIM,)),
    layers.Dense(NEURONS, activation='softmax'),
    # layers.SpatialDropout1D(0.3),
    layers.Dense(NEURONS, activation='swish'),
    layers.GroupNormalization(groups=4, axis=-1),
    layers.Dense(NEURONS, activation='tanh'),
    layers.BatchNormalization(),
    layers.Dense(NEURONS, activation='elu'),
    layers.Dropout(0.1),
    layers.Dense(NEURONS),
    layers.LayerNormalization(),
    layers.Dense(OUTPUT_DIM, activation='linear'),
    layers.Rescaling(scale=1.0),
    layers.Activation(custom_activation, name="custom_activation"),
    layers.LeakyReLU(negative_slope=0.5)
])
"""
IT IS CONVENTION TO TYPICALLY USE THE SAME ACTIVATION FUNCTION
FOR THE ENTIRE MODEL, BUT THIS IS JUST AN EXAMPLE.
"""

#compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=keras.losses.MeanSquaredError()
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
model_filename = f"{FILE}.h5"
model.save(model_filename)

# #save normalization
# np.save("input_mean.npy", X_mean)
# np.save("input_std.npy", X_std)
# np.save("output_mean.npy", y_mean)
# np.save("output_std.npy", y_std)

#predict model
model.summary()
# trained_model = load_model(model_filename)
# prediction = trained_model.predict(input)
# plt.figure()
# plt.plot(output[:, 0], '-k', label=f'actual Output')
# plt.plot(prediction[:, 0], '--r', label=f'predicted Output')
# plt.legend()
# plt.show()