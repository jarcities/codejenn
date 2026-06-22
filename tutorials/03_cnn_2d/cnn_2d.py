import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#parameters
FILE = 'cnn_2d'
np.random.seed(1)
NUM_SAMPLES = 10000  #10,000
INPUT_H, INPUT_W = 32, 32   #32x32 pixel image
OUTPUT_DIM = 10
BATCH = 64
LEARNING_RATE = 0.001
EPOCHS = 10
EPS = 1e-8

#input
X = np.random.rand(NUM_SAMPLES, INPUT_H, INPUT_W, 1).astype(np.float32)
y = np.random.rand(NUM_SAMPLES, OUTPUT_DIM).astype(np.float32)

#normalize data (per spatial position/channel across the batch)
X_min = X.min(axis=0, keepdims=True)
X_max = X.max(axis=0, keepdims=True)
y_min = y.min(axis=0) + EPS
y_max = y.max(axis=0) + EPS

X = (X - X_min) / (X_max - X_min + EPS)
y = (y - y_min) / (y_max - y_min + EPS)

#build model
inputs = layers.Input(shape=(INPUT_H, INPUT_W, 1))
#layer 1
x = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='sigmoid', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
#layer 2
x = layers.DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
#layer 3
x = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)
x = layers.Activation('silu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#layer 4
x = layers.SeparableConv2D(filters=96, kernel_size=(3, 3), activation='swish', padding='same')(x)
x = layers.BatchNormalization()(x)
#layer 5
x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='tanh', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#layer 6
x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='softmax', padding='same')(x)
x = layers.BatchNormalization()(x)
#layer 7
x = layers.GlobalAveragePooling2D()(x)
#layer 8
x = layers.Dense(256)(x)
x = layers.Activation('mish')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='gelu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(OUTPUT_DIM, activation='linear')(x)
model = Model(inputs=inputs, outputs=outputs)

#compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.LogCosh()
)

early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=25, min_lr=1e-7)

#train
history = model.fit(
    X, y,
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
np.save("input_min.npy", X_min)
np.save("input_max.npy", X_max)
np.save("output_min.npy", y_min)
np.save("output_max.npy", y_max)

#summary (and optional predict/plot)
model.summary()
#trained_model = load_model(model_filename)
#prediction = trained_model.predict(X)
#plt.figure()
#plt.plot(y[:, 0], '-k', label='actual Output')
#plt.plot(prediction[:, 0], '--r', label='predicted Output')
#plt.legend()
#plt.show()