import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#parameters

FILE = 'cnn_3d'
np.random.seed(1)
NUM_SAMPLES = 10000  #10,000
INPUT_D, INPUT_H, INPUT_W = 16, 32, 32   # 16x32x32 volume
OUTPUT_H, OUTPUT_W = 16, 16
OUTPUT_DIM = OUTPUT_H * OUTPUT_W
BATCH = 64
LEARNING_RATE = 0.001
EPOCHS = 3
EPS = 1e-8

X = np.random.rand(NUM_SAMPLES, INPUT_D, INPUT_H, INPUT_W, 1).astype(np.float32)
y = np.random.rand(NUM_SAMPLES, OUTPUT_H, OUTPUT_W).astype(np.float32)
y = y.reshape(NUM_SAMPLES, OUTPUT_H, OUTPUT_W, 1)  # Reshape labels to 3D

X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + EPS
y_mean = y.mean(axis=0, keepdims=True) 
y_std = y.std(axis=0, keepdims=True) + EPS

X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

inputs = layers.Input(shape=(INPUT_D, INPUT_H, INPUT_W, 1))
# layer 1
x = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.AveragePooling3D(pool_size=(2, 2, 2))(x)
# layer 2
x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
# layer 3
x = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), activation='tanh', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.SpatialDropout3D(0.1)(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
# layer 4
x = layers.Conv3D(filters=96, kernel_size=(3, 3, 3), activation='swish', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.SpatialDropout3D(0.1)(x)
# layer 5 
x = layers.Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.SpatialDropout3D(0.15)(x)
# layer 6
x = layers.GlobalAveragePooling3D()(x)
# layer 7
x = layers.Dense(256)(x)
x = layers.Activation('mish')(x)
x = layers.Dropout(0.3)(x)
# layer 8
x = layers.Dense(128, activation='gelu')(x)
x = layers.Dropout(0.2)(x)
# layer 9
x = layers.Dense(OUTPUT_DIM, activation='linear')(x)
outputs = layers.Reshape((OUTPUT_H, OUTPUT_W, 1))(x) 
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
np.save("input_mean.npy", X_mean)
np.save("input_std.npy", X_std)
np.save("output_mean.npy", y_mean)
np.save("output_std.npy", y_std)

#summary
model.summary()
#trained_model = load_model(model_filename)
#prediction = trained_model.predict(X)
#plt.figure()
#plt.imshow(y[0].reshape(OUTPUT_H, OUTPUT_W), cmap='gray')
#plt.title('Actual Output')
#plt.figure()
#plt.imshow(prediction[0].reshape(OUTPUT_H, OUTPUT_W), cmap='gray')
#plt.title('Predicted Output')
#plt.show()