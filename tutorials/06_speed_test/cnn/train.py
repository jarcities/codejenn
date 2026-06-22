import os
import sys
import time
import random
import warnings
import math
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

def main():
    # mlp stuff
    NUM_SAMPLES = 10000
    INPUT_DIM_1 = 10
    INPUT_DIM_2 = 100
    INPUT_DIM = INPUT_DIM_1 * INPUT_DIM_2
    OUTPUT_DIM = 100
    SPLIT = 0.2
    LR = 1e-4
    EPOCHS = 10
    BS = 64
    BIT = np.float64

    # seed stuff
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
    X = np.random.randn(NUM_SAMPLES, INPUT_DIM).astype(BIT)
    y = np.random.randint(0, OUTPUT_DIM, NUM_SAMPLES).astype(int)

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=SPLIT, random_state=SEED
    )

    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = X_train.reshape(-1, INPUT_DIM_1, INPUT_DIM_2, 1)
    X_val = X_val.reshape(-1, INPUT_DIM_1, INPUT_DIM_2, 1)

    # #SMALLER MODEL
    # model = keras.Sequential([
    #     keras.layers.Input(shape=(INPUT_DIM_1, INPUT_DIM_2, 1)),
    #     keras.layers.Conv2D(8, (3,3), activation='softplus'),
    #     keras.layers.MaxPooling2D((2,2)),
    #     keras.layers.Conv2D(16, (3,3), activation='tanh'),
    #     keras.layers.Conv2DTranspose(1, (3,3), activation='relu'),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(32, activation='mish'),
    #     keras.layers.Dense(OUTPUT_DIM, activation='softmax'),
    # ])

    #BIGGER MODEL (UNBOUNDED MEMORY)
    model = keras.Sequential([
        keras.layers.Input(shape=(INPUT_DIM_1, INPUT_DIM_2, 1)),
        keras.layers.Conv2D(16, (3,3), activation='softplus'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(32, (3,3), activation='tanh'),
        keras.layers.Conv2DTranspose(32, (3,3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='mish'),
        keras.layers.Dense(OUTPUT_DIM, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=10)
    ]

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_val, y_val)

    preds = np.argmax(model.predict(X_val), axis=1)

    print("Accuracy:", acc)
    print(sklearn.metrics.classification_report(y_val, preds))

    model.save("model.keras")

if __name__ == "__main__":
    main()
