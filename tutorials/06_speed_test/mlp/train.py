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
os.environ["KERAS_BACKEND"] = "torch"
import keras

def main():
    # mlp stuff
    NUM_SAMPLES = 10000
    INPUT_DIM = 1000
    OUTPUT_DIM = 100
    SPLIT = 0.2
    LR = 1e-4
    EPOCHS = 25
    BS = 64
    BIT = np.float64

    # seed stuff
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
    X = np.random.randn(NUM_SAMPLES, INPUT_DIM).astype(BIT)
    y = np.random.randint(0, 2, (NUM_SAMPLES, OUTPUT_DIM)).astype(BIT)

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=SPLIT, random_state=SEED
    )

    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(INPUT_DIM,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(OUTPUT_DIM, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
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

    preds = (model.predict(X_val) > 0.5).astype(int)

    print("Accuracy:", acc)
    print(sklearn.metrics.classification_report(y_val, preds))

    model.save("model.keras")

if __name__ == "__main__":
    main()
