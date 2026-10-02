import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    type=str,
    required=True,
    choices=["tensorflow", "torch"],
)
parser.add_argument("--jit", action="store_true")
args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = args.backend
if args.backend == "tensorflow":
    import tensorflow as tf
else:
    import torch
import keras
import h5py

# settings
NUM_SAMPLES = 10000
INPUT_DIM = 1000
DATA_DIR = "data"
DTYPE = np.float64

if args.backend == "tensorflow":
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
else:
    torch.set_num_threads(1)

# load model
with keras.device("cpu"):
    model = keras.models.load_model("model.keras")

# load data
batch = []
for i in range(NUM_SAMPLES):
    path = os.path.join(DATA_DIR, f"data_{i:04d}.h5")
    with h5py.File(path, "r") as f:
        x = f["x"][...]
    x = np.array(x.reshape(1, INPUT_DIM), dtype=DTYPE)
    if args.backend == "tensorflow":
        tensor = tf.constant(x, dtype=tf.float64)
    else:
        tensor = torch.from_numpy(x)
    batch.append(tensor)

with keras.device("cpu"):
    # eager inference
    if not args.jit:
        # warmup
        _ = model(batch[0], training=False)
        # time loop
        start = time.perf_counter()
        for i in range(NUM_SAMPLES):
            _ = model(batch[i], training=False)
        end = time.perf_counter()

    elif args.backend == "tensorflow":
        # jit xla inference
        @tf.function(jit_compile=True)
        def run_one(x):
            return model(x, training=False)

        # warmup compile
        _ = run_one(batch[0])
        # time loop
        start = time.perf_counter()
        for i in range(NUM_SAMPLES):
            _ = run_one(batch[i])
        end = time.perf_counter()

    else:
        # jit py inference
        model.eval()
        model = torch.jit.trace(model, batch[0])
        # model = torch.compile(model)

        # warmup
        _ = model(batch[0])
        # time loop
        start = time.perf_counter()
        for i in range(NUM_SAMPLES):
            _ = model(batch[i])
        end = time.perf_counter()

print(f"{(end - start):.6f} seconds!")
