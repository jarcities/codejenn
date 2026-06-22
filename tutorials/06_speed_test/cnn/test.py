import os
import warnings
import tensorflow as tf
# import torch  
warnings.filterwarnings("ignore")

import time
import numpy as np
## TORCH OR TENSORFLOW ##
#####################################
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  
os.environ["KERAS_BACKEND"] = "tensorflow"  
#####################################
import keras
import h5py

# settings
NUM_SAMPLES = 10000
INPUT_DIM_1 = 10
INPUT_DIM_2 = 100
INPUT_DIM = INPUT_DIM_1 * INPUT_DIM_2
DATA_DIR = "data"
DTYPE = np.float64
JIT_TF = False
JIT_PY = False

## TORCH OR TENSORFLOW ##
#########################################################
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
#########################################################
# torch.set_num_threads(1)
#########################################################

# load model
model = keras.models.load_model("model.keras")

# load data
batch = []
for i in range(NUM_SAMPLES):
    path = os.path.join(DATA_DIR, f"data_{i:04d}.h5")
    with h5py.File(path, "r") as f:
        x = f["x"][...]
    ## TORCH OR TENSORFLOW ##
    #########################################################
    # tensor = torch.from_numpy(np.array(x.reshape(1, INPUT_DIM_1, INPUT_DIM_2, 1), dtype=DTYPE))
    tensor = tf.convert_to_tensor(np.array(x.reshape(1, INPUT_DIM_1, INPUT_DIM_2, 1), dtype=DTYPE))
    #########################################################
    batch.append(tensor)

# eager inference
if not JIT_TF and not JIT_PY:
    # warmup
    _ = model(batch[0], training=False)
    # time loop
    start = time.perf_counter()
    for i in range(NUM_SAMPLES):
        _ = model(batch[i], training=False)
    end = time.perf_counter()

## UNCOMMENT FOR TENSORFLOW JIT ##
##########################################
elif JIT_TF:
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
##########################################


## UNCOMMENT FOR TORCH JIT ##
##########################################
# else:
#     # jit py inference
#     model.eval()  
#     model = torch.jit.trace(model, batch[0]) 
#     # model = torch.compile(model)
#     # warmup
#     _ = model(batch[0])  
#     # time loop
#     start = time.perf_counter()
#     for i in range(NUM_SAMPLES):
#         _ = model(batch[i])  
#     end = time.perf_counter()
##########################################

print(f"{(end - start):.6f} seconds!")