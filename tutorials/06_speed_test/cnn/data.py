import os
import numpy as np
import h5py

NUM_SAMPLES = 10000
INPUT_DIM = 1000
OUT_DIR = "data"
BIT = np.float64
SEED = 45

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for i in range(NUM_SAMPLES):
        x = rng.standard_normal(INPUT_DIM).astype(BIT)
        path = os.path.join(OUT_DIR, f"data_{i:04d}.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("x", data=x)  # store as 1d


if __name__ == "__main__":
    print(f"\nCreating {NUM_SAMPLES} data samples of size {INPUT_DIM}")
    main()
