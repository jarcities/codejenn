#!/bin/bash
BIT=32
BACKEND=tensorflow
OPT_FLAGS="-O3 -ffast-math -fopenmp"
ARCH_FLAGS="-march=native"
export OMP_NUM_THREADS=1
export KERAS_BACKEND=$BACKEND

python3 -B train.py --backend="$BACKEND" --bit=$BIT || exit 1
python3 -B \
    ../../../src/api-core/main.py \
    --input="." \
    --output="." \
    --backend="$BACKEND" \
    --bit=$BIT \
    --model_image \
    --debug || exit 1
rm -rf .vscode/ ../../../src/api-core/__pycache__ ../../../src/dump_model/__pycache__
g++-15 DEBUG_model.cpp $OPT_FLAGS $ARCH_FLAGS -o a.out || exit 1
./a.out > DEBUG_cpp.txt || exit 1
python3 -B DEBUG_model.py > DEBUG_keras.txt || exit 1
python3 - <<'EOF'
import re
import sys
import numpy as np

cpp_output = open("DEBUG_cpp.txt").read()
keras_output = open("DEBUG_keras.txt").read()

cpp_layers = {}
for name, index, values in re.findall(
    r"\((.+?)\) layer (\d+):\nShape -> .*\nValues -> (.*)", cpp_output
):
    values = values.replace(". . .", "").split(",")
    cpp_layers[int(index)] = (name, np.array([float(v) for v in values if v.strip()]))

keras_layers = {}
for name, index, values in re.findall(
    r"\((.+?)\) Layer (\d+):\nValues -> \[(.*?)\]", keras_output, re.DOTALL
):
    keras_layers[int(index) + 1] = (name, np.array([float(v) for v in values.split()]))

if not cpp_layers or not keras_layers:
    sys.exit("__Error__ -> no layer outputs found, was the model code generated with --debug?")

labels = {index: f"layer {index} ({keras_layers[index][0]})" for index in keras_layers}
width = max(len(label) for label in labels.values())

print("\nerrors of all values of each layer:")
print('='*75)
for index in sorted(keras_layers):
    keras_values = keras_layers[index][1]
    if index not in cpp_layers:
        print(f"{labels[index]:<{width}} -> not printed by c++")
        continue
    cpp_values = cpp_layers[index][1]
    if len(cpp_values) != len(keras_values):
        print(f"{labels[index]:<{width}} -> size mismatch, c++ {len(cpp_values)} values, keras {len(keras_values)} values")
        continue
    difference = np.abs(cpp_values - keras_values)
    max_abs_error = np.max(difference)
    mean_abs_error = np.mean(difference)
    keras_norm = np.linalg.norm(keras_values)
    relative_error = np.linalg.norm(difference) / keras_norm if keras_norm > 0 else np.linalg.norm(difference)
    print(
        f"{labels[index]:<{width}} -> max abs err = {max_abs_error:.2e}, "
        f"mean abs err = {mean_abs_error:.2e}, rel err = {relative_error:.2e}"
    )
EOF
