python3 cnn_3d.py
python3 \
    ../../src/api-core/main.py \
    --input="." \
    --output="." \
    --backend="tensorflow" \
    --bit=32 \
    --debug \
    # --model_image
clang++ DEBUG_cnn_3d.cpp
./a.out > DEBUG_cpp.txt
python3 DEBUG_cnn_3d.py > DEBUG_keras.txt
python3 - <<'EOF'
import re
import sys
import numpy as np

cpp_output = open("DEBUG_cpp.txt").read()
keras_output = open("DEBUG_keras.txt").read()

#c++ layers start at 1, a layer printed more than once keeps its last print
cpp_layers = {}
for name, index, values in re.findall(
    r"\((.+?)\) layer (\d+):\nShape -> .*\nValues -> (.*)", cpp_output
):
    values = values.replace(". . .", "").split(",")
    cpp_layers[int(index)] = (name, np.array([float(v) for v in values if v.strip()]))

#keras layers start at 0
keras_layers = {}
for name, index, values in re.findall(
    r"\((.+?)\) Layer (\d+):\nValues -> \[(.*?)\]", keras_output, re.DOTALL
):
    keras_layers[int(index) + 1] = (name, np.array([float(v) for v in values.split()]))

if not cpp_layers or not keras_layers:
    sys.exit("__Error__ -> no layer outputs found, was the model code generated with --debug?")

#final output of the model
output_index = max(keras_layers) + 1
cpp_final = re.search(r"Output:\n(.*)", cpp_output, re.DOTALL)
keras_final = re.search(r"Output:\n(.*)", keras_output, re.DOTALL)
if cpp_final and keras_final:
    cpp_layers[output_index] = ("output", np.array([float(v) for v in cpp_final.group(1).split()]))
    keras_layers[output_index] = ("output", np.array([float(v) for v in keras_final.group(1).split()]))

#errors of all values of each layer
labels = {
    index: "model output" if index == output_index else f"layer {index} ({keras_layers[index][0]})"
    for index in keras_layers
}
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
rm -rf .vscode/ api-core/__pycache__ dump_model/__pycache__