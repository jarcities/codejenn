"""
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""


def cppTestCode(precision_type, base_file_name, layer_shape):

    input_code = "\n"
    output_code = "\n"
    input_shape = layer_shape[0]
    input_shape = tuple(dim for dim in input_shape if dim != 1)
    output_shape = tuple(layer_shape[-1])

    if len(input_shape) == 0:
        raise ValueError("Unsupported input shape")
    if len(output_shape) == 0:
        raise ValueError("Unsupported output shape")

    input_type = "Scalar"
    for dim in reversed(input_shape):
        input_type = f"std::array<{input_type}, {dim}>"
    loop_vars = [f"i{n}" for n in range(len(input_shape))]
    input_code += f"\t{input_type} input;\n\n    int val = 0;\n"
    for n, (loop_var, dim) in enumerate(zip(loop_vars, input_shape)):
        input_code += (
            "    " * (n + 1)
            + f"for (int {loop_var} = 0; {loop_var} < {dim}; ++{loop_var}) {{\n"
        )
    indent = "    " * (len(input_shape) + 1)
    input_code += (
        f"{indent}input{''.join(f'[{v}]' for v in loop_vars)} = static_cast<Scalar>(val);\n"
        f"{indent}++val;\n"
    )
    for n in range(len(input_shape), 0, -1):
        input_code += "    " * n + "}\n"

    loop_vars = [f"i{n}" for n in range(len(output_shape))]
    separator = "'\\n'" if len(output_shape) == 1 else "' '"
    for n, (loop_var, dim) in enumerate(zip(loop_vars, output_shape)):
        output_code += (
            "\t"
            + "    " * n
            + f"for (int {loop_var} = 0; {loop_var} < {dim}; ++{loop_var}) {{\n"
        )
    output_code += (
        "\t"
        + "    " * len(output_shape)
        + f"std::cout << output{''.join(f'[{v}]' for v in loop_vars)} << {separator};\n"
    )
    for n in range(len(output_shape) - 1, -1, -1):
        output_code += "\t" + "    " * n + "}\n"
        if n > 0:
            output_code += "\t" + "    " * n + "std::cout << '\\n';\n"

    cpp_test_code = f"""#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <iomanip>
#include "{base_file_name}.hpp"

using Scalar = {precision_type};

int main() {{
    {input_code}
    auto output = {base_file_name}<Scalar>(input);

    std::cout << std::scientific << std::setprecision(15);  // scientific notation precision
    std::cout << "Output:\\n";  
    {output_code}
    std::cout << std::endl;

    return 0;
}}

/*
clang++ -std=c++23 -Wall -O3 -march=native -o test test.cpp
./test
*/
"""

    return cpp_test_code


def pyTestCode(precision_type, file_path, layer_shape, which_norm, custom_activation, backend):

    input_code = ""
    activation_code = ""
    norm_code = ""
    denorm_code = ""
    input_shape = tuple(layer_shape[0])
    np_dtype = "float64" if precision_type == "double" else "float32"

    custom_objects_code = ""
    if custom_activation is not None:
        activation_code += f"import {custom_activation}\n"
        activation_code += f"custom_objects = {{name: getattr({custom_activation}, name) for name in {custom_activation}.C_FUNCTIONS if hasattr({custom_activation}, name)}}\n"
        custom_objects_code = ", custom_objects=custom_objects"

    if which_norm:
        norm_code = "\n#normalization parameters\n"
    if which_norm.get("input") == "std/mean":
        norm_code += f'input_scale = np.load("input_std.npy").astype("{np_dtype}")\n'
        norm_code += f'input_shift = np.load("input_mean.npy").astype("{np_dtype}")\n'
    elif which_norm.get("input") == "max/min":
        norm_code += f'input_shift = np.load("input_min.npy").astype("{np_dtype}")\n'
        norm_code += f'input_scale = np.load("input_max.npy").astype("{np_dtype}") - input_shift\n'
    if which_norm.get("output") == "std/mean":
        norm_code += f'output_scale = np.load("output_std.npy").astype("{np_dtype}")\n'
        norm_code += f'output_shift = np.load("output_mean.npy").astype("{np_dtype}")\n'
    elif which_norm.get("output") == "max/min":
        norm_code += f'output_shift = np.load("output_min.npy").astype("{np_dtype}")\n'
        norm_code += f'output_scale = np.load("output_max.npy").astype("{np_dtype}") - output_shift\n'

    if "output" in which_norm:
        denorm_code = "output = output * output_scale.flatten() + output_shift.flatten()"

    try:
        total = 1
        for d in input_shape:
            total *= int(d)
        shape_str = ", ".join(str(int(d)) for d in input_shape)
        input_code += f"data = np.arange({total}, dtype='{np_dtype}')\n"
    except Exception:
        raise ValueError("Unsupported input shape")

    if which_norm.get("input") in ("std/mean", "max/min"):
        input_code += "data = (data - input_shift.flatten()) / input_scale.flatten()\n"
    input_code += f"data = data.reshape(1, {shape_str})\n"

    py_test_code = f"""
import os
import sys
import h5py
import numpy as np
os.environ["KERAS_BACKEND"] = "{backend}"
import keras
from keras.models import load_model
from keras import layers
{activation_code}
keras.config.set_floatx("{np_dtype}")
keras.config.set_dtype_policy("{np_dtype}")
np.set_printoptions(precision={17 if precision_type == "double" else 9}, threshold=sys.maxsize)
{norm_code}
#input data
{input_code}
#load model
file_name = "{file_path}"
with keras.device("cpu"):
    saved_model = load_model(file_name, compile=False{custom_objects_code})
    config = keras.saving.serialize_keras_object(saved_model)
    stack = [config]
    while stack:
        item = stack.pop()
        for key, value in item.items() if isinstance(item, dict) else enumerate(item):
            if key == "dtype" and "float" in str(value):
                item[key] = "{np_dtype}"
            elif isinstance(value, (dict, list)):
                stack.append(value)
    model = keras.saving.deserialize_keras_object(config{custom_objects_code})
    model.set_weights([w.astype(v.dtype) for w, v in zip(saved_model.get_weights(), model.weights)])
    model_layers = [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]
    extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model_layers])

    #extract each layer
    layer_outputs = extractor.predict(data)

print("\\nDebug printing all outputs of each layer:\\n")

#print all values of each layer
for i, layer_output in enumerate(layer_outputs):
    layer_name = model_layers[i].name
    print(f"({{layer_name}}) Layer {{i}}:")
    flat_output = layer_output.flatten()
    print(f"Values -> {{flat_output}}\\n")

#print final output
output = layer_outputs[-1].flatten()
{denorm_code}
print("Output:")
for value in output:
    print(f"{{value:.15e}}")
"""
    return py_test_code


##SAVE FOR LATER##
# #parameters
# output_folder = "layer_outputs"
# os.makedirs(output_folder, exist_ok=True)

###


# for i, layer_output in enumerate(layers):
#     layer_name = model.layers[i].name
#     file_name = f"layer_{{i}}_{{layer_name}}_output.csv"
#     file_path = os.path.join(output_folder, file_name)

#     #if last layer
#     if i == len(layers) - 1:
#         denormalized_output = layer_output * output_std + output_mean
#         flattened = denormalized_output.flatten()
#         print(denormalized_output)
#     else:
#         flattened = layer_output.flatten()

#     np.savetxt(file_path, flattened, delimiter=",")
