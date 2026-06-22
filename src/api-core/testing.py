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
    output_shape = layer_shape[-1]
    output_shape = tuple(dim for dim in output_shape if dim != 1)

    if len(input_shape) == 1:
        n = input_shape[0]
        input_code += f"\tstd::array<Scalar, {n}> input;\n\n"
        input_code += (
            f"    for (int i = 0; i < {n}; ++i) {{\n"
            "        input[i] = static_cast<Scalar>(i);\n"
            "    }\n"
        )
    elif len(input_shape) == 2:
        rows, cols = input_shape
        input_code += f"\tstd::array<std::array<Scalar, {cols}>, {rows}> input;\n\n"
        input_code += (
            "    int val = 0;\n"
            f"    for (int i = 0; i < {rows}; ++i) {{\n"
            f"        for (int j = 0; j < {cols}; ++j) {{\n"
            "            input[i][j] = static_cast<Scalar>(val);\n"
            "            ++val;\n"
            "        }\n"
            "    }\n"
        )
    elif len(input_shape) == 3:
        depth, rows, cols = input_shape
        input_code += f"\tstd::array<std::array<std::array<Scalar, {cols}>, {rows}>, {depth}> input;\n\n"
        input_code += (
            "    int val = 0;\n"
            f"    for (int d = 0; d < {depth}; ++d) {{\n"
            f"        for (int i = 0; i < {rows}; ++i) {{\n"
            f"            for (int j = 0; j < {cols}; ++j) {{\n"
            "                input[d][i][j] = static_cast<Scalar>(val);\n"
            "                ++val;\n"
            "            }\n"
            "        }\n"
            "    }\n"
        )
    else:
        raise ValueError("Unsupported input shape")

    if len(output_shape) == 1:
        m = output_shape[0]
        output_code += (
            f"\tfor (int i = 0; i < {m}; ++i) {{\n"
            f"\t    std::cout << output[i] << '\\n';\n"
            "\t}"
        )
    elif len(output_shape) == 2:
        rows, cols = output_shape
        output_code += (
            f"\tfor (int i = 0; i < {rows}; ++i) {{\n"
            f"\t    for (int j = 0; j < {cols}; ++j) {{\n"
            "\t        std::cout << output[i][j] << ' ';\n"
            "\t    }\n"
            "\t    std::cout << '\\n';\n"
            "\t}"
        )
    elif len(output_shape) == 3:
        depth, rows, cols = output_shape
        output_code += (
            f"\tfor (int d = 0; d < {depth}; ++d) {{\n"
            f"\t    for (int i = 0; i < {rows}; ++i) {{\n"
            f"\t        for (int j = 0; j < {cols}; ++j) {{\n"
            "\t            std::cout << output[d][i][j] << ' ';\n"
            "\t        }\n"
            "\t        std::cout << '\\n';\n"
            "\t    }\n"
            "\t    std::cout << '\\n';\n"
            "\t}"
        )
    else:
        raise ValueError("Unsupported output shape")

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

    if custom_activation is not None:
        activation_code += f"from {custom_activation} import {custom_activation}\n"

    if which_norm:
        norm_code = "\n#normalization parameters\n"
    if which_norm.get("input") == "std/mean":
        norm_code += 'input_scale = np.load("input_std.npy")\n'
        norm_code += 'input_shift = np.load("input_mean.npy")\n'
    elif which_norm.get("input") == "max/min":
        norm_code += 'input_scale = np.load("input_max.npy")\n'
        norm_code += 'input_shift = np.load("input_min.npy")\n'
    if which_norm.get("output") == "std/mean":
        norm_code += 'output_scale = np.load("output_std.npy")\n'
        norm_code += 'output_shift = np.load("output_mean.npy")\n'
    elif which_norm.get("output") == "max/min":
        norm_code += 'output_scale = np.load("output_max.npy")\n'
        norm_code += 'output_shift = np.load("output_min.npy")\n'

    if "output" in which_norm:
        denorm_code = """
    #denormalize if last layer
    if i == len(layer_outputs) - 1:
        layer_output = layer_output * output_scale + output_shift
"""

    try:
        total = 1
        for d in input_shape:
            total *= int(d)
        if len(input_shape) == 1:
            input_code += f"data = np.arange({total}, dtype='{np_dtype}').reshape(1, -1)\n"
        else:
            shape_str = ", ".join(str(int(d)) for d in input_shape)
            input_code += (
                f"data = np.arange({total}, dtype='{np_dtype}').reshape({shape_str})\n"
            )
    except Exception:
        raise ValueError("Unsupported input shape")

    if which_norm.get("input") == "std/mean":
        input_code += "data = (data - input_shift) / input_scale\n"
    elif which_norm.get("input") == "max/min":
        input_code += "data = (data - input_shift) / input_scale\n"

    py_test_code = f"""
import os
import h5py
import numpy as np
os.environ["KERAS_BACKEND"] = "{backend}"
import keras
from keras.models import load_model
from keras import layers
{activation_code}

#load model
file_name = "{file_path}"
model = load_model(file_name)
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
{norm_code}
#input data
{input_code}
#extract each layer
layer_outputs = extractor.predict(data)

print("\\nDebug printing first ~10 outputs of each layer:\\n")

#print first 10 values of each layer
for i, layer_output in enumerate(layer_outputs):
    layer_name = model.layers[i].name
    print(f"({{layer_name}}) Layer {{i}}:")
    {denorm_code}
    flat_output = layer_output.flatten()
    preview = flat_output[:10]  # first 10 values (or fewer if not available)
    print(f"Values -> {{preview}}\\n")
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
