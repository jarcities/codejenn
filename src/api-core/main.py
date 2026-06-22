"""
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

#######################################################################

# standard libraries
import os
import re
import argparse
import math
import operator
import warnings
from pathlib import Path
from shutil import which
from functools import reduce
from collections import namedtuple
from statistics import stdev
from multiprocessing import pool

#######################################################################

## ARG PARSING ##
# https://docs.python.org/3/library/argparse.html
parser = argparse.ArgumentParser(
    description="code generate trained neural net files into a given directory."
)
parser.add_argument(
    "--input", 
    type=str, 
    required=True, 
    default="../dump_model/",
    help="path of folder with trained model files"
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    default="../bin/",
    help="path of folder to save generated header files",
)
parser.add_argument(
    "--backend",
    type=str,
    required=False,
    default="tensorflow",
    help="which backend deep learning framework to use for keras' api",
)
parser.add_argument(
    "--bit",
    type=int,
    required=False,
    default=32,
    help="use float(32)/double(64) precision for neural net computations",
)
parser.add_argument(
    "--custom_activation",
    type=str,
    required=False,
    help="custom activation function to use, if any",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="include debug print statements in the generated code",
)
parser.add_argument(
    "--model_image",
    action="store_true",
    help="save a visualization of the model architecture as a PNG file",
)

args = parser.parse_args()

#######################################################################

# scientific libraries
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sympy import use

# DL APIs
import absl.logging

## SET WHICH BACKEND TO USE ##
# https://keras.io/getting_started/
os.environ["KERAS_BACKEND"] = args.backend
import keras
from keras.models import load_model
from keras.utils import plot_model
from keras.models import Sequential

# local files
from load_model import loadModel
from extract_model import extractModel
from build_model import buildModel
from code_generate import preambleHeader, codeGen
from normalization import normParam
from testing import cppTestCode, pyTestCode

# Configure logging and warnings
absl.logging.set_verbosity("error")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#######################################################################

## DATA TYPE PRECISION ##
if args.bit == 64:
    BIT = np.float64
    BIT_ = np.int64
    precision_type = "double"
elif args.bit == 32:
    BIT = np.float32
    BIT_ = np.int32
    precision_type = "float"

input_dir = Path(args.input)
output_dir = Path(args.output)

if args.custom_activation is not None:
    custom_activation = args.custom_activation
    # print(custom_activation)
else:
    custom_activation = None

## CHECK INPUT AND OUTPUT DIRECTORIES ##
if not input_dir.exists():
    print(f"__Error__ -> Input directory '{input_dir}' does not exist.")
    exit(1)
if not output_dir.exists():
    print(
        f"__Warning__ -> Output directory '{output_dir}' does not exist. Creating it now..."
    )
    output_dir.mkdir(parents=True, exist_ok=True)

## PROCESS EACH MODEL IN INPUT DIRECTORY ##
for entry in sorted(input_dir.iterdir()):
    if entry.name == ".gitkeep" or entry.name.startswith("."):
        continue
    if entry.suffix == ".npy":
        continue
    if not entry.is_file():
        continue

    try:
        file_name = entry.name
        base_file_name = entry.stem
        file_path = str(entry)

        #########################################
        ## 1. PROCESS NORMALIZATION PARAMETERS ##
        #########################################
        input_scale, input_shift, output_scale, output_shift, which_norm = normParam(
            str(input_dir), args.debug, BIT
        )

        ###################
        ## 2. LOAD MODEL ##
        ###################
        try:
            model, file_extension = loadModel(
                file_path, base_file_name, custom_activation
            )
        except ValueError:
            print(f'\n__Skipping__ "{file_name}" -> not a compatible file.')
            continue

        #################################
        ## 3. EXTRACT MODEL EVERYTHING ##
        #################################
        try:
            (
                weights_list,
                biases_list,
                activation_functions,
                activation_configs,
                alphas,
                dropout_rates,
                norm_layer_params,
                conv_layer_params,
                input_flat_size,
                output_flat_size,
                layer_shape,
                layer_type,
            ) = extractModel(model, file_extension, base_file_name)

            # cast weights and biases to the chosen precision
            weights_list = [w.astype(BIT) if w is not None else w for w in weights_list]
            biases_list = [b.astype(BIT) if b is not None else b for b in biases_list]

            # https://keras.io/api/utils/model_plotting_utils/
            if args.model_image:
                plot_model(
                    model,
                    to_file=f"{base_file_name}_architecture.png",
                    show_shapes=True,
                    show_dtype=True,
                    show_layer_names=True,
                    show_layer_activations=True,
                )

            ## DEBUG PRINTS ##
            if args.debug:
                print(f"\nModel Summary for {file_name}:")
                print("----------------------------------")
                model.summary()
                print(f"\nWhat CodeJeNN extracted for {file_name}:")
                print("------------------------------------------------------")
                print(f"Input Size -> {input_flat_size}\n")
                print(f"Output Size -> {output_flat_size}\n")
                print(f"Layer Shape [{len(layer_shape)}] -> {layer_shape}\n")
                print(f"Layer Types [{len(layer_type)}] -> {layer_type}\n")
                print(
                    f"Activation Functions [{len(activation_functions)}] -> {activation_functions}\n"
                )
                print(
                    f"Convolutional Params (# of params per layer) [{len(conv_layer_params)}] -> ",
                    end="",
                )
                for i, params in enumerate(conv_layer_params):
                    if params is not None:
                        num_of_info = len(params)
                        print(f"{num_of_info}", end="")
                    else:
                        print("-", end="")
                    if i < len(conv_layer_params) - 1:
                        print(",", end=" ")
                print("]", end="")
                print("\n")
                print(
                    f"Normalization Params (# of params per layer) [{len(norm_layer_params)}] -> ",
                    end="",
                )
                for i, params in enumerate(norm_layer_params):
                    if params is not None:
                        num_of_info = len(params)
                        print(f"{num_of_info}", end="")
                    else:
                        print("-", end="")
                    if i < len(norm_layer_params) - 1:
                        print(",", end=" ")
                print("]", end="")
                print("\n------------------------------------------------------")

        except ValueError as e:
            print("\n__Error__ in extract_model.py -> ", e)
            continue

        ############################
        ## 4. INITIALIZE C++ CODE ##
        ############################
        # base path (without extension) for output files for this model
        output_base = output_dir / base_file_name
        cpp_code = preambleHeader()

        ######################
        ## 5. REBUILD MODEL ##
        ######################
        try:
            cpp_code, cpp_lambda = buildModel(
                cpp_code, activation_functions, layer_type, base_file_name
            )
        except ValueError as e:
            print("\n__Error__ in build_model.py -> ", e)
            continue

        ################################
        ## 6. GENERATE FINAL C++ CODE ##
        ################################
        try:
            cpp_code = codeGen(
                cpp_code,
                cpp_lambda,
                precision_type,
                weights_list,
                biases_list,
                activation_functions,
                alphas,
                dropout_rates,
                norm_layer_params,
                conv_layer_params,
                input_flat_size,
                output_flat_size,
                str(output_base),
                input_scale,
                input_shift,
                output_scale,
                output_shift,
                layer_shape,
                layer_type,
                base_file_name,
                custom_activation,
                args.debug,
            )
        except ValueError as e:
            print("\n__Error__ in code_generate.py -> ", e)
            continue

        resolved_output_dir = output_dir.resolve(strict=False)
        header_file = output_base.with_suffix(".hpp")

        with open(header_file, "w") as f:
            f.write(cpp_code)
        print(f'\nSaved "{file_name}" model in {resolved_output_dir}/')

        #######################################
        ## GENERATE TEST SCRIPT IF NECESSARY ##
        #######################################
        if args.debug:

            try:
                cpp_test_code = cppTestCode(precision_type, base_file_name, layer_shape)
            except ValueError as e:
                print("\n__Error__ in testing.py -> ", e)
                continue
            source_file = output_base.with_suffix(".cpp")
            source_file = source_file.with_name("DEBUG_" + source_file.name)
            with open(source_file, "w") as f:
                f.write(cpp_test_code)
            print(f'\nSaved "{source_file}" test code in {resolved_output_dir}/')

            try:
                py_test_code = pyTestCode(
                    precision_type,
                    file_path,
                    layer_shape,
                    which_norm,
                    custom_activation,
                    args.backend,
                )
            except ValueError as e:
                print("\n__Error__ in testing.py -> ", e)
                continue
            python_file = output_base.with_suffix(".py")
            python_file = python_file.with_name("DEBUG_" + python_file.name)
            with open(python_file, "w") as f:
                f.write(py_test_code)
            print(f'\nSaved "{python_file}" test code in {resolved_output_dir}/')

    except ValueError as e:
        print(f"\n__Skipping__ '{entry.name}' -> {e}\n")
        continue

print("\nAll done!\n")
