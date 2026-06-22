"""
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import os
import operator
from functools import reduce


def ERROR(layer_type, layer_idx, e):
    print(f"\n__Error__ in code_generate.py: {layer_type} layer {layer_idx} --> ", e)


def debug_printing(layer_idx, layer_type, layer_shape, last_layer_name, num_values=10):
    # ===============================================
    # generate the debug printing code for each layer
    # ===============================================
    if isinstance(layer_shape, tuple):
        total_size = 1
        for dim in layer_shape:
            total_size *= dim
    else:
        total_size = layer_shape
    num_val = min(num_values, total_size)
    cpp_debug_code = (
        f"    // DEBUGGING, 1st {num_val} values of {layer_type} layer {layer_idx}:\n"
        f'    std::cout << "({layer_type}) layer {layer_idx}:" << "\\n";\n'
        '    std::cout << "Shape -> ";\n'
    )
    if isinstance(layer_shape, tuple):
        shape_str = "(" + ", ".join(map(str, layer_shape)) + ")"
    else:
        shape_str = f"({layer_shape},)"
    cpp_debug_code += (
        f'    std::cout << "{shape_str}" << "\\n";\n'
        '    std::cout << "Values -> ";\n'
        f"    for (int ii = 0; ii < {num_val}; ++ii)\n"
        "    {\n"
        f"        std::cout << {last_layer_name}[ii];\n"
        f'        if (ii < {num_val} - 1) std::cout << ", ";\n'
        "    }\n"
        '    std::cout << " . . .\\n\\n";\n\n'
    )
    return cpp_debug_code


def preambleHeader():
    # ===========================================
    # generate the preamble for the header file
    # ===========================================
    cpp_code = (
        "#pragma once\n"
        "#include <iostream>\n"
        "#include <array>\n"
        "#include <random>\n"
        "#include <cmath>\n"
        "#include <functional>\n"
        "#include <stdexcept>\n"
        "#include <algorithm>\n"
        "#include <cstddef>\n"
        "#include <vector>\n"
        "#include <limits>\n"
        "\n/*\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//*/ \n\n"
    )
    return cpp_code


def codeGen(
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
    input_size,
    output_size,
    user_file,
    input_scale,
    input_shift,
    output_scale,
    output_shift,
    layer_shape,
    layer_type,
    base_file_name,
    user_activation,
    debug_outputs,
):
    # ===============================================================================
    # function to generate put all the cpp code together from the previous scripts
    # and print out all the layer parameters, activation function, and function
    # calls for the model.

    # args:
    #   cpp_code: the code to be generated
    #   cpp_lambda: the activation function lambda definitions
    #   precision_type: the precision type to be used in the model
    #   weights_list: the list of weights for each layer
    #   biases_list: the list of biases for each layer
    #   activation_functions: the list of activation functions for each layer
    #   alphas: the list of alphas for each layer
    #   dropout_rates: the list of dropout rates for each layer
    #   norm_layer_params: the list of normalization parameters for each layer
    #   conv_layer_params: the list of convolutional layer parameters for each layer
    #   input_size: the size of the input layer
    #   user_file: the name of the user file
    #   input_scale: the input normalization parameters
    #   input_shift: the input minimum values
    #   output_scale: the output normalization parameters
    #   output_shift: the output minimum values
    #   layer_shape: the shape of the layers
    #   layer_type: the type of the layers

    # returns:
    #   cpp_code: the fully generated cpp code
    # ===============================================================================

    # precision-dependent format specifier
    # float32 needs ~9 significant digits, float64 needs ~17
    val_fmt = "20.17e" if precision_type == "double" else "10.9e"
    def fmt(v):
        return f"{v:{val_fmt}}"
    def fmt_arr(arr):
        return ", ".join(fmt(v) for v in arr)

    # compute the flat size of the input layer
    def get_flat_size(shape):
        if isinstance(shape, tuple):
            prod = 1
            for d in shape:
                prod *= d
            return prod
        return shape

    # list of supported activation functions
    activation_func_map = {
        "relu": "relu",
        "sigmoid": "sigmoid",
        "tanh": "tanhCustom",
        "leakyrelu": "leakyrelu",
        "linear": "linear",
        "elu": "elu",
        "selu": "selu",
        "swish": "swish",
        "prelu": "prelu",
        "silu": "silu",
        "gelu": "gelu",
        "softmax": "softmax",
        "mish": "mish",
        "softplus": "softplus",
        "flatten": None,
    }
    if user_activation:
        activation_func_map[user_activation] = user_activation

    name_space = os.path.splitext(os.path.basename(user_file))[0]
    name_space = name_space.replace("-", "_").replace(" ", "_")

    raw_shape = layer_shape[0]
    if isinstance(raw_shape, tuple) and len(raw_shape) > 1:
        dims = [d for d in raw_shape if d != 1]
        input_type = "Scalar"
        for d in reversed(dims):
            input_type = f"std::array<{input_type}, {d}>"
    else:
        d = raw_shape[0] if isinstance(raw_shape, tuple) else raw_shape
        input_type = f"std::array<Scalar, {d}>"

    cpp_code += "\n\n/*\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//*/ \n\n"

    cpp_code += f"""
template <typename Scalar = {precision_type}>
inline auto {name_space}(const {input_type}& initial_input) {{\n
"""

    ##################################
    ## PRINT EACH LAYERS PARAMETERS ##
    ##################################
    for i, (w, b, norm_params, conv_dict, ltype) in enumerate(
        zip(weights_list, biases_list, norm_layer_params, conv_layer_params, layer_type)
    ):
        layer_idx = i + 1

        ## PREPROCESSING LAYERS ##
        if ltype == "Rescale":
            try:
                scale, offset = norm_params
                cpp_code += (
                    f"    // Rescale Input/Output {layer_idx}\n"
                    f"    constexpr std::array<Scalar, {len(scale)}> scale_{layer_idx} = {{"
                    + fmt_arr(scale)
                    + "};\n"
                    f"    constexpr std::array<Scalar, {len(offset)}> offset_{layer_idx} = {{"
                    + fmt_arr(offset)
                    + "};\n\n"
                )
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        ## DENSE LAYERS ##
        if ltype == "Dense":
            try:
                wflat = w.flatten()
                bflat = b.flatten()
                cpp_code += (
                    f"    // Dense layer {layer_idx}\n"
                    f"    constexpr std::array<Scalar, {len(wflat)}> weights_{layer_idx} = {{"
                    + fmt_arr(wflat)
                    + "};\n"
                    f"    constexpr std::array<Scalar, {len(bflat)}> biases_{layer_idx} = {{"
                    + fmt_arr(bflat)
                    + "};\n\n"
                )
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        ## NORMALIZATION LAYERS ##
        if norm_params is not None and ltype != "Rescale":
            try:
                if len(norm_params) == 6:
                    gamma, beta, mean, var, eps, groups = norm_params
                    cpp_code += (
                        f"    // Layer {layer_idx}: GroupNormalization\n"
                        f"    constexpr int groups_{layer_idx} = {groups};\n"
                    )
                else:
                    gamma, beta, mean, var, eps = norm_params
                    cpp_code += f"    // Layer {layer_idx}: Normalization\n"

                if gamma is not None:
                    gflat = gamma.flatten()
                    cpp_code += (
                        f"    constexpr std::array<Scalar, {len(gflat)}> gamma_{layer_idx} = {{"
                        + fmt_arr(gflat)
                        + "};\n"
                    )
                if beta is not None:
                    bflat = beta.flatten()
                    cpp_code += (
                        f"    constexpr std::array<Scalar, {len(bflat)}> beta_{layer_idx} = {{"
                        + fmt_arr(bflat)
                        + "};\n"
                    )
                if mean is not None:
                    mflat = mean.flatten()
                    cpp_code += (
                        f"    constexpr std::array<Scalar, {len(mflat)}> mean_{layer_idx} = {{"
                        + fmt_arr(mflat)
                        + "};\n"
                    )
                if var is not None:
                    vflat = var.flatten()
                    cpp_code += (
                        f"    constexpr std::array<Scalar, {len(vflat)}> variance_{layer_idx} = {{"
                        + fmt_arr(vflat)
                        + "};\n"
                    )
                cpp_code += (
                    f"    constexpr Scalar epsilon_{layer_idx} = {fmt(eps)};\n\n"
                )
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        ## CONVOLUTIONAL LAYERS ##
        if conv_dict is not None:
            ltype = conv_dict.get("layer_type", None)
            cpp_code += f"    // Layer {layer_idx}: {ltype}\n"

            try:
                if ltype in ["Conv1D", "Conv2D", "Conv3D"] and ltype not in [
                    "Conv1DTranspose",
                    "Conv2DTranspose",
                    "Conv3DTranspose",
                ]:
                    kernel = conv_dict.get("weights", None)
                    bias = conv_dict.get("biases", None)
                    if kernel is not None:
                        kflat = kernel.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(kflat)}> convKernel_{layer_idx} = {{"
                            + fmt_arr(kflat)
                            + "};\n"
                        )
                    out_shape = conv_dict.get(
                        "out_shape", conv_dict.get("output_shape", None)
                    )
                    if bias is not None:
                        bflat = bias.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(bflat)}> convBias_{layer_idx} = {{"
                            + fmt_arr(bflat)
                            + "};\n"
                        )
                    else:
                        size = (
                            out_shape[-1]
                            if out_shape
                            else (conv_dict.get("filters") or 1)
                        )
                        cpp_code += f"    constexpr std::array<Scalar, {size}> convBias_{layer_idx} = {{}};\n"
                    cpp_code += "\n"
                    continue

                elif ltype in ["DepthwiseConv1D", "DepthwiseConv2D"]:
                    kernel = conv_dict.get(
                        "depthwise_kernel", conv_dict.get("weights", None)
                    )
                    bias = conv_dict.get(
                        "depthwise_bias", conv_dict.get("biases", None)
                    )
                    depth = int(conv_dict.get("depth_multiplier", 1))
                    cpp_code += (
                        f"    constexpr int depthMultiplier_{layer_idx} = {depth};\n"
                    )

                    if kernel is not None:
                        kflat = kernel.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(kflat)}> depthwiseKernel_{layer_idx} = {{"
                            + fmt_arr(kflat)
                            + "};\n"
                        )

                    out_shape = conv_dict.get(
                        "out_shape", conv_dict.get("output_shape", None)
                    )

                    if bias is not None:
                        bflat = bias.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(bflat)}> depthwiseBias_{layer_idx} = {{"
                            + fmt_arr(bflat)
                            + "};\n"
                        )
                    else:
                        size = (
                            out_shape[-1]
                            if out_shape
                            else (conv_dict.get("filters") or 1)
                        )
                        cpp_code += f"    constexpr std::array<Scalar, {size}> depthwiseBias_{layer_idx} = {{}};\n"
                    cpp_code += "\n"
                    continue

                elif ltype in ["SeparableConv1D", "SeparableConv2D"]:
                    dk = conv_dict.get("depthwise_kernel", None)
                    pk = conv_dict.get("pointwise_kernel", None)
                    db = conv_dict.get("depthwise_bias", None)
                    pb = conv_dict.get("pointwise_bias", None)
                    if dk is not None:
                        dkflat = dk.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(dkflat)}> sepDepthwise_{layer_idx} = {{"
                            + fmt_arr(dkflat)
                            + "};\n"
                        )
                    if db is not None:
                        dbflat = db.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(dbflat)}> sepDepthwiseBias_{layer_idx} = {{"
                            + fmt_arr(dbflat)
                            + "};\n"
                        )
                    if pk is not None:
                        pkflat = pk.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(pkflat)}> sepPointwise_{layer_idx} = {{"
                            + fmt_arr(pkflat)
                            + "};\n"
                        )
                    if pb is not None:
                        pbflat = pb.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(pbflat)}> sepPointwiseBias_{layer_idx} = {{"
                            + fmt_arr(pbflat)
                            + "};\n"
                        )
                    cpp_code += "\n"
                    continue

                elif ltype in [
                    "Conv1DTranspose",
                    "Conv2DTranspose",
                    "Conv3DTranspose",
                ] and ltype not in ["Conv1D", "Conv2D", "Conv3D"]:
                    kernel = conv_dict.get("weights", None)
                    bias = conv_dict.get("biases", None)
                    if kernel is not None:
                        kflat = kernel.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(kflat)}> convKernel_{layer_idx} = {{"
                            + fmt_arr(kflat)
                            + "};\n"
                        )
                    if bias is not None:
                        bflat = bias.flatten()
                        cpp_code += (
                            f"    constexpr std::array<Scalar, {len(bflat)}> convBias_{layer_idx} = {{"
                            + fmt_arr(bflat)
                            + "};\n"
                        )
                    cpp_code += "\n"
                    continue

                ##########################################################################
                # -------------------------------------------------------------------------
                # elif ltype == "ConvLSTM2D":
                #     # Layer {layer_idx}: ConvLSTM2D parameters

                #     print("error before printing arrays")
                #     kernel           = conv_dict.get("kernel", None)
                #     recurrent_kernel = conv_dict.get("recurrent_kernel", None)
                #     bias             = conv_dict.get("bias", None)

                #     if kernel is not None:
                #         kflat = kernel.flatten()
                #         cpp_code += (
                #             f"    constexpr std::array<Scalar, {len(kflat)}> convKernel_{layer_idx} = {{"
                #             + ", ".join(f"{val:10.9e}" for val in kflat)
                #             + "};\n"
                #         )
                #     if recurrent_kernel is not None:
                #         rkflat = recurrent_kernel.flatten()
                #         cpp_code += (
                #             f"    constexpr std::array<Scalar, {len(rkflat)}> recurrentKernel_{layer_idx} = {{"
                #             + ", ".join(f"{val:10.9e}" for val in rkflat)
                #             + "};\n"
                #         )
                #     if bias is not None:
                #         bflat = bias.flatten()
                #         cpp_code += (
                #             f"    constexpr std::array<Scalar, {len(bflat)}> convBias_{layer_idx} = {{"
                #             + ", ".join(f"{val:10.9e}" for val in bflat)
                #             + "};\n\n"
                #         )

                #     print("error after printing arrays")
                # -------------------------------------------------------------------------
                ##########################################################################

                ## POOLING LAYERS ##
                elif ltype in ["MaxPooling1D", "AvgPooling1D"]:
                    pool_size = conv_dict.get("pool_size", 2)
                    strides = conv_dict.get("strides", pool_size)
                    padding = conv_dict.get("padding", "valid")
                    cpp_code += (
                        f"    constexpr std::array<int, 1> poolSize_{layer_idx} = {{{pool_size}}};\n"
                        f"    constexpr std::array<int, 1> poolStrides_{layer_idx} = {{{strides}}};\n"
                        f'    constexpr const char* poolPadding_{layer_idx} = "{padding}";\n\n'
                    )
                    continue

                elif ltype in ["MaxPooling2D", "AvgPooling2D"]:
                    pool_size = conv_dict.get("pool_size", (2, 2))
                    strides = conv_dict.get("strides", pool_size)
                    padding = conv_dict.get("padding", "valid")
                    cpp_code += (
                        f"    constexpr std::array<int, 2> poolSize_{layer_idx} = {{{pool_size[0]}, {pool_size[1]}}};\n"
                        f"    constexpr std::array<int, 2> poolStrides_{layer_idx} = {{{strides[0]}, {strides[1]}}};\n"
                        f'    constexpr const char* poolPadding_{layer_idx} = "{padding}";\n\n'
                    )
                    continue

                elif ltype in ["MaxPooling3D", "AvgPooling3D"]:
                    pool_size = conv_dict.get("pool_size", (2, 2, 2))
                    strides = conv_dict.get("strides", pool_size)
                    padding = conv_dict.get("padding", "valid")
                    cpp_code += (
                        f"    constexpr std::array<int, 3> poolSize_{layer_idx} = {{{pool_size[0]}, {pool_size[1]}, {pool_size[2]}}};\n"
                        f"    constexpr std::array<int, 3> poolStrides_{layer_idx} = {{{strides[0]}, {strides[1]}, {strides[2]}}};\n"
                        f'    constexpr const char* poolPadding_{layer_idx} = "{padding}";\n\n'
                    )
                    continue

                elif ltype in ["GlobalMaxPooling1D", "GlobalAvgPooling1D"]:
                    in_shape = conv_dict["in_shape"]
                    cpp_code += f"    constexpr std::array<int, 1> poolSize_{layer_idx} = {{{in_shape[0]}}};\n\n"
                    continue

                elif ltype in ["GlobalMaxPooling2D", "GlobalAvgPooling2D"]:
                    in_shape = conv_dict["in_shape"]
                    cpp_code += f"    constexpr std::array<int, 2> poolSize_{layer_idx} = {{{in_shape[0]}, {in_shape[1]}}};\n\n"
                    continue

                elif ltype in ["GlobalMaxPooling3D", "GlobalAvgPooling3D"]:
                    in_shape = conv_dict["in_shape"]
                    cpp_code += f"    constexpr std::array<int, 3> poolSize_{layer_idx} = {{{in_shape[0]}, {in_shape[1]}, {in_shape[2]}}};\n\n"
                    continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

    cpp_code += "\n/*\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//*/ \n\n\n"

    ## NORMALIZE INPUT AND OUTPUTS ##
    try:
        if input_scale is not None and input_shift is not None:
            input_scale = (
                input_scale.flatten()
                if hasattr(input_scale, "flatten")
                else input_scale
            )
            input_shift = (
                input_shift.flatten()
                if hasattr(input_shift, "flatten")
                else input_shift
            )
            cpp_code += (
                f"    constexpr std::array<Scalar, {len(input_scale)}> input_scale = {{"
                + fmt_arr(float(x) for x in input_scale)
                + "};\n\n"
                f"    constexpr std::array<Scalar, {len(input_shift)}> input_shift = {{"
                + fmt_arr(float(x) for x in input_shift)
                + "};\n\n"
            )
    except ValueError as e:
        ERROR("input normalization", "0", e)
        pass

    out_norm_size = output_size
    out_size = layer_shape[len(layer_shape) - 1]
    try:
        if output_scale is not None and output_shift is not None:
            output_scale = (
                output_scale.flatten()
                if hasattr(output_scale, "flatten")
                else output_scale
            )
            output_shift = (
                output_shift.flatten()
                if hasattr(output_shift, "flatten")
                else output_shift
            )
            cpp_code += (
                f"    constexpr std::array<Scalar, {len(output_scale)}> output_scale = {{"
                + fmt_arr(float(x) for x in output_scale)
                + "};\n\n"
                f"    constexpr std::array<Scalar, {len(output_shift)}> output_shift = {{"
                + fmt_arr(float(x) for x in output_shift)
                + "};\n\n"
                "\n/*\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//*/ \n\n"
            )
    except ValueError as e:
        ERROR("output normalization", "0", e)
        pass

    #################################
    ## INSERT ACTIVATION FUNCTIONS ##
    #################################
    if isinstance(cpp_lambda, dict):
        relevant_activations = set(activation_functions)
        for key, val in cpp_lambda.items():
            if key in relevant_activations:
                cpp_code += val
    else:
        cpp_code += cpp_lambda

    cpp_code += "\n\n/*\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//*/ \n\n"

    ###################
    ## FLATTEN INPUT ##
    ###################
    cpp_code += f"""    
    // model input and flattened
    constexpr int flat_size = {input_size}; 
    std::array<Scalar, flat_size> model_input;\n
    """

    dims = layer_shape[0]
    indent = ""

    ## FLATTEN INPUT ##
    if isinstance(dims, tuple) and len(dims) > 1:
        try:
            dims = [d for d in raw_shape if d != 1]
            loop_vars = [f"i{j}" for j in range(len(dims))]
            for d_i, d_val in enumerate(dims):
                cpp_code += f"{indent}for (int {loop_vars[d_i]} = 0; {loop_vars[d_i]} < {d_val}; {loop_vars[d_i]}++) {{\n"
                indent = "      " * (d_i + 1)

            index_expr = ""
            for d_i in range(len(dims)):
                stride = 1
                for d_j in range(d_i + 1, len(dims)):
                    stride *= dims[d_j]
                if d_i > 0:
                    index_expr += " + "
                index_expr += f"{loop_vars[d_i]} * {stride}"
            cpp_code += "    " * (len(dims) + 1) + f"int flatIndex = {index_expr};\n"

            if input_scale is None:
                cpp_code += (
                    "    " * (len(dims) + 1) + "model_input[flatIndex] = initial_input"
                )
                for lv in loop_vars:
                    cpp_code += f"[{lv}]"
                cpp_code += ";\n"
            else:
                cpp_code += (
                    "    " * (len(dims) + 1) + "model_input[flatIndex] = (initial_input"
                )
                for lv in loop_vars:
                    cpp_code += f"[{lv}]"
                cpp_code += " - input_shift[flatIndex]) / (input_scale[flatIndex]);\n"

            for d_i in range(len(dims), 0, -1):
                cpp_code += "    " * d_i + "}\n"

            cpp_code += f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'
        except ValueError as e:
            ERROR("input layer", "0", e)
            pass

    else:
        if input_scale is None:
            cpp_code += (
                "// pass input\n"
                "    for (int i=0; i<flat_size; i++) { model_input[i] = initial_input[i]; }\n\n"
                f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'
            )
        else:
            cpp_code += (
                "// normalize input\n"
                f"    for (int i = 0; i < {input_size}; i++) {{ model_input[i] = (initial_input[i] - input_shift[i]) / (input_scale[i]); }} \n\n"
                f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'
            )

    ######################################
    ## PRINT EACH LAYERS FUNCTION CALLS ##
    ######################################
    last_layer = "model_input"
    last_shape = layer_shape[0]
    layer_idx = 0

    if debug_outputs:
        cpp_code += (
            "    // DEBUG PRINTING FLAG IS ON\n"
            '    std::cout << "\\n";\n'
            '    std::cout << "Debug printing first ~10 outputs of each layer:" << "\\n";\n'
            '    std::cout << "\\n";\n\n'
        )

    for i, (w, b, norm_params, conv_dict, ltype, alpha, act_fun) in enumerate(
        zip(
            weights_list,
            biases_list,
            norm_layer_params,
            conv_layer_params,
            layer_type,
            alphas,
            activation_functions,
        )
    ):

        layer_idx = i + 1

        if len(layer_shape) > i + 1:
            current_shape = layer_shape[i + 1]
        else:
            current_shape = None

        ## MAYBE SOFTMAX ISSUE?? ##
        if act_fun == "softmax" and ltype != "Activation":
            mapped_act = "linear"
            alpha = 0.0
        else:
            mapped_act = activation_func_map.get(act_fun, "linear")
            # alpha = alpha #leave alpha as is
        # mapped_act = activation_func_map.get(act_fun, "linear")

        ## THIS IS SOFTMAX FROM PREVIOUS LAYER ##
        if (
            i > 0
            and activation_functions[i - 1] == "softmax"
            and (layer_type[i - 1] if isinstance(layer_type, (list, tuple)) else None)
            != "Activation"
        ):
            if isinstance(last_shape, tuple) and len(last_shape) > 1:
                channels = last_shape[-1]
                length = 1
                for d in last_shape[:-1]:
                    length *= d
                cpp_code += (
                    f"    // standalone softmax (from previous layer), layer {layer_idx-1}: "
                    f"along last axis (channels={channels}, groups={length})\n"
                    f"    for (size_t g = 0; g < {length}; ++g) {{\n"
                    f"        softmax({last_layer}.data() + g*{channels}, "
                    f"        {last_layer}.data() + g*{channels}, {channels});\n"
                    "    }\n\n"
                )
            else:
                size = get_flat_size(last_shape)
                cpp_code += (
                    f"    // standalone softmax layer for layer {layer_idx-1}\n"
                    f"    softmax({last_layer}.data(), {last_layer}.data(), {size});\n\n"
                )
            # debug printing flag
            if debug_outputs:
                cpp_code += debug_printing(
                    layer_idx - 1, "SOFTMAX FROM LAST LAYER", last_shape, last_layer
                )
            # continue

        ## ACTIVATION LAYERS ##
        if ltype == "Activation":
            try:
                if act_fun == "softmax":
                    if isinstance(last_shape, tuple) and len(last_shape) > 1:
                        channels = last_shape[-1]
                        length = 1
                        for d in last_shape[:-1]:
                            length *= d
                        total_size = channels * length
                        cpp_code += (
                            "    // Standalone softmax layer\n"
                            f"    static std::array<Scalar, {total_size}> layer_{layer_idx}_output;\n"
                            f"    for (size_t g = 0; g < {length}; ++g) {{\n"
                            f"        softmax(layer_{layer_idx}_output.data() + g*{channels}, {last_layer}.data() + g*{channels}, {channels});\n"
                            "    }\n\n"
                        )
                    else:
                        size = get_flat_size(last_shape)
                        cpp_code += (
                            "    // Standalone softmax layer\n"
                            f"    static std::array<Scalar, {size}> layer_{layer_idx}_output;\n"
                            f"    softmax(layer_{layer_idx}_output.data(), {last_layer}.data(), {size});\n\n"
                        )
                    last_layer = f"layer_{layer_idx}_output"
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue

                else:
                    cpp_code += f"    // Pure {ltype}, layer {layer_idx}\n"
                    if act_fun == user_activation and user_activation is not None:
                        cpp_code += '\tprintf("WARNING: CUSTOM ACTIVATION HAS NOT BEEN IMPLEMENTED");\n'
                    cpp_code += (
                        f"    static std::array<Scalar, {get_flat_size(last_shape)}> layer_{layer_idx}_output;\n"
                        f"    for (int i = 0; i < {get_flat_size(last_shape)}; ++i) {{\n"
                        f"        {act_fun}(layer_{layer_idx}_output[i], {last_layer}[i], {alpha});\n"
                        "    }\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        ## PREPROCESSING LAYERS ##
        if ltype == "Rescale":
            try:
                cpp_code += (
                    f"    // {ltype}, layer {layer_idx}\n"
                    f"    static std::array<Scalar,  {get_flat_size(current_shape)}> layer_{layer_idx}_output;\n"
                    f"    Rescale_{base_file_name}<Scalar, {get_flat_size(current_shape)}>(\n"
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    f"        scale_{layer_idx}.data(), offset_{layer_idx}.data());\n\n"
                )
                last_layer = f"layer_{layer_idx}_output"
                last_shape = current_shape
                # debug printing flag
                if debug_outputs:
                    cpp_code += debug_printing(layer_idx, ltype, last_shape, last_layer)
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        ## RESHAPE LAYERS ##
        if ltype == "Reshape":
            try:
                cpp_code += (
                    f"    // {ltype}, layer {layer_idx}\n"
                    f"    static std::array<Scalar, {get_flat_size(current_shape)}> layer_{layer_idx}_output;\n"
                    f"    Reshape_{base_file_name}<Scalar, {get_flat_size(current_shape)}>(\n"
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data());\n\n"
                )
                last_layer = f"layer_{layer_idx}_output"
                last_shape = current_shape
                # debug printing flag
                if debug_outputs:
                    cpp_code += debug_printing(layer_idx, ltype, last_shape, last_layer)
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        ## DENSE LAYER ##
        if ltype == "Dense":

            try:
                out_size = w.shape[1]
                cpp_code += (
                    f"    // {ltype}, layer {layer_idx}\n"
                    f"    static std::array<Scalar, {out_size}> layer_{layer_idx}_output;\n"
                    f"    Dense_{base_file_name}<Scalar, {out_size}>(\n"
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    f"        weights_{layer_idx}.data(), biases_{layer_idx}.data(),\n"
                    f"        {get_flat_size(last_shape)}, {mapped_act}, {alpha});\n\n"
                )
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (out_size,)
                # debug printing flag
                if debug_outputs:
                    cpp_code += debug_printing(layer_idx, ltype, last_shape, last_layer)
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        ## NORMALIZATION LAYERS ##
        if (
            ltype is not None
            and ltype.lower() in ["batchnormalization", "batchnormalization2d"]
        ) and norm_params is not None:
            gamma, beta, mean, var, eps = norm_params

            try:
                if isinstance(last_shape, tuple) and len(last_shape) > 1:
                    if len(last_shape) == 3:
                        if last_shape[2] == 1:
                            length, channels, _ = last_shape
                        else:
                            height, width, channels = last_shape
                            length = height * width
                    elif len(last_shape) == 4:
                        depth, height, width, channels = last_shape
                        length = depth * height * width
                    else:
                        channels = last_shape[-1]
                        length = reduce(operator.mul, last_shape[:-1], 1)
                else:
                    channels = len(gamma)
                    length = 1

                total_size = channels * length
                cpp_code += (
                    f"    // {ltype}, layer {layer_idx}\n"
                    f"    static std::array<Scalar, {total_size}> layer_{layer_idx}_output;\n"
                    f"    BatchNormalization_{base_file_name}<Scalar, {channels}, {length}>(\n"
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                    f"        mean_{layer_idx}.data(), variance_{layer_idx}.data(),\n"
                    f"        epsilon_{layer_idx});\n\n"
                )
                last_layer = f"layer_{layer_idx}_output"

                if length == 1:
                    last_shape = (channels,)
                else:
                    last_shape = last_shape
                # debug printing flag
                if debug_outputs:
                    cpp_code += debug_printing(layer_idx, ltype, last_shape, last_layer)
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        if (
            ltype is not None
            and ltype.lower() in ["layernormalization", "layernormalization2d"]
        ) and norm_params is not None:
            gamma, beta, mean, var, eps = norm_params

            try:
                if isinstance(last_shape, tuple) and len(last_shape) > 1:
                    if len(last_shape) == 3:
                        if last_shape[2] == 1:
                            length, channels, _ = last_shape
                        else:
                            height, width, channels = last_shape
                            length = height * width
                    elif len(last_shape) == 4:
                        depth, height, width, channels = last_shape
                        length = depth * height * width
                    else:
                        channels = last_shape[-1]
                        length = reduce(operator.mul, last_shape[:-1], 1)
                else:
                    channels = len(gamma)
                    length = 1

                total_size = channels * length
                cpp_code += (
                    f"    // {ltype}, layer {layer_idx}\n"
                    f"    static std::array<Scalar, {total_size}> layer_{layer_idx}_output;\n"
                    f"    LayerNormalization_{base_file_name}<Scalar, {channels}, {length}>(\n"
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                    f"        epsilon_{layer_idx});\n\n"
                )
                last_layer = f"layer_{layer_idx}_output"
                if length == 1:
                    last_shape = (channels,)
                else:
                    last_shape = last_shape
                # debug printing flag
                if debug_outputs:
                    cpp_code += debug_printing(layer_idx, ltype, last_shape, last_layer)
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        if ltype == "UnitNormalization":
            try:
                if isinstance(last_shape, tuple) and len(last_shape) > 1:
                    if len(last_shape) == 3:
                        if last_shape[2] == 1:
                            length, channels, _ = last_shape
                        else:
                            height, width, channels = last_shape
                            length = height * width
                    elif len(last_shape) == 4:
                        depth, height, width, channels = last_shape
                        length = depth * height * width
                    else:
                        channels = last_shape[-1]
                        length = reduce(operator.mul, last_shape[:-1], 1)
                else:
                    channels = get_flat_size(last_shape)
                    length = 1

                total_size = channels * length
                cpp_code += (
                    f"    // {ltype}, layer {layer_idx}\n"
                    f"    static std::array<Scalar, {total_size}> layer_{layer_idx}_output;\n"
                    f"    UnitNormalization_{base_file_name}<Scalar, {channels}, {length}>(\n"
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    f"        epsilon_{layer_idx});\n\n"
                )
                last_layer = f"layer_{layer_idx}_output"
                if length == 1:
                    last_shape = (channels,)
                else:
                    last_shape = last_shape
                # debug printing flag
                if debug_outputs:
                    cpp_code += debug_printing(layer_idx, ltype, last_shape, last_layer)
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        if (
            ltype is not None
            and ltype.lower() in ["groupnormalization", "groupnormalization2d"]
        ) and norm_params is not None:
            try:
                if len(norm_params) == 6:
                    gamma, beta, mean, var, eps, groups = norm_params
                else:
                    gamma, beta, mean, var, eps = norm_params
                    groups = 32  # default to 32 groups

                if isinstance(last_shape, tuple) and len(last_shape) > 1:
                    if len(last_shape) == 3:
                        if last_shape[2] == 1:
                            length, channels, _ = last_shape
                        else:
                            height, width, channels = last_shape
                            length = height * width
                    elif len(last_shape) == 4:
                        depth, height, width, channels = last_shape
                        length = depth * height * width
                    else:
                        channels = last_shape[-1]
                        length = reduce(operator.mul, last_shape[:-1], 1)
                else:
                    channels = len(gamma)
                    length = 1

                total_size = channels * length
                cpp_code += (
                    f"    // {ltype}, layer {layer_idx}\n"
                    f"    static std::array<Scalar, {total_size}> layer_{layer_idx}_output;\n"
                    f"    GroupNormalization_{base_file_name}<Scalar, {channels}, {length}, {groups}>(\n"
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                    f"        epsilon_{layer_idx});\n\n"
                )
                last_layer = f"layer_{layer_idx}_output"
                if length == 1:
                    last_shape = (channels,)
                else:
                    last_shape = last_shape
                # debug printing flag
                if debug_outputs:
                    cpp_code += debug_printing(layer_idx, ltype, last_shape, last_layer)
                continue
            except ValueError as e:
                ERROR(ltype, layer_idx, e)
                continue

        ## CONVOLUTIONAL LAYERS ##
        if ltype is not None and conv_dict is not None:

            in_shape = conv_dict.get("in_shape", None)
            out_shape = conv_dict.get("out_shape", conv_dict.get("output_shape", None))

            if in_shape is None:
                in_shape = (
                    last_shape if isinstance(last_shape, tuple) else (last_shape,)
                )

            if ltype == "Conv1D" and ltype not in ["Conv1DTranspose"]:
                try:
                    input_length = in_shape[0] if len(in_shape) > 0 else 1
                    input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                    output_length = out_shape[0] if out_shape else 1
                    output_channels = (
                        out_shape[-1]
                        if out_shape and len(out_shape) > 1
                        else (conv_dict.get("filters") or 1)
                    )

                    kernel_size_val = conv_dict.get("kernel_size", 3)
                    strides_val = conv_dict.get("strides", 1)
                    pad = (
                        kernel_size_val // 2
                        if str(conv_dict.get("padding", "valid")).lower() == "same"
                        else 0
                    )
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {output_length * output_channels}> layer_{layer_idx}_output;\n"
                        f"    Conv1D_{base_file_name}<Scalar, {output_channels}, {output_length}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                        f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                        f"        {input_channels}, {input_length}, {kernel_size_val}, {strides_val}, {pad},\n"
                        f"        {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (output_length, output_channels)
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "Conv2D" and ltype not in ["Conv2DTranspose"]:
                try:
                    in_shape = conv_dict.get(
                        "in_shape",
                        ("/* in_height */", "/* in_width */", "/* in_channels */"),
                    )
                    out_shape = conv_dict.get(
                        "out_shape",
                        conv_dict.get(
                            "output_shape",
                            (
                                "/* out_height */",
                                "/* out_width */",
                                "/* out_channels */",
                            ),
                        ),
                    )
                    if len(in_shape) == 2:
                        in_shape = (in_shape[0], in_shape[1], 1)

                    kernel = conv_dict.get("kernel_size", (3, 3))
                    strides = conv_dict.get("strides", (1, 1))
                    padding = conv_dict.get("padding", "valid")
                    pad_h = pad_w = 0
                    if padding.lower() == "same":
                        pad_h = kernel[0] // 2
                        pad_w = kernel[1] // 2
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                        f"    Conv2D_{base_file_name}<Scalar, {out_shape[2]}, {out_shape[0]}, {out_shape[1]}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                        f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                        f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                        f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                        f"        {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "Conv3D" and ltype not in ["Conv3DTranspose"]:
                try:
                    kernel = conv_dict.get("kernel_size", (3, 3, 3))
                    strides = conv_dict.get("strides", (1, 1, 1))
                    padding = conv_dict.get("padding", "valid")
                    pd = ph = pw = 0
                    if padding.lower() == "same":
                        pd = kernel[0] // 2
                        ph = kernel[1] // 2
                        pw = kernel[2] // 2
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]} * {out_shape[3]})> layer_{layer_idx}_output;\n"
                        f"    Conv3D_{base_file_name}<Scalar, {out_shape[3]}, {out_shape[0]}, {out_shape[1]}, {out_shape[2]}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                        f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                        f"        {in_shape[3]}, {in_shape[0]}, {in_shape[1]}, {in_shape[2]},\n"
                        f"        {kernel[0]}, {kernel[1]}, {kernel[2]}, {strides[0]}, {strides[1]}, {strides[2]}, {pd}, {ph}, {pw},\n"
                        f"        {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "DepthwiseConv1D":
                try:
                    input_length = in_shape[0] if len(in_shape) > 0 else 1
                    input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                    output_length = out_shape[0] if out_shape else 1
                    output_channels = (
                        out_shape[-1]
                        if out_shape and len(out_shape) > 1
                        else (conv_dict.get("filters") or input_channels)
                    )

                    kernel_size_val = conv_dict.get("kernel_size", 3)
                    strides_val = conv_dict.get("strides", 1)
                    pad = (
                        kernel_size_val // 2
                        if str(conv_dict.get("padding", "valid")).lower() == "same"
                        else 0
                    )
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {output_length * output_channels}> layer_{layer_idx}_output;\n"
                        f"    DepthwiseConv1D_{base_file_name}(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                        f"        depthwiseKernel_{layer_idx}.data(), depthwiseBias_{layer_idx}.data(),\n"
                        f"        {output_channels}, {output_length}, {input_channels}, {input_length},\n"
                        f"        {kernel_size_val}, {strides_val}, {pad},\n"
                        f"        {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (output_length, output_channels)
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            # elif ltype == "DepthwiseConv2D":
            #     try:
            #         kernel = conv_dict.get("kernel_size", (3, 3))
            #         strides = conv_dict.get("strides", (1, 1))
            #         padding = conv_dict.get("padding", "valid")
            #         pad_h = pad_w = 0
            #         if padding.lower() == "same":
            #             pad_h = kernel[0] // 2
            #             pad_w = kernel[1] // 2
            #         cpp_code += f"    // {ltype}, layer {layer_idx}\n"
            #         cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
            #         cpp_code += f"    DepthwiseConv2D_{base_file_name}(\n"
            #         cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
            #         cpp_code += f"        depthwiseKernel_{layer_idx}.data(), depthwiseBias_{layer_idx}.data(),\n"
            #         cpp_code += (
            #             f"        {out_shape[2]}, {out_shape[0]}, {out_shape[1]},\n"
            #         )
            #         cpp_code += (
            #             f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
            #         )
            #         cpp_code += f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
            #         cpp_code += f"        {mapped_act}, {alpha});\n\n"
            #         last_layer = f"layer_{layer_idx}_output"
            #         last_shape = out_shape
            #         # debug printing flag
            #         if debug_outputs:
            #             cpp_code += debug_printing(
            #                 layer_idx, ltype, last_shape, last_layer
            #             )
            #         continue
            #     except ValueError as e:
            #         ERROR(ltype, layer_idx, e)
            #         continue
            elif ltype == "DepthwiseConv2D":
                try:
                    kernel = conv_dict.get("kernel_size", (3, 3))
                    strides = conv_dict.get("strides", (1, 1))
                    padding = conv_dict.get("padding", "valid")
                    pad_h = kernel[0] // 2 if padding.lower() == "same" else 0
                    pad_w = kernel[1] // 2 if padding.lower() == "same" else 0
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                        f"    DepthwiseConv2D_{base_file_name}(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                        f"        depthwiseKernel_{layer_idx}.data(), depthwiseBias_{layer_idx}.data(),\n"
                        f"        {out_shape[2]}, {out_shape[0]}, {out_shape[1]},\n"
                        f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                        f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w}, depthMultiplier_{layer_idx},\n"
                        f"        {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "SeparableConv1D":
                try:
                    input_length = in_shape[0] if len(in_shape) > 0 else 1
                    input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                    output_length = out_shape[0] if out_shape else 1
                    output_channels = (
                        out_shape[-1]
                        if out_shape and len(out_shape) > 1
                        else (conv_dict.get("filters") or 1)
                    )
                    kernel_size_val = conv_dict.get("kernel_size", 3)
                    if isinstance(kernel_size_val, (list, tuple)):
                        kernel_size_val = kernel_size_val[0]
                    strides_val = conv_dict.get("strides", 1)
                    if isinstance(strides_val, (list, tuple)):
                        strides_val = strides_val[0]

                    pad = (
                        kernel_size_val // 2
                        if str(conv_dict.get("padding", "valid")).lower() == "same"
                        else 0
                    )
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {output_length * output_channels}> layer_{layer_idx}_output;\n"
                        f"    SeparableConv1D_{base_file_name}<Scalar, {output_channels}, {output_length}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                        f"        sepDepthwise_{layer_idx}.data(), sepPointwise_{layer_idx}.data(), sepPointwiseBias_{layer_idx}.data(),\n"
                        f"        {input_channels}, {input_length}, {kernel_size_val}, {strides_val}, {pad},\n"
                        f"        {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (output_length, output_channels)
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "SeparableConv2D":
                try:
                    kernel = conv_dict.get("kernel_size", (3, 3))
                    strides = conv_dict.get("strides", (1, 1))
                    padding = conv_dict.get("padding", "valid")
                    pad_h = pad_w = 0
                    if padding.lower() == "same":
                        pad_h = kernel[0] // 2
                        pad_w = kernel[1] // 2
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                        f"    SeparableConv2D_{base_file_name}<Scalar, {out_shape[2]}, {out_shape[0]}, {out_shape[1]}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                        f"        sepDepthwise_{layer_idx}.data(), sepPointwise_{layer_idx}.data(), sepPointwiseBias_{layer_idx}.data(),\n"
                        f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                        f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                        f"        {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "Conv1DTranspose" and ltype not in ["Conv1D"]:
                try:
                    input_length = in_shape[0] if len(in_shape) > 0 else 1
                    input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                    output_length = (
                        out_shape[0] if out_shape and len(out_shape) > 0 else 1
                    )
                    output_channels = (
                        out_shape[-1] if out_shape and len(out_shape) > 1 else 1
                    )
                    output_size = output_length * output_channels
                    kernel = conv_dict.get("kernel_size", 3)
                    strides = conv_dict.get("strides", 1)
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {output_size}> layer_{layer_idx}_output;\n"
                        f"    Conv1DTranspose_{base_file_name}<Scalar, {output_channels}, {output_length}>("
                    )
                    kernel_val = (
                        kernel[0] if isinstance(kernel, (tuple, list)) else kernel
                    )
                    strides_val = (
                        strides[0] if isinstance(strides, (tuple, list)) else strides
                    )
                    if strides_val == 2 and kernel_val == 3:
                        pad = 0
                    else:
                        pad = kernel_val // 2
                    cpp_code += (
                        f"\n        layer_{layer_idx}_output.data(), {last_layer}.data(), convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                        f"        {input_channels}, {input_length}, {kernel_val}, {strides_val}, {pad}, {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (
                        (output_length, output_channels)
                        if out_shape
                        else (output_length, output_channels)
                    )
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "Conv2DTranspose" and ltype not in ["Conv2D"]:
                try:
                    kernel = conv_dict.get("kernel_size", (3, 3))
                    strides = conv_dict.get("strides", (1, 1))
                    padding = conv_dict.get("padding", "valid")
                    pad_h = pad_w = 0
                    if padding.lower() == "same":
                        if strides[0] == 2 and kernel[0] == 3:
                            pad_h = 0
                        else:
                            pad_h = kernel[0] // 2

                        if strides[1] == 2 and kernel[1] == 3:
                            pad_w = 0
                        else:
                            pad_w = kernel[1] // 2
                    out_shape = conv_dict.get("out_shape")
                    in_shape = conv_dict.get("in_shape")
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                        f"    Conv2DTranspose_{base_file_name}<Scalar, {out_shape[2]}, {out_shape[0]}, {out_shape[1]}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                        f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                        f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                        f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                        f"        {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "Conv3DTranspose" and ltype not in ["Conv3D"]:
                try:
                    out_shape = conv_dict["out_shape"]
                    in_shape = conv_dict["in_shape"]
                    kd, kh, kw = conv_dict.get("kernel_size", (3, 3, 3))
                    sd, sh, sw = conv_dict.get("strides", (1, 1, 1))
                    pd = ph = pw = 0
                    if conv_dict.get("padding", "valid").lower() == "same":
                        if sd == 2 and kd == 3:
                            pd = 0
                        else:
                            pd = kd // 2

                        if sh == 2 and kh == 3:
                            ph = 0
                        else:
                            ph = kh // 2

                        if sw == 2 and kw == 3:
                            pw = 0
                        else:
                            pw = kw // 2
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]} * {out_shape[3]})> layer_{layer_idx}_output;\n"
                        f"    Conv3DTranspose_{base_file_name}<Scalar, {out_shape[3]}, {out_shape[0]}, {out_shape[1]}, {out_shape[2]}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                        f"        {in_shape[3]}, {in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {kd}, {kh}, {kw}, {sd}, {sh}, {sw}, {pd}, {ph}, {pw}, {mapped_act}, {alpha});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            #############################################################################################
            # -------------------------------------------------------------------------------------------
            # elif ltype == "ConvLSTM2D":
            #     # ConvLSTM2D, layer {layer_idx}

            #     print("error before passing parameters to ConvLSTM2D")

            #     in_h, in_w, in_ch = conv_dict["in_shape"]
            #     out_h, out_w, filt = conv_dict["out_shape"]
            #     k_h, k_w         = conv_dict["kernel_size"]
            #     s_h, s_w         = conv_dict["strides"]
            #     pad_h = k_h//2 if conv_dict["padding"].lower()=="same" else 0
            #     pad_w = k_w//2 if conv_dict["padding"].lower()=="same" else 0

            #     print("error after passing parameters to ConvLSTM2D")

            #     cpp_code += f"    // ConvLSTM2D, layer {layer_idx}\n"
            #     cpp_code += (
            #         f"    static std::array<Scalar, /*time_steps*/ * {filt} * {out_h} * {out_w}> "
            #         f"layer_{layer_idx}_output;\n"
            #     )
            #     cpp_code += (
            #         f"    ConvLSTM2DForward<Scalar>(\n"
            #         f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
            #         f"        convKernel_{layer_idx}.data(), recurrentKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
            #         f"        /* time_steps */, /*in_ch*/ {in_h}, {in_w}, {in_ch},\n"
            #         f"        {filt}, {k_h}, {k_w}, {s_h}, {s_w}, {pad_h}, {pad_w},\n"
            #         f"        {mapped_act}, {recurrent_mapped_act}, {alpha});\n\n"
            #     )
            #     last_layer = f"layer_{layer_idx}_output"
            #     last_shape = (out_h, out_w, filt)
            #     continue
            # -------------------------------------------------------------------------------------------
            #############################################################################################

            ## POOLING LAYERS ##
            elif ltype == "MaxPooling1D":
                try:
                    input_length = in_shape[0] if len(in_shape) > 0 else 1
                    input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                    output_size = (
                        out_shape[0] * out_shape[-1]
                        if out_shape and len(out_shape) > 1
                        else input_channels
                    )

                    pool_size = conv_dict.get("pool_size", 2)
                    strides = conv_dict.get("strides", pool_size)
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {output_size}> layer_{layer_idx}_output;\n"
                        f"    MaxPooling1D_{base_file_name}<Scalar, {pool_size}, {strides}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_length}, {input_channels});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (
                        out_shape
                        if out_shape
                        else (output_size // input_channels, input_channels)
                    )
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "MaxPooling2D":
                try:
                    pool_size = conv_dict.get("pool_size", (2, 2))
                    strides = conv_dict.get("strides", pool_size)
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                        f"    MaxPooling2D_{base_file_name}<Scalar, {pool_size[0]}, {pool_size[1]}, {strides[0]}, {strides[1]}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "MaxPooling3D":
                try:
                    pool_size = conv_dict.get("pool_size", (2, 2, 2))
                    strides = conv_dict.get("strides", pool_size)
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]} * {out_shape[3]})> layer_{layer_idx}_output;\n"
                        f"    MaxPooling3D_{base_file_name}<Scalar, {pool_size[0]}, {pool_size[1]}, {pool_size[2]}, {strides[0]}, {strides[1]}, {strides[2]}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "AvgPooling1D":
                try:
                    input_length = in_shape[0] if len(in_shape) > 0 else 1
                    input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                    output_size = (
                        out_shape[0] * out_shape[-1]
                        if out_shape and len(out_shape) > 1
                        else input_channels
                    )

                    pool_size = conv_dict.get("pool_size", 2)
                    strides = conv_dict.get("strides", pool_size)
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {output_size}> layer_{layer_idx}_output;\n"
                        f"    AvgPooling1D_{base_file_name}<Scalar, {pool_size}, {strides}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_length}, {input_channels});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (
                        out_shape
                        if out_shape
                        else (output_size // input_channels, input_channels)
                    )
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "AvgPooling2D":
                try:
                    pool_size = conv_dict.get("pool_size", (2, 2))
                    strides = conv_dict.get("strides", pool_size)
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                        f"    AvgPooling2D_{base_file_name}<Scalar, {pool_size[0]}, {pool_size[1]}, {strides[0]}, {strides[1]}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "AvgPooling3D":
                try:
                    pool_size = conv_dict.get("pool_size", (2, 2, 2))
                    strides = conv_dict.get("strides", pool_size)
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]} * {out_shape[3]})> layer_{layer_idx}_output;\n"
                        f"    AvgPooling3D_{base_file_name}<Scalar, {pool_size[0]}, {pool_size[1]}, {pool_size[2]}, {strides[0]}, {strides[1]}, {strides[2]}>(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "GlobalMaxPooling1D":
                try:
                    input_length = in_shape[0] if len(in_shape) > 0 else 1
                    input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                        f"    GlobalMaxPooling1D_{base_file_name}(\n"
                        f"         layer_{layer_idx}_output.data(), {last_layer}.data(), {input_length}, {input_channels});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (input_channels,)
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "GlobalMaxPooling2D":
                try:
                    input_height = in_shape[0] if len(in_shape) > 0 else 1
                    input_width = in_shape[1] if len(in_shape) > 1 else 1
                    input_channels = in_shape[2] if len(in_shape) > 2 else 1
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                        f"    GlobalMaxPooling2D_{base_file_name}(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_height}, {input_width}, {input_channels});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (input_channels,)
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "GlobalMaxPooling3D":
                try:
                    input_depth = in_shape[0] if len(in_shape) > 0 else 1
                    input_height = in_shape[1] if len(in_shape) > 1 else 1
                    input_width = in_shape[2] if len(in_shape) > 2 else 1
                    input_channels = in_shape[3] if len(in_shape) > 3 else 1
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                        f"    GlobalMaxPooling3D_{base_file_name}(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                        f"        {in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (in_shape[3],)
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "GlobalAvgPooling1D":
                try:
                    input_length = in_shape[0] if len(in_shape) > 0 else 1
                    input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                        f"    GlobalAvgPooling1D_{base_file_name}(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_length}, {input_channels});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (input_channels,)
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "GlobalAvgPooling2D":
                try:
                    input_height = in_shape[0] if len(in_shape) > 0 else 1
                    input_width = in_shape[1] if len(in_shape) > 1 else 1
                    input_channels = in_shape[2] if len(in_shape) > 2 else 1
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                        f"    GlobalAvgPooling2D_{base_file_name}(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_height}, {input_width}, {input_channels});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (input_channels,)
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

            elif ltype == "GlobalAvgPooling3D":
                try:
                    input_depth = in_shape[0] if len(in_shape) > 0 else 1
                    input_height = in_shape[1] if len(in_shape) > 1 else 1
                    input_width = in_shape[2] if len(in_shape) > 2 else 1
                    input_channels = in_shape[3] if len(in_shape) > 3 else 1
                    cpp_code += (
                        f"    // {ltype}, layer {layer_idx}\n"
                        f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                        f"    GlobalAvgPooling3D_{base_file_name}(\n"
                        f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_depth}, {input_height}, {input_width}, {input_channels});\n\n"
                    )
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (input_channels,)
                    # debug printing flag
                    if debug_outputs:
                        cpp_code += debug_printing(
                            layer_idx, ltype, last_shape, last_layer
                        )
                    continue
                except ValueError as e:
                    ERROR(ltype, layer_idx, e)
                    continue

    cpp_code += "\n/*\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//*/ \n\n\n"

    ## CHECK IF LAST LAYER NEEDS SOFTMAX ##
    if (
        len(activation_functions) > 0
        and activation_functions[-1] == "softmax"
        and (layer_type[-1] if isinstance(layer_type, (list, tuple)) else None)
        != "Activation"
    ):
        if isinstance(last_shape, tuple) and len(last_shape) > 1:
            channels = last_shape[-1]
            length = 1
            for d in last_shape[:-1]:
                length *= d
            cpp_code += (
                f"    // standalone softmax for final layer: "
                f"along last axis (channels={channels}, groups={length})\n"
                f"    for (size_t g = 0; g < {length}; ++g) {{\n"
                f"        softmax({last_layer}.data() + g*{channels}, "
                f"{last_layer}.data() + g*{channels}, {channels});\n"
                "    }\n\n"
            )
        else:
            size = get_flat_size(last_shape)
            cpp_code += (
                f"    // standalone softmax for final layer\n"
                f"    softmax({last_layer}.data(), {last_layer}.data(), {size});\n\n"
            )
        # debug printing flag
        if debug_outputs:
            cpp_code += debug_printing(
                len(activation_functions),
                "SOFTMAX (final layer)",
                last_shape,
                last_layer,
            )

    ## OUTPUT LAYER ##
    out_size = last_shape

    if isinstance(out_size, tuple):
        dims = len(out_size)
        if dims == 1:
            if output_scale is not None:
                cpp_code += (
                    f"    static std::array<Scalar, {out_size[0]}> model_output;\n\n"
                    f"    for (int i = 0; i < {out_size[0]}; i++) {{ model_output[i] = ({last_layer}[i] * output_scale[i]) + output_shift[i]; }}\n\n"
                )
            else:
                cpp_code += f"    static std::array<Scalar, {out_size[0]}> model_output = {last_layer};\n\n"
        elif dims == 2:
            cpp_code += f"    static std::array<std::array<Scalar, {out_size[1]}>, {out_size[0]}> model_output;\n\n"
            if output_scale is not None:
                cpp_code += (
                    f"    for(int i = 0; i < {out_size[0]}; i++) {{\n"
                    f"        for(int j = 0; j < {out_size[1]}; j++) {{\n"
                    f"            int flatIdx = i * {out_size[1]} + j;\n"
                    f"            model_output[i][j] = ({last_layer}[flatIdx] * output_scale[flatIdx]) + output_shift[flatIdx];\n"
                    "        }\n    }\n\n"
                )
            else:
                cpp_code += (
                    f"    for(int i = 0; i < {out_size[0]}; i++) {{\n"
                    f"        for(int j = 0; j < {out_size[1]}; j++) {{\n"
                    f"            model_output[i][j] = {last_layer}[i * {out_size[1]} + j];\n"
                    "        }\n    }\n\n"
                )
        elif dims == 3:
            cpp_code += f"    static std::array<std::array<std::array<Scalar, {out_size[2]}>, {out_size[1]}>, {out_size[0]}> model_output;\n\n"
            if output_scale is not None:
                cpp_code += (
                    f"    for(int i = 0; i < {out_size[0]}; i++) {{\n"
                    f"        for(int j = 0; j < {out_size[1]}; j++) {{\n"
                    f"            for(int k = 0; k < {out_size[2]}; k++) {{\n"
                    f"                int flatIdx = i * {out_size[1] * out_size[2]} + j * {out_size[2]} + k;\n"
                    f"                model_output[i][j][k] = ({last_layer}[flatIdx] * output_scale[flatIdx]) + output_shift[flatIdx];\n"
                    "            }\n        }\n    }\n\n"
                )
            else:
                cpp_code += (
                    f"    for(int i = 0; i < {out_size[0]}; i++) {{\n"
                    f"        for(int j = 0; j < {out_size[1]}; j++) {{\n"
                    f"            for(int k = 0; k < {out_size[2]}; k++) {{\n"
                    f"                model_output[i][j][k] = {last_layer}[i * {out_size[1] * out_size[2]} + j * {out_size[2]} + k];\n"
                    "            }\n        }\n    }\n\n"
                )
    else:
        if output_scale is not None:
            cpp_code += (
                f"    static std::array<Scalar, {out_norm_size}> model_output;\n\n"
                f"    for (int i = 0; i < {out_norm_size}; i++) {{ model_output[i] = ({last_layer}[i] * output_scale[i]) + output_shift[i]; }}\n\n"
            )
        else:
            cpp_code += f"    static std::array<Scalar, {out_size}> model_output = {last_layer};\n\n"

    cpp_code += "    return model_output;\n\n}"

    return cpp_code
