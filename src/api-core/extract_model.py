"""
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import numpy as np
import math
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras


def ERROR(layer_type, layer_idx, e):
    print(f"\n__Error__ in extract_model.py: {layer_type} layer {layer_idx} --> ", e)


def getAlphaForActivation(layer, activation):
    # ===================================================================================
    # function that helps exract the alpha value for LeakyReLU or ELU activations.

    # args:
    #     layer: Keras layer object.
    #     activation: Activation function name or configuration.

    # returns:
    #     alpha value for LeakyReLU or ELU activations, defaulting to 0.1 for LeakyReLU
    #     and 1.0 for ELU, or 0.0 for other activations.
    # ===================================================================================
    if isinstance(layer, keras.layers.LeakyReLU):
        config = layer.get_config()
        alpha = config.get("alpha", config.get("negative_slope", 0.1))
        return alpha
    if isinstance(activation, dict):
        if activation.get("class_name") == "LeakyReLU":
            config = activation.get("config", {})
            alpha = config.get("alpha", config.get("negative_slope", 0.1))
            return alpha
    if activation == "leakyrelu":
        if hasattr(layer, "get_config"):
            config = layer.get_config()
            if "alpha" in config or "negative_slope" in config:
                return config.get("alpha", config.get("negative_slope", 0.1))
        return 0.1
    elif activation == "elu":
        return layer.get_config().get("alpha", 1.0)
    return 0.0


def getActivation(layer, config):
    # ===================================================================================
    # extracts the name of the activation function from a Keras layer.
    # ===================================================================================
    if isinstance(layer, keras.layers.LeakyReLU):
        return "leakyrelu"
    elif isinstance(layer, keras.layers.ReLU):
        return "relu"
    elif isinstance(layer, keras.layers.ELU):
        return "elu"
    elif isinstance(layer, keras.layers.PReLU):
        return "prelu"

    act_name = config.get("activation", None)

    if act_name in (None, "linear"):
        activation_attr = getattr(layer, "activation", None)
        if activation_attr is None:
            return "linear"
        elif hasattr(activation_attr, "__name__"):
            return activation_attr.__name__
        else:
            return str(activation_attr)
    return act_name


def extractModel(model, file_type, base_file_name=None):
    # ===================================================================================
    # function to process and extract model information from a Keras model layer
    # by layer, including layer types, weights, biases, activation functions,
    # alphas, dropout rates, normalization layer parameters, convolution layer parameters,
    # and layer shapes.

    # args:
    #     model: Keras model object.
    #     file_type: File type of the model, e.g., ".h5", ".keras".

    # returns:
    #     a tuple containing lists of layer types, weights, biases, activation functions,
    #     alphas, dropout rates, normalization layer parameters, convolution layer parameters,
    #     and layer shapes.

    # raises:
    #     ValueError: If the model type is not supported or if the file type is not recognized.
    #     TypeError: If the model is not a Keras model.
    #     RuntimeError: If the model input shape cannot be determined.
    #     ImportError: If the required libraries for Keras are not installed.
    #     Exception: If an unexpected error occurs during model extraction.
    #     NotImplementedError: If the model contains unsupported layers or configurations.
    #     KeyError: If a required configuration key is missing in the model layer.
    # ===================================================================================
    layer_type = []
    weights_list = []
    biases_list = []
    activation_functions = []
    alphas = []
    dropout_rates = []
    norm_layer_params = []
    conv_layer_params = []
    layer_shape = []
    activation_configs = []

    if file_type in [".h5", ".keras"]:
        ## INPUT SHAPE ##
        this_input_shape = model.input_shape
        if this_input_shape[0] is None:
            this_raw_intput_shape = this_input_shape[1:]
        else:
            this_raw_intput_shape = this_input_shape
        input_flat_size = int(np.prod(this_raw_intput_shape))
        layer_shape.append(tuple(this_raw_intput_shape))
        current_shape = model.input_shape[1:]
        layer_idx = 0

        #####################
        ## LAYER ITERATION ##
        #####################
        for layer in model.layers:
            layer_idx += 1

            ################################
            ## GET LAYER CONFIG & ACT FUN ##
            ################################
            try:
                layer_input_shape = layer.input_shape
            except AttributeError:
                layer_input_shape = current_shape

            config = layer.get_config()

            layer_weights = layer.get_weights()

            activation = getActivation(layer, config)

            if not isinstance(activation, str):
                activation = activation.get("class_name", "linear").lower()

            raw_act = config.get("activation", "linear")

            if isinstance(raw_act, dict):
                act_name = raw_act["class_name"].lower()
                act_params = raw_act.get("config", {})

            elif raw_act:
                act_name = str(raw_act).lower()
                act_params = {}

            else:
                act_name = layer.__class__.__name__.lower()
                layer_cfg = config
                act_params = {
                    k: layer_cfg[k]
                    for k in ("alpha", "negative_slope", "shared_axes")
                    if k in layer_cfg
                }

            # for custom activation functions
            builtin_activations = [
                "relu",
                "sigmoid",
                "tanh",
                "leakyrelu",
                "linear",
                "elu",
                "selu",
                "swish",
                "prelu",
                "silu",
                "gelu",
                "softmax",
                "mish",
                "softplus",
            ]
            if isinstance(layer, keras.layers.Activation) and hasattr(
                layer.activation, "__name__"
            ):
                custom_act_name = layer.activation.__name__
                if custom_act_name not in builtin_activations:
                    act_name = custom_act_name
            elif hasattr(layer, "activation") and layer.activation is not None:
                if hasattr(layer.activation, "__name__"):
                    activation_name = layer.activation.__name__
                    if activation_name not in builtin_activations:
                        act_name = activation_name
                elif callable(layer.activation) and hasattr(
                    layer.activation, "__name__"
                ):
                    activation_name = layer.activation.__name__
                    if activation_name not in builtin_activations:
                        act_name = activation_name
            if act_name in builtin_activations and hasattr(layer, "get_config"):
                layer_config = layer.get_config()
                activation_config = layer_config.get("activation", None)
                if (
                    isinstance(activation_config, dict)
                    and "class_name" in activation_config
                ):
                    class_name = activation_config["class_name"]
                    if class_name not in builtin_activations:
                        act_name = class_name.lower()

            alpha_value = getAlphaForActivation(layer, activation)

            #######################
            ## ACTIVATION LAYERS ##
            #######################
            # pure activation layers
            if (
                "activation" in layer.name.lower()
                or isinstance(layer, keras.layers.Activation)
                or isinstance(layer, keras.layers.LeakyReLU)
                or isinstance(layer, keras.layers.ReLU)
                or isinstance(layer, keras.layers.ELU)
                or isinstance(layer, keras.layers.PReLU)
            ) or (
                not layer.get_weights()
                and layer.__class__.__name__.lower()
                in [
                    "relu",
                    "sigmoid",
                    "tanh",
                    "leakyrelu",
                    "linear",
                    "elu",
                    "selu",
                    "swish",
                    "prelu",
                    "silu",
                    "gelu",
                    "softmax",
                    "mish",
                    "softplus",
                ]
            ):
                try:
                    if isinstance(layer, keras.layers.LeakyReLU):
                        act_name = "leakyrelu"
                    elif isinstance(layer, keras.layers.ReLU):
                        act_name = "relu"
                    elif isinstance(layer, keras.layers.ELU):
                        act_name = "elu"
                    elif isinstance(layer, keras.layers.PReLU):
                        act_name = "prelu"

                    alpha_value = getAlphaForActivation(layer, act_name)

                    activation_functions.append(act_name)
                    activation_configs.append(act_params)
                    conv_layer_params.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(current_shape)
                    layer_type.append("Activation")
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            #########################
            ## PREPROCESSING LAYERS ##
            #########################
            if (
                isinstance(layer, keras.layers.Rescaling)
                or "rescaling" in layer.name.lower()
            ):
                try:
                    rescale_size = None
                    if current_shape is not None:
                        try:
                            if isinstance(current_shape, tuple):
                                prod = 1
                                for d in current_shape:
                                    prod *= d
                                rescale_size = prod
                            else:
                                rescale_size = int(current_shape)
                        except Exception:
                            rescale_size = None

                    raw_scale = config.get("scale", None)
                    if raw_scale is None:
                        raw_scale = 1.0
                    if isinstance(raw_scale, dict):
                        if "value" in raw_scale:
                            raw_scale = raw_scale["value"]
                        elif (
                            "config" in raw_scale
                            and isinstance(raw_scale["config"], dict)
                            and "value" in raw_scale["config"]
                        ):
                            raw_scale = raw_scale["config"]["value"]

                    arr_scale = np.array(raw_scale, dtype=float)
                    if arr_scale.size == 1:
                        s_val = float(arr_scale)
                        if rescale_size is None or int(rescale_size) == 1:
                            scale = [s_val]
                        else:
                            scale = [s_val] * int(rescale_size)
                    else:
                        scale = arr_scale.flatten().tolist()

                    raw_offset = config.get("offset", None)
                    if raw_offset is None:
                        raw_offset = 0.0
                    if isinstance(raw_offset, dict):
                        if "value" in raw_offset:
                            raw_offset = raw_offset["value"]
                        elif (
                            "config" in raw_offset
                            and isinstance(raw_offset["config"], dict)
                            and "value" in raw_offset["config"]
                        ):
                            raw_offset = raw_offset["config"]["value"]

                    arr_offset = np.array(raw_offset, dtype=float)
                    if arr_offset.size == 1:
                        o_val = float(arr_offset)
                        if rescale_size is None or int(rescale_size) == 1:
                            offset = [o_val]
                        else:
                            offset = [o_val] * int(rescale_size)
                    else:
                        offset = arr_offset.flatten().tolist()

                    norm_layer_params.append((scale, offset))
                    conv_layer_params.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    activation_functions.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                    layer_shape.append(current_shape)
                    layer_type.append("Rescale")
                    activation_configs.append({})
                    continue
                except Exception as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            #################
            ## CORE LAYERS ##
            #################
            if isinstance(layer, keras.layers.Dense) or "dense" in layer.name.lower():
                try:
                    w, b = layer_weights
                    conv_layer_params.append(None)
                    weights_list.append(w)
                    biases_list.append(b)
                    norm_layer_params.append(None)
                    current_shape = (w.shape[1],)
                    layer_shape.append(current_shape)
                    layer_type.append("Dense")
                    activation_functions.append(act_name)
                    activation_configs.append(act_params)
                    alphas.append(alpha_value)
                    dropout_rates.append(config.get("dropout_rate", 0.0))
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            ###########################
            ## REGULARIZATION LAYERS ##
            ###########################
            if (
                isinstance(layer, keras.layers.Dropout)
                or "dropout" in layer.name.lower()
            ):
                try:
                    dropout_rate = config.get("rate", 0.0)
                    activation_functions.append(None)
                    conv_layer_params.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(dropout_rate)
                    layer_shape.append(current_shape)
                    layer_type.append("Dropout")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.SpatialDropout1D)
                or "spatialdropout1d" in layer.name.lower()
            ):
                try:
                    dropout_rate = config.get("rate", 0.0)
                    activation_functions.append(None)
                    conv_layer_params.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(dropout_rate)
                    layer_shape.append(current_shape)
                    layer_type.append("Dropout")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.SpatialDropout2D)
                or "spatialdropout2d" in layer.name.lower()
            ):
                try:
                    dropout_rate = config.get("rate", 0.0)
                    activation_functions.append(None)
                    conv_layer_params.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(dropout_rate)
                    layer_shape.append(current_shape)
                    layer_type.append("Dropout")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.SpatialDropout3D)
                or "spatialdropout3d" in layer.name.lower()
            ):
                try:
                    dropout_rate = config.get("rate", 0.0)
                    activation_functions.append(None)
                    conv_layer_params.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(dropout_rate)
                    layer_shape.append(current_shape)
                    layer_type.append("Dropout")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            ####################
            ## RESHAPE LAYERS ##
            ####################
            if (
                isinstance(layer, keras.layers.Reshape)
                or "reshape" in layer.name.lower()
            ):
                try:
                    if len(layer_weights) == 0:
                        new_shape = config.get("target_shape", None)
                        if new_shape is None:
                            raise ValueError(
                                f"Reshape layer {layer.name} has no target shape defined."
                            )
                        if isinstance(new_shape, list):
                            new_shape = tuple(new_shape)
                        elif isinstance(new_shape, int):
                            new_shape = (new_shape,)
                    else:
                        new_shape = layer_weights[0].shape

                    conv_layer_params.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Reshape")
                    current_shape = new_shape
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.Flatten)
                or "flatten" in layer.name.lower()
            ):
                try:
                    activation_functions.append(None)
                    conv_layer_params.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                    current_shape = (int(np.prod(current_shape)),)
                    layer_shape.append(current_shape)
                    layer_type.append("Flatten")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            ##########################
            ## NORMALIZATION LAYERS ##
            ##########################
            if (
                isinstance(layer, keras.layers.BatchNormalization)
                or "batchnormalization" in layer.name.lower()
            ):
                try:
                    norm_type = "BatchNormalization"
                    if len(layer_weights) == 4:
                        gamma, beta, moving_mean, moving_variance = layer_weights
                        epsilon = config.get("epsilon", 1e-5)
                        norm_layer_params.append(
                            (gamma, beta, moving_mean, moving_variance, epsilon)
                        )
                        layer_shape.append(current_shape)
                        conv_layer_params.append(None)
                        weights_list.append(None)
                        biases_list.append(None)
                        activation_functions.append(None)
                        alphas.append(0.0)
                        dropout_rates.append(0.0)
                        layer_type.append(norm_type)
                        activation_configs.append({})
                    else:
                        conv_layer_params.append(None)
                        norm_layer_params.append(None)
                        activation_functions.append(None)
                        activation_configs.append({})          
                        layer_shape.append(current_shape)
                        layer_type.append(None)
                        alphas.append(0.0)
                        weights_list.append(None)              
                        biases_list.append(None)               
                        dropout_rates.append(0.0)              
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.LayerNormalization)
                or "layernormalization" in layer.name.lower()
            ):
                try:
                    norm_type = "LayerNormalization"
                    if len(layer_weights) == 2:
                        gamma, beta = layer_weights
                        epsilon = config.get("epsilon", 1e-5)
                        norm_layer_params.append((gamma, beta, None, None, epsilon))
                        layer_shape.append(current_shape)
                        activation_functions.append(None)
                        conv_layer_params.append(None)
                        weights_list.append(None)
                        biases_list.append(None)
                        alphas.append(alpha_value)
                        dropout_rates.append(0.0)
                        layer_type.append(norm_type)
                        activation_configs.append({})
                    else:
                        norm_layer_params.append(None)
                        conv_layer_params.append(None)
                        activation_functions.append(None)
                        activation_configs.append({})          
                        layer_shape.append(current_shape)
                        layer_type.append(None)
                        alphas.append(alpha_value)
                        weights_list.append(None)             
                        biases_list.append(None)               
                        dropout_rates.append(0.0)             
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.UnitNormalization)
                or "unitnormalization" in layer.name.lower()
            ):
                try:
                    epsilon = config.get("epsilon", 1e-5)
                    norm_layer_params.append((None, None, None, None, epsilon))
                    layer_shape.append(current_shape)
                    activation_functions.append(None)
                    conv_layer_params.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                    layer_type.append("UnitNormalization")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.GroupNormalization)
                or "groupnormalization" in layer.name.lower()
            ):
                try:
                    norm_type = "GroupNormalization"
                    if len(layer_weights) == 2:
                        gamma, beta = layer_weights
                        epsilon = config.get("epsilon", 1e-3)
                        groups = config.get("groups", 32)
                        norm_layer_params.append(
                            (gamma, beta, None, None, epsilon, groups)
                        )
                        layer_shape.append(current_shape)
                        activation_functions.append(None)
                        conv_layer_params.append(None)
                        weights_list.append(None)
                        biases_list.append(None)
                        alphas.append(alpha_value)
                        dropout_rates.append(0.0)
                        layer_type.append(norm_type)
                        activation_configs.append({})
                    else:
                        norm_layer_params.append(None)
                        activation_functions.append(None)
                        activation_configs.append({})         
                        conv_layer_params.append(None)
                        layer_shape.append(current_shape)
                        layer_type.append(None)
                        alphas.append(alpha_value)
                        weights_list.append(None)             
                        biases_list.append(None)               
                        dropout_rates.append(0.0)              
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            ####################
            ## POOLING LAYERS ##
            ####################
            if (
                isinstance(layer, keras.layers.MaxPooling1D)
                or "maxpooling1d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", 2)
                    pool_size = raw_pool if isinstance(raw_pool, int) else raw_pool[0]
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        raw_strides if isinstance(raw_strides, int) else raw_strides[0]
                    )

                    padding = config.get("padding", "valid")
                    in_shape = current_shape
                    pool_size = (
                        pool_size[0]
                        if isinstance(pool_size, (tuple, list))
                        else pool_size
                    )
                    strides = (
                        strides[0] if isinstance(strides, (tuple, list)) else strides
                    )
                    if padding == "valid":
                        length = math.floor((in_shape[0] - pool_size) / strides) + 1
                    else:
                        length = math.ceil(in_shape[0] / strides)
                    new_shape = (
                        (length,) + tuple(in_shape[1:])
                        if isinstance(in_shape, (tuple, list))
                        else (length,)
                    )

                    pool_params = {
                        "layer_type": layer.__class__.__name__,
                        "pool_size": pool_size,
                        "strides": strides,
                        "padding": padding,
                        "in_shape": in_shape,
                        "output_shape": new_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = new_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("MaxPooling1D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.MaxPooling2D)
                or "maxpooling2d" in layer.name.lower()
            ):
                try:
                    in_shape = current_shape
                    if isinstance(in_shape, tuple) and len(in_shape) == 3:
                        H, W, C = in_shape
                    elif isinstance(in_shape, tuple) and len(in_shape) == 2:
                        H, W = in_shape
                        C = 1
                    else:
                        raise ValueError(
                            f"Unexpected in_shape for {layer.__class__.__name__}: {in_shape}"
                        )

                    raw_pool = config.get("pool_size", (2, 2))
                    pool_size = (
                        (raw_pool, raw_pool)
                        if isinstance(raw_pool, int)
                        else tuple(raw_pool)
                    )

                    raw_strides = config.get("strides", None)
                    if raw_strides is None:
                        raw_strides = pool_size
                    strides = (
                        (raw_strides, raw_strides)
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )

                    padding = config.get("padding", "valid")

                    if padding.lower() == "same":
                        out_H = math.ceil(H / strides[0])
                        out_W = math.ceil(W / strides[1])
                    elif padding.lower() == "valid":
                        out_H = math.floor((H - pool_size[0]) / strides[0]) + 1
                        out_W = math.floor((W - pool_size[1]) / strides[1]) + 1
                    else:
                        out_H, out_W = H, W

                    out_C = C
                    new_shape = (out_H, out_W, out_C)

                    pool_params = {
                        "layer_type": layer.__class__.__name__,
                        "pool_size": pool_size,
                        "strides": strides,
                        "padding": padding,
                        "in_shape": in_shape,
                        "output_shape": new_shape,
                        "out_shape": new_shape,
                    }

                    conv_layer_params.append(pool_params)
                    current_shape = new_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("MaxPooling2D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.MaxPooling3D)
                or "maxpooling3d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", (2, 2, 2))
                    pool_size = (
                        (raw_pool, raw_pool, raw_pool)
                        if isinstance(raw_pool, int)
                        else tuple(raw_pool)
                    )
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        (raw_strides, raw_strides, raw_strides)
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )
                    padding = config.get("padding", "valid")
                    in_shape = current_shape
                    if padding == "valid":
                        d = math.floor((in_shape[0] - pool_size[0]) / strides[0]) + 1
                        h = math.floor((in_shape[1] - pool_size[1]) / strides[1]) + 1
                        w = math.floor((in_shape[2] - pool_size[2]) / strides[2]) + 1
                    else:
                        d = math.ceil(in_shape[0] / strides[0])
                        h = math.ceil(in_shape[1] / strides[1])
                        w = math.ceil(in_shape[2] / strides[2])

                    new_shape = (
                        (d, h, w) + tuple(in_shape[3:])
                        if isinstance(in_shape, (tuple, list))
                        else (d, h, w)
                    )

                    pool_params = {
                        "layer_type": layer.__class__.__name__,
                        "pool_size": pool_size,
                        "strides": strides,
                        "padding": padding,
                        "in_shape": in_shape,
                        "output_shape": new_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = new_shape

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("MaxPooling3D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.AveragePooling1D)
                or "averagepooling1d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", 2)
                    pool_size = raw_pool if isinstance(raw_pool, int) else raw_pool[0]
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        raw_strides if isinstance(raw_strides, int) else raw_strides[0]
                    )
                    padding = config.get("padding", "valid")
                    in_shape = current_shape
                    if padding == "valid":
                        length = math.floor((in_shape[0] - pool_size) / strides) + 1
                    else:
                        length = math.ceil(in_shape[0] / strides)
                    new_shape = (
                        (length,) + tuple(in_shape[1:])
                        if isinstance(in_shape, (tuple, list))
                        else (length,)
                    )

                    pool_params = {
                        "layer_type": layer.__class__.__name__,
                        "pool_size": pool_size,
                        "strides": strides,
                        "padding": padding,
                        "in_shape": in_shape,
                        "output_shape": new_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = new_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("AvgPooling1D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.AveragePooling2D)
                or "averagepooling2d" in layer.name.lower()
            ):
                try:
                    in_shape = current_shape
                    if isinstance(in_shape, tuple) and len(in_shape) == 3:
                        H, W, C = in_shape
                    elif isinstance(in_shape, tuple) and len(in_shape) == 2:
                        H, W = in_shape
                        C = 1
                    else:
                        raise ValueError(
                            f"Unexpected in_shape for {layer.__class__.__name__}: {in_shape}"
                        )

                    raw_pool = config.get("pool_size", (2, 2))
                    pool_size = (
                        (raw_pool, raw_pool)
                        if isinstance(raw_pool, int)
                        else tuple(raw_pool)
                    )

                    raw_strides = config.get("strides", None)
                    if raw_strides is None:
                        raw_strides = pool_size
                    strides = (
                        (raw_strides, raw_strides)
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )

                    padding = config.get("padding", "valid")

                    if padding.lower() == "same":
                        out_H = math.ceil(H / strides[0])
                        out_W = math.ceil(W / strides[1])
                    elif padding.lower() == "valid":
                        out_H = math.floor((H - pool_size[0]) / strides[0]) + 1
                        out_W = math.floor((W - pool_size[1]) / strides[1]) + 1
                    else:
                        out_H, out_W = H, W

                    out_C = C
                    new_shape = (out_H, out_W, out_C)

                    pool_params = {
                        "layer_type": layer.__class__.__name__,
                        "pool_size": pool_size,
                        "strides": strides,
                        "padding": padding,
                        "in_shape": in_shape,
                        "output_shape": new_shape,
                        "out_shape": new_shape,
                    }

                    conv_layer_params.append(pool_params)
                    current_shape = new_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("AvgPooling2D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.AveragePooling3D)
                or "averagepooling3d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", (2, 2, 2))
                    pool_size = (
                        (raw_pool, raw_pool, raw_pool)
                        if isinstance(raw_pool, int)
                        else tuple(raw_pool)
                    )
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        (raw_strides, raw_strides, raw_strides)
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )
                    padding = config.get("padding", "valid")
                    in_shape = current_shape
                    d = in_shape[0]
                    h = in_shape[1]
                    w = in_shape[2]
                    if padding.lower() == "same":
                        out_d = math.ceil(d / strides[0])
                        out_h = math.ceil(h / strides[1])
                        out_w = math.ceil(w / strides[2])
                    elif padding.lower() == "valid":
                        out_d = math.floor((d - pool_size[0]) / strides[0]) + 1
                        out_h = math.floor((h - pool_size[1]) / strides[1]) + 1
                        out_w = math.floor((w - pool_size[2]) / strides[2]) + 1
                    else:
                        out_d, out_h, out_w = d, h, w

                    new_shape = (
                        (out_d, out_h, out_w) + tuple(in_shape[3:])
                        if isinstance(in_shape, (tuple, list))
                        else (out_d, out_h, out_w)
                    )

                    pool_params = {
                        "layer_type": "AvgPooling3D",
                        "pool_size": pool_size,
                        "strides": strides,
                        "padding": padding,
                        "in_shape": in_shape,
                        "output_shape": new_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = new_shape

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("AvgPooling3D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.GlobalMaxPooling1D)
                or "globalmaxpooling1d" in layer.name.lower()
            ):
                try:
                    out_shape = (current_shape[-1],)
                    pool_params = {
                        "layer_type": "GlobalMaxPooling1D",
                        "in_shape": current_shape,
                        "out_shape": out_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = out_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(out_shape)
                    layer_type.append("GlobalMaxPooling1D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.GlobalMaxPooling2D)
                or "globalmaxpooling2d" in layer.name.lower()
            ):
                try:
                    out_shape = (current_shape[-1],)
                    pool_params = {
                        "layer_type": "GlobalMaxPooling2D",
                        "in_shape": current_shape,
                        "out_shape": out_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = out_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(out_shape)
                    layer_type.append("GlobalMaxPooling2D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.GlobalMaxPooling3D)
                or "globalmaxpooling3d" in layer.name.lower()
            ):
                try:
                    out_shape = (current_shape[-1],)
                    pool_params = {
                        "layer_type": "GlobalMaxPooling3D",
                        "in_shape": current_shape,
                        "out_shape": out_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = out_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(out_shape)
                    layer_type.append("GlobalMaxPooling3D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.GlobalAveragePooling1D)
                or "globalaveragepooling1d" in layer.name.lower()
            ):
                try:
                    out_shape = (current_shape[-1],)
                    pool_params = {
                        "layer_type": "GlobalAveragePooling1D",
                        "in_shape": current_shape,
                        "out_shape": out_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = out_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(out_shape)
                    layer_type.append("GlobalAvgPooling1D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.GlobalAveragePooling2D)
                or "globalaveragepooling2d" in layer.name.lower()
            ):
                try:
                    out_shape = (current_shape[-1],)
                    pool_params = {
                        "layer_type": "GlobalAveragePooling2D",
                        "in_shape": current_shape,
                        "out_shape": out_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = out_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(out_shape)
                    layer_type.append("GlobalAvgPooling2D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.GlobalAveragePooling3D)
                or "globalaveragepooling3d" in layer.name.lower()
            ):
                try:
                    out_shape = (current_shape[-1],)
                    pool_params = {
                        "layer_type": "GlobalAveragePooling3D",
                        "in_shape": current_shape,
                        "out_shape": out_shape,
                    }
                    conv_layer_params.append(pool_params)
                    current_shape = out_shape
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(out_shape)
                    layer_type.append("GlobalAvgPooling3D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            ########################
            ## CONVOLUTION LAYERS ##
            ########################
            if (
                isinstance(layer, keras.layers.DepthwiseConv1D)
                or "depthwiseconv1d" in layer.name.lower()
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        depthwise_kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        depthwise_kernel, bias = layer_weights[0], None
                    else:
                        depthwise_kernel, bias = None, None

                    in_C = current_shape[-1]
                    inferred_dm = None
                    if hasattr(layer, "filters") and in_C is not None and in_C > 0:
                        try:
                            inferred_dm = max(1, layer.filters // in_C)
                        except Exception:
                            inferred_dm = None

                    depth_multiplier = config.get(
                        "depth_multiplier",
                        inferred_dm if inferred_dm is not None else 1,
                    )

                    k_raw = config.get("kernel_size", None)
                    if k_raw is None and hasattr(layer, "kernel_size"):
                        k_raw = layer.kernel_size
                    if isinstance(k_raw, (list, tuple)):
                        k = k_raw[0]
                    else:
                        k = k_raw if k_raw is not None else 3

                    s_raw = config.get("strides", None)
                    if s_raw is None and hasattr(layer, "strides"):
                        s_raw = layer.strides
                    if isinstance(s_raw, (list, tuple)):
                        s = s_raw[0]
                    else:
                        s = s_raw if s_raw is not None else 1

                    d_raw = config.get("dilation_rate", None)
                    if d_raw is None and hasattr(layer, "dilation_rate"):
                        d_raw = layer.dilation_rate
                    if isinstance(d_raw, (list, tuple)):
                        d = d_raw[0]
                    else:
                        d = d_raw if d_raw is not None else 1

                    pad = config.get("padding", None)
                    if pad is None and hasattr(layer, "padding"):
                        pad = layer.padding
                    if pad is None:
                        pad = "valid"

                    conv_params = {
                        "layer_type": "DepthwiseConv1D",
                        "depthwise_kernel": depthwise_kernel,
                        "depthwise_bias": bias,
                        "pointwise_kernel": None,
                        "pointwise_bias": None,
                        "filters": depth_multiplier,
                        "kernel_size": k,
                        "strides": s,
                        "padding": pad,
                        "dilation_rate": d,
                        "use_bias": use_bias,
                    }
                    L, C = current_shape
                    eff_k = (k - 1) * d + 1
                    if pad.lower() == "same":
                        out_L = math.ceil(L / s)
                    elif pad.lower() == "valid":
                        out_L = math.floor((L - eff_k) / s) + 1
                    else:
                        out_L = math.ceil(L / s)
                    out_C = C * depth_multiplier
                    new_shape = (out_L, out_C)
                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params.append(conv_params)
                    weights_list.append(None)
                    biases_list.append(None)
                    alphas.append(alpha_value)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("DepthwiseConv1D")
                    activation_configs.append({})
                    continue

                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.DepthwiseConv2D)
                or "depthwiseconv2d" in layer.name.lower()
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        depthwise_kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        depthwise_kernel, bias = layer_weights[0], None
                    else:
                        depthwise_kernel, bias = None, None

                    H, W, C = current_shape
                    kH, kW = config.get("kernel_size", (3, 3))
                    sH, sW = config.get("strides", (1, 1))
                    pad = config.get("padding", "valid")
                    dilH, dilW = config.get("dilation_rate", (1, 1))
                    depth_multiplier = int(config.get("depth_multiplier", 1))

                    if pad.lower() == "same":
                        out_H = math.ceil(H / sH)
                        out_W = math.ceil(W / sW)
                    elif pad.lower() == "valid":
                        out_H = math.floor((H - kH) / sH) + 1
                        out_W = math.floor((W - kW) / sW) + 1
                    else:
                        out_H, out_W = H, W

                    out_C = C * depth_multiplier
                    new_shape = (out_H, out_W, out_C)

                    conv_params = {
                        "layer_type": "DepthwiseConv2D",
                        "depthwise_kernel": depthwise_kernel,
                        "depthwise_bias": bias,
                        "kernel_size": (kH, kW),
                        "strides": (sH, sW),
                        "padding": pad,
                        "dilation_rate": (dilH, dilW),
                        "use_bias": use_bias,
                        "depth_multiplier": depth_multiplier,
                        "in_shape": current_shape,
                        "out_shape": new_shape,
                    }

                    current_shape = new_shape
                    conv_layer_params.append(conv_params)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("DepthwiseConv2D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.SeparableConv1D)
                or "separableconv1d" in layer.name.lower()
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 3:
                        depthwise_kernel, pointwise_kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 2:
                        depthwise_kernel, pointwise_kernel, bias = (
                            layer_weights[0],
                            layer_weights[1],
                            None,
                        )
                    else:
                        depthwise_kernel, pointwise_kernel, bias = None, None, None

                    conv_params = {
                        "layer_type": "SeparableConv1D",
                        "depthwise_kernel": depthwise_kernel,
                        "depthwise_bias": None,
                        "pointwise_kernel": pointwise_kernel,
                        "pointwise_bias": bias,
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", 3),
                        "strides": config.get("strides", 1),
                        "padding": config.get("padding", "valid"),
                        "dilation_rate": config.get("dilation_rate", 1),
                        "use_bias": use_bias,
                    }

                    L, C = (
                        current_shape
                        if len(current_shape) == 2
                        else (current_shape[0], 1)
                    )
                    k = conv_params["kernel_size"]
                    if isinstance(k, (list, tuple)):
                        k = k[0]
                    s = conv_params["strides"]
                    if isinstance(s, (list, tuple)):
                        s = s[0]
                    pad = conv_params["padding"]
                    filt = conv_params.get("filters")

                    if pad.lower() == "same":
                        out_L = math.ceil(L / s)
                    elif pad.lower() == "valid":
                        out_L = math.floor((L - k) / s) + 1
                    else:
                        out_L = L

                    out_C = filt if filt is not None else C
                    new_shape = (out_L, out_C)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params.append(conv_params)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("SeparableConv1D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.SeparableConv2D)
                or "separableconv2d" in layer.name.lower()
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 3:
                        depthwise_kernel, pointwise_kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 2:
                        depthwise_kernel, pointwise_kernel, bias = (
                            layer_weights[0],
                            layer_weights[1],
                            None,
                        )
                    else:
                        depthwise_kernel, pointwise_kernel, bias = None, None, None
                    conv_params = {
                        "layer_type": "SeparableConv2D",
                        "depthwise_kernel": depthwise_kernel,
                        "depthwise_bias": None,
                        "pointwise_kernel": pointwise_kernel,
                        "pointwise_bias": bias,
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", (3, 3)),
                        "strides": config.get("strides", (1, 1)),
                        "padding": config.get("padding", "valid"),
                        "dilation_rate": config.get("dilation_rate", (1, 1)),
                        "use_bias": use_bias,
                    }

                    H, W, C = current_shape
                    kH, kW = conv_params["kernel_size"]
                    sH, sW = conv_params["strides"]
                    pad = conv_params["padding"]
                    filt = conv_params.get("filters")

                    if pad.lower() == "same":
                        out_H = math.ceil(H / sH)
                        out_W = math.ceil(W / sW)
                    elif pad.lower() == "valid":
                        out_H = math.floor((H - kH) / sH) + 1
                        out_W = math.floor((W - kW) / sW) + 1
                    else:
                        out_H, out_W = H, W

                    out_C = filt if filt is not None else C
                    new_shape = (out_H, out_W, out_C)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params.append(conv_params)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("SeparableConv2D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.Conv1DTranspose)
                or ("conv1dtranspose" in layer.name.lower()
                and not isinstance(layer, keras.layers.Conv1D))
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        kernel, bias = layer_weights[0], None
                    else:
                        kernel, bias = None, None

                    in_length = current_shape[0]
                    raw_strides = config.get("strides", 1)
                    raw_kernel_size = config.get("kernel_size", 3)
                    padding = config.get("padding", "valid").lower()
                    filters = config.get("filters", None)
                    strides = (
                        raw_strides[0]
                        if isinstance(raw_strides, (tuple, list))
                        else raw_strides
                    )
                    kernel_size = (
                        raw_kernel_size[0]
                        if isinstance(raw_kernel_size, (tuple, list))
                        else raw_kernel_size
                    )

                    if padding == "same":
                        out_length = in_length * strides
                    else:  # valid
                        out_length = (in_length - 1) * strides + kernel_size

                    new_shape = (out_length, filters)
                    conv_params = {
                        "layer_type": "Conv1DTranspose",
                        "weights": kernel,
                        "biases": bias,
                        "filters": filters,
                        "kernel_size": kernel_size,
                        "strides": strides,
                        "padding": padding,
                        "dilation_rate": config.get("dilation_rate", 1),
                        "use_bias": use_bias,
                        "in_shape": current_shape,
                        "out_shape": new_shape,
                    }
                    current_shape = new_shape
                    conv_layer_params.append(conv_params)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv1DTranspose")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.Conv2DTranspose)
                or ("conv2dtranspose" in layer.name.lower()
                and not isinstance(layer, keras.layers.Conv2D))
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        kernel, bias = layer_weights[0], None
                    else:
                        kernel, bias = None, None

                    conv_params = {
                        "layer_type": "Conv2DTranspose",
                        "weights": kernel,
                        "biases": bias,
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", (3, 3)),
                        "strides": config.get("strides", (1, 1)),
                        "padding": config.get("padding", "valid").lower(),
                        "dilation_rate": config.get("dilation_rate", (1, 1)),
                        "use_bias": use_bias,
                    }
                    in_shape = current_shape

                    if isinstance(in_shape[0], tuple):
                        flat_shape = in_shape[0]
                        if len(flat_shape) >= 2:
                            H, W = flat_shape[:2]
                        else:
                            raise ValueError("Invalid nested shape for Conv2DTranspose")
                        C = in_shape[1] if len(in_shape) > 1 else 1
                    else:
                        if len(in_shape) < 3:
                            H, W = in_shape
                            C = 1
                        else:
                            H, W, C = in_shape

                    kH, kW = conv_params["kernel_size"]
                    sH, sW = conv_params["strides"]
                    pad = conv_params["padding"]
                    filters = conv_params["filters"]

                    if pad.lower() == "same":
                        out_H = H * sH
                        out_W = W * sW
                    elif pad.lower() == "valid":
                        out_H = (H - 1) * sH + kH
                        out_W = (W - 1) * sW + kW
                    else:
                        out_H, out_W = H, W

                    out_C = filters if filters is not None else C
                    new_shape = (out_H, out_W, out_C)

                    conv_params["in_shape"] = in_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params.append(conv_params)

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv2DTranspose")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                isinstance(layer, keras.layers.Conv3DTranspose)
                or ("conv3dtranspose" in layer.name.lower()
                    and not isinstance(layer, keras.layers.Conv3D))
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        kernel, bias = layer_weights[0], None
                    else:
                        kernel, bias = None, None

                    in_shape = current_shape
                    if (
                        isinstance(in_shape, (list, tuple))
                        and len(in_shape) > 0
                        and isinstance(in_shape[0], (list, tuple))
                    ):
                        flat_shape = tuple(in_shape[0])
                        if len(flat_shape) >= 3:
                            in_d, in_h, in_w = flat_shape[:3]
                        else:
                            raise ValueError(
                                f"Invalid nested shape for Conv3DTranspose: {in_shape}"
                            )
                        in_c = in_shape[1] if len(in_shape) > 1 else 1
                    else:
                        if not isinstance(in_shape, (list, tuple)):
                            raise ValueError(
                                f"Unexpected in_shape type for Conv3DTranspose: {type(in_shape)}"
                            )
                        if len(in_shape) >= 4:
                            in_d, in_h, in_w, in_c = in_shape[:4]
                        elif len(in_shape) == 3:
                            in_d, in_h, in_w = in_shape
                            in_c = 1
                        else:
                            raise ValueError(
                                f"Unexpected in_shape for Conv3DTranspose: {in_shape}"
                            )

                    raw_strides = config.get("strides", (1, 1, 1))
                    if isinstance(raw_strides, int):
                        strides = (raw_strides, raw_strides, raw_strides)
                    else:
                        strides = tuple(raw_strides)

                    raw_kernel = config.get("kernel_size", (3, 3, 3))
                    if isinstance(raw_kernel, int):
                        kernel_size = (raw_kernel, raw_kernel, raw_kernel)
                    else:
                        kernel_size = tuple(raw_kernel)

                    padding = config.get("padding", "valid").lower()
                    filters = config.get("filters", None)

                    if padding == "same":
                        out_d = in_d * strides[0]
                        out_h = in_h * strides[1]
                        out_w = in_w * strides[2]
                    else:
                        out_d = (in_d - 1) * strides[0] + kernel_size[0]
                        out_h = (in_h - 1) * strides[1] + kernel_size[1]
                        out_w = (in_w - 1) * strides[2] + kernel_size[2]

                    out_c = filters if filters is not None else in_c
                    new_shape = (out_d, out_h, out_w, out_c)
                    conv_params = {
                        "layer_type": "Conv3DTranspose",
                        "weights": kernel,
                        "biases": bias,
                        "filters": filters,
                        "kernel_size": kernel_size,
                        "strides": strides,
                        "padding": padding,
                        "dilation_rate": config.get("dilation_rate", (1, 1, 1)),
                        "use_bias": use_bias,
                        "in_shape": current_shape,
                        "out_shape": new_shape,
                    }
                    current_shape = new_shape
                    conv_layer_params.append(conv_params)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv3DTranspose")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                (isinstance(layer, keras.layers.Conv1D)
                and not isinstance(layer, keras.layers.Conv1DTranspose))
                or ("conv1d" in layer.name.lower()
                and "conv1dtranspose" not in layer.name.lower())
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        kernel, bias = layer_weights[0], None
                    else:
                        kernel, bias = None, None

                    conv_params = {
                        "layer_type": "Conv1D",
                        "weights": kernel,
                        "biases": bias,
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", None),
                        "strides": config.get("strides", None),
                        "padding": config.get("padding", None),
                        "dilation_rate": config.get("dilation_rate", None),
                        "use_bias": use_bias,
                        "in_shape": None,
                        "out_shape": None,
                    }

                    in_length = current_shape[0]
                    kernel_size = conv_params["kernel_size"]
                    strides = conv_params["strides"]
                    padding = conv_params["padding"]
                    filters = conv_params["filters"]

                    kernel_size = (
                        kernel_size[0]
                        if isinstance(kernel_size, (tuple, list))
                        else kernel_size
                    )
                    strides = (
                        strides[0] if isinstance(strides, (tuple, list)) else strides
                    )

                    conv_params["kernel_size"] = kernel_size
                    conv_params["strides"] = strides

                    if padding == "same":
                        out_length = math.ceil(in_length / strides)
                    elif padding == "valid":
                        out_length = math.floor((in_length - kernel_size) / strides) + 1
                    else:
                        out_length = in_length

                    new_shape = (out_length, filters)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params.append(conv_params)
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv1D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                (isinstance(layer, keras.layers.Conv2D)
                and not isinstance(layer, keras.layers.Conv2DTranspose))
                or ("conv2d" in layer.name.lower()
                and "conv2dtranspose" not in layer.name.lower())
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        kernel, bias = layer_weights[0], None
                    else:
                        kernel, bias = None, None

                    conv_params = {
                        "layer_type": "Conv2D",
                        "weights": kernel,
                        "biases": bias,
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", None),
                        "strides": config.get("strides", None),
                        "padding": config.get("padding", None),
                        "dilation_rate": config.get("dilation_rate", None),
                        "use_bias": use_bias,
                    }

                    if isinstance(current_shape[0], tuple):
                        flat_shape = current_shape[0]
                        if len(flat_shape) >= 2:
                            H, W = flat_shape[:2]
                        else:
                            raise ValueError("Invalid nested shape for Conv2D")
                        C = current_shape[1] if len(current_shape) > 1 else 1
                    else:
                        if len(current_shape) < 3:
                            H, W = current_shape
                            C = 1
                        else:
                            H, W, C = current_shape

                    kH, kW = conv_params["kernel_size"]
                    sH, sW = conv_params["strides"]
                    pad = conv_params["padding"]
                    filt = conv_params["filters"]

                    if pad.lower() == "same":
                        out_H = math.ceil(H / sH)
                        out_W = math.ceil(W / sW)
                    elif pad.lower() == "valid":
                        out_H = math.floor((H - kH) / sH) + 1
                        out_W = math.floor((W - kW) / sW) + 1
                    else:
                        out_H, out_W = H, W

                    out_C = filt if filt is not None else C
                    new_shape = (out_H, out_W, out_C)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params.append(conv_params)

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv2D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            if (
                (isinstance(layer, keras.layers.Conv3D)
                and not isinstance(layer, keras.layers.Conv3DTranspose))
                or ("conv3d" in layer.name.lower()
                and "conv3dtranspose" not in layer.name.lower())
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        kernel, bias = layer_weights[0], None
                    else:
                        kernel, bias = None, None

                    conv_params = {
                        "layer_type": "Conv3D",
                        "weights": kernel,
                        "biases": bias,
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", None),
                        "strides": config.get("strides", None),
                        "padding": config.get("padding", None),
                        "dilation_rate": config.get("dilation_rate", None),
                        "use_bias": use_bias,
                    }
                    D, H, W, C = current_shape
                    kD, kH, kW = conv_params["kernel_size"]
                    sD, sH, sW = conv_params["strides"]
                    pad = conv_params["padding"]
                    filt = conv_params["filters"]

                    if pad.lower() == "same":
                        d = math.ceil(D / sD)
                        h = math.ceil(H / sH)
                        w = math.ceil(W / sW)
                    elif pad.lower() == "valid":
                        d = math.floor((D - kD) / sD) + 1
                        h = math.floor((H - kH) / sH) + 1
                        w = math.floor((W - kW) / sW) + 1
                    else:
                        d, h, w = D, H, W

                    out_C = filt if filt is not None else C
                    new_shape = (d, h, w, out_C)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params.append(conv_params)

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(alpha_value)
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv3D")
                    activation_configs.append({})
                    continue
                except ValueError as e:
                    ERROR(layer.name.lower(), layer_idx, e)
                    continue

            ##################################################################
            # -----------------------------------------------------------------
            # # Section: Process ConvLSTM2D Layers
            # if isinstance(layer, keras.layers.ConvLSTM2D) or "convlstm2d" in layer.name.lower() or "conv_lstm2d" in layer.name.lower():
            #     # pull config
            #     use_bias = config.get("use_bias", True)
            #     # weights come as [kernel, recurrent_kernel, bias] (if use_bias)
            #     if use_bias and len(layer_weights) == 3:
            #         kernel, recurrent_kernel, bias = layer_weights
            #     elif not use_bias and len(layer_weights) == 2:
            #         kernel, recurrent_kernel = layer_weights
            #         bias = None
            #     else:
            #         kernel = recurrent_kernel = bias = None
            #     # if len(layer_weights) >= 3:
            #     #     kernel    = layer_weights[0]
            #     #     recurrent_kernel = layer_weights[1]
            #     #     bias      = layer_weights[2] if use_bias else None
            #     # else:
            #     #     kernel = recurrent_kernel = bias = None

            #     conv_params = {
            #         "layer_type":          "ConvLSTM2D",
            #         "kernel":              kernel,
            #         "recurrent_kernel":    recurrent_kernel,
            #         "bias":                bias,
            #         "filters":             config.get("filters", None),
            #         "kernel_size":         config.get("kernel_size", None),
            #         "strides":             config.get("strides", (1,1)),
            #         "padding":             config.get("padding", "valid"),
            #         "dilation_rate":       config.get("dilation_rate", (1,1)),
            #         "activation":          config.get("activation", "tanh"),
            #         "recurrent_activation":config.get("recurrent_activation", "hard_sigmoid"),
            #         "use_bias":            use_bias,
            #         "dropout":             config.get("dropout", 0.0),
            #         "recurrent_dropout":   config.get("recurrent_dropout", 0.0),
            #         "return_sequences":    config.get("return_sequences", False),
            #         "go_backwards":        config.get("go_backwards", False),
            #         "stateful":            config.get("stateful", False),
            #     }

            #     # compute spatial output shape just like Conv2D
            #     new_shape = compute_output_shape_2d(
            #         current_shape,
            #         conv_params["kernel_size"],
            #         conv_params["strides"],
            #         conv_params["padding"],
            #         filters=conv_params["filters"],
            #     )
            #     conv_params["in_shape"] = current_shape
            #     conv_params["out_shape"] = new_shape
            #     current_shape = new_shape

            #     # store into your lists
            #     conv_layer_params[-1]    = conv_params
            #     weights_list.append(None)
            #     biases_list.append(None)
            #     norm_layer_params.append(None)
            #     activation_functions.append(conv_params["activation"])
            #     alphas.append(0.0)
            #     dropout_rates.append(0.0)
            #     layer_shape.append(new_shape)
            #     layer_type.append("ConvLSTM2D")
            #     continue
            # -----------------------------------------------------------------
            ##################################################################


        ## OUTPUT SHAPE ## 
        try:
            this_output_shape = model.output_shape
            if this_output_shape[0] is None:
                this_raw_output_shape = this_output_shape[1:]
            else:
                this_raw_output_shape = this_output_shape
            output_flat_size = int(np.prod(this_raw_output_shape))
        except ValueError as e:
            ERROR("output shape", layer_idx, e)

    return (
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
    )
