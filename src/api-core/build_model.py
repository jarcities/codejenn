"""
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""


def buildModel(cpp_code, activation_functions, layer_type, base_file_name):
    """
    Generate C++ lambda-based activation functions (with no indentation for the lambdas)
    and normalization functions. ForwardPass also remains as Code 2 style.
    """

    # regular forward pass
    dense_function = {
        "Dense": f"""
template<typename Scalar, int output_size, typename ActFun>
inline void Dense_{base_file_name}(Scalar* __restrict outputs, const Scalar* __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases, int input_size, ActFun activation_function, Scalar alpha) noexcept 
{{
    for(int i = 0; i < output_size; ++i){{
        Scalar sum = 0;
        
        for(int j = 0; j < input_size; ++j){{
            sum += inputs[j] * weights[j * output_size + i];
        }}
        sum += biases[i];
        activation_function(outputs[i], sum, alpha);
    }}
}}
"""
    }

    # reshape functions
    reshape_functions = {
        "Reshape": f"""
template<typename Scalar, int N>
inline void Reshape_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs) noexcept 
{{
    for (int i = 0; i < N; ++i) {{
        outputs[i] = inputs[i];
    }}
}}
"""
    }

    # preprocessing functions
    preprocessing_functions = {
        "Rescale": f"""
template<typename Scalar, int output_size>
inline void Rescale_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict scale, const Scalar * __restrict offset) noexcept 
{{
    for (int i = 0; i < output_size; ++i) {{
        outputs[i] = inputs[i] * scale[i] + offset[i];
    }}
}}
"""
    }

    # lambda activation functions
    lambda_functions = {
        "relu": """
    auto relu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input > 0 ? input : 0;
    };
""",
        "sigmoid": """
    auto sigmoid = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = 1 / (1 + std::exp(-input));
    };
""",
        "tanhCustom": """
    auto tanhCustom = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = std::tanh(input);
    };
""",
        "leakyrelu": """
    auto leakyrelu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input > 0 ? input : alpha * input;
    };
""",
        "linear": """
    auto linear = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input;
    };
""",
        "elu": """
    auto elu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input > 0 ? input : alpha * (std::exp(input) - 1);
    };
""",
        "selu": """
    auto selu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = 1.0507009873554804934193349852946 * (input > 0 ? input : 1.6732632423543772848170429916717 * (std::exp(input) - 1));
    };
""",
        "swish": """
    auto swish = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input / (1 + std::exp(-alpha * input));
    };
""",
        "prelu": """
    auto prelu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input > 0 ? input : alpha * input;
    };
""",
        "silu": """
    auto silu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        auto sigmoid = 1 / (1 + std::exp(-input));
        output = input * sigmoid;
    };
""",
        "gelu": """
    static constexpr Scalar kC0 = 0.044715;
    static constexpr Scalar kSqrt2PiInv = Scalar(0.7978845608028654);
    auto gelu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        Scalar x3 = input * input * input;
        Scalar y  = kSqrt2PiInv * (input + kC0 * x3);
        output     = Scalar(0.5) * input * (Scalar(1) + std::tanh(y));
    };
""",
        "softmax": """
    auto softmax = +[](Scalar* __restrict outputs, const Scalar* __restrict inputs, int size) noexcept
    {
        const Scalar max_val = *std::max_element(inputs, inputs + size);
        Scalar sum = 0;
        for (int i = 0; i < size; ++i) {
            const Scalar e = std::exp(inputs[i] - max_val);
            outputs[i] = e;
            sum += e;
        }
        for (int i = 0; i < size; ++i) {
            outputs[i] /= sum;
        }
    };
""",
        "mish": """
    auto mish = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        Scalar softplus;
        if (input > Scalar(20))          
            softplus = input;
        else if (input < Scalar(-20))    
            softplus = std::exp(input);
        else
            softplus = std::log1p(std::exp(input));
        output = input * std::tanh(softplus);
    };
""",
        "softplus": """
    auto softplus = +[](Scalar& output, Scalar input, Scalar alpha) noexcept
    {
        // output = std::log1p(std::exp(input));
        output = (input > 20) ? input : std::log1p(std::exp(input));
    };
""",
        "custom_act": """
    auto {act_name} = +[](Scalar& output, Scalar input, Scalar index /*can use "alpha" for index*/) noexcept
    {{
        //TODO: implement custom activation '{act_name}'
        printf("WARNING: CUSTOM ACTIVATION HAS NOT BEEN IMPLEMENTED");
        output = input; //default fallback
    }};
""",
    }

    # normalization functions
    normalization_functions = {
        "LayerNormalization": f"""
template <typename Scalar, int channels, int length>
inline void LayerNormalization_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict gamma, const Scalar * __restrict beta, Scalar epsilon) noexcept
{{
    //1d case or MLP case
    if constexpr (length == 1) 
    {{
        Scalar mean = 0;
        Scalar variance = 0;
        
        for (int i = 0; i < channels; ++i) 
        {{
            mean += inputs[i];
        }}
        mean /= channels;
        
        for (int i = 0; i < channels; ++i) 
        {{
            variance += (inputs[i] - mean) * (inputs[i] - mean);
        }}
        variance /= channels;
        
        for (int i = 0; i < channels; ++i) 
        {{
            outputs[i] = gamma[i] * ((inputs[i] - mean) / std::sqrt(variance + epsilon)) + beta[i];
        }}
    }} 
    //multi-dimension case
    else 
    {{
        for (int c = 0; c < channels; ++c) 
        {{
            // Compute mean for this channel across spatial dimensions
            Scalar mean = 0;
            for (int t = 0; t < length; ++t) 
            {{
                int idx = t * channels + c;
                mean += inputs[idx];
            }}
            mean /= length;
            
            Scalar variance = 0;
            for (int t = 0; t < length; ++t) 
            {{
                int idx = t * channels + c;
                variance += (inputs[idx] - mean) * (inputs[idx] - mean);
            }}
            variance /= length;
            
            Scalar scale = gamma[c] / std::sqrt(variance + epsilon);
            Scalar shift = beta[c] - mean * scale;
            
            for (int t = 0; t < length; ++t) 
            {{
                int idx = t * channels + c;
                outputs[idx] = inputs[idx] * scale + shift;
            }}
        }}
    }}
}}
""",
        "BatchNormalization": f"""
template <typename Scalar, int channels, int length>
inline void BatchNormalization_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict gamma, const Scalar * __restrict beta, const Scalar * __restrict mean, const Scalar * __restrict variance, const Scalar epsilon) noexcept
{{
    //1d case or MLP case
    if constexpr (length == 1) 
    {{
        for (int i = 0; i < channels; ++i) 
        {{
            outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
        }}
    }} 
    //multi-dimension case
    else 
    {{
        for (int c = 0; c < channels; ++c) 
        {{
            Scalar scale = gamma[c] / std::sqrt(variance[c] + epsilon);
            Scalar shift = beta[c] - mean[c] * scale;
            
            for (int t = 0; t < length; ++t) 
            {{
                int idx = t * channels + c;
                outputs[idx] = inputs[idx] * scale + shift;
            }}
        }}
    }}
}}
""",
        "UnitNormalization": f"""
template <typename Scalar, int channels, int length>
inline void UnitNormalization_{base_file_name}(Scalar * __restrict outputs,
                              const Scalar * __restrict inputs,
                              Scalar epsilon) noexcept
{{
    //1d case or MLP case
    if constexpr (length == 1) {{
        Scalar sum_sq = 0;
        for (int i = 0; i < channels; ++i) 
        {{
            sum_sq += inputs[i] * inputs[i];
        }}
        Scalar inv_norm = Scalar(1) / std::sqrt(sum_sq + epsilon);
        
        for (int i = 0; i < channels; ++i) 
        {{
            outputs[i] = inputs[i] * inv_norm;
        }}
    }} 
    //multi-dimension case
    else 
    {{
        for (int c = 0; c < channels; ++c) 
        {{
            Scalar sum_sq = 0;
            for (int t = 0; t < length; ++t) 
            {{
                int idx = t * channels + c;
                sum_sq += inputs[idx] * inputs[idx];
            }}
            Scalar inv_norm = Scalar(1) / std::sqrt(sum_sq + epsilon);
            
            for (int t = 0; t < length; ++t) 
            {{
                int idx = t * channels + c;
                outputs[idx] = inputs[idx] * inv_norm;
            }}
        }}
    }}
}}
""",
        "GroupNormalization": f"""
template <typename Scalar, int channels, int length, int groups>
inline void GroupNormalization_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict gamma, const Scalar * __restrict beta, Scalar epsilon) noexcept
{{
    constexpr int channels_per_group = channels / groups;
    constexpr int group_size = channels_per_group * length;
    
    //1d case or MLP case
    if constexpr (length == 1) 
    {{
        for (int g = 0; g < groups; ++g) 
        {{
            int group_start = g * channels_per_group;
            
            Scalar mean = 0;
            for (int i = 0; i < channels_per_group; ++i) 
            {{
                mean += inputs[group_start + i];
            }}
            mean /= channels_per_group;
            
            Scalar variance = 0;
            for (int i = 0; i < channels_per_group; ++i) 
            {{
                variance += (inputs[group_start + i] - mean) * (inputs[group_start + i] - mean);
            }}
            variance /= channels_per_group;
            
            for (int i = 0; i < channels_per_group; ++i) 
            {{
                int idx = group_start + i;
                outputs[idx] = gamma[idx] * ((inputs[idx] - mean) / std::sqrt(variance + epsilon)) + beta[idx];
            }}
        }}
    //multi-dimension case
    }} 
    else 
    {{
        for (int g = 0; g < groups; ++g) 
        {{
            int group_start_channel = g * channels_per_group;
            
            Scalar mean = 0;
            for (int c = 0; c < channels_per_group; ++c) 
            {{
                for (int t = 0; t < length; ++t) 
                {{
                    int idx = t * channels + (group_start_channel + c);
                    mean += inputs[idx];
                }}
            }}
            mean /= group_size;
            
            Scalar variance = 0;
            for (int c = 0; c < channels_per_group; ++c) 
            {{
                for (int t = 0; t < length; ++t) 
                {{
                    int idx = t * channels + (group_start_channel + c);
                    variance += (inputs[idx] - mean) * (inputs[idx] - mean);
                }}
            }}
            variance /= group_size;
            
            Scalar scale = 1 / std::sqrt(variance + epsilon);
            Scalar shift = -mean * scale;
            
            for (int c = 0; c < channels_per_group; ++c) 
            {{
                int param_idx = group_start_channel + c;
                Scalar final_scale = gamma[param_idx] * scale;
                Scalar final_shift = gamma[param_idx] * shift + beta[param_idx];
                
                for (int t = 0; t < length; ++t) 
                {{
                    int idx = t * channels + (group_start_channel + c);
                    outputs[idx] = inputs[idx] * final_scale + final_shift;
                }}
            }}
        }}
    }}
}}
""",
    }

    # convolution functions
    convolution_functions = {
        "Conv1D": f"""
template <typename Scalar, int out_channels, int out_length, typename ActFun>
inline void Conv1D_{base_file_name}(
    Scalar* __restrict outputs,
    const Scalar* __restrict inputs,
    const Scalar* __restrict weights,
    const Scalar* __restrict biases,
    int in_channels, int in_length,
    int kernel_width, int stride, int padding,
    ActFun activation_function, Scalar alpha) noexcept
{{
    for (int ow = 0; ow < out_length; ++ow) 
    {{
        const int in_center = ow * stride - padding;
        for (int oc = 0; oc < out_channels; ++oc) 
        {{
            Scalar sum = biases ? biases[oc] : Scalar(0);
            for (int ic = 0; ic < in_channels; ++ic) 
            {{
                for (int k = 0; k < kernel_width; ++k) 
                {{
                    const int iw = in_center + k;
                    if (iw < 0 || iw >= in_length) continue;
                    // weight layout -> [k, ic, oc]
                    const int w_idx = ((k * in_channels) + ic) * out_channels + oc;
                    const int in_idx = iw * in_channels + ic;
                    sum += inputs[in_idx] * weights[w_idx];
                }}
            }}
            activation_function(outputs[ow * out_channels + oc], sum, alpha);
        }}
    }}
}}
""",
        "Conv1DTranspose": f"""
template <typename Scalar, int out_channels, int out_length, typename ActFun>
inline void Conv1DTranspose_{base_file_name}(
    Scalar *__restrict outputs,
    const Scalar *__restrict inputs,
    const Scalar *__restrict kernels,
    const Scalar *__restrict biases,
    int in_channels, int in_length,
    int kernel_width, int stride, int padding,
    ActFun activation_function, Scalar alpha) noexcept
{{
    // init
    for (int ow = 0; ow < out_length; ++ow)
    {{
        for (int oc = 0; oc < out_channels; ++oc)
        {{
            outputs[ow * out_channels + oc] = biases ? biases[oc] : Scalar(0);
        }}
    }}
    // ow = iw * stride - padding + k
    for (int iw = 0; iw < in_length; ++iw)
    {{
        const int base = iw * stride - padding;

        for (int ic = 0; ic < in_channels; ++ic)
        {{
            const Scalar val = inputs[iw * in_channels + ic];

            for (int k = 0; k < kernel_width; ++k)
            {{
                const int ow = base + k;
                if (ow < 0 || ow >= out_length)
                    continue;

                const int out_base = ow * out_channels;
                const int ker_k_base = k * (out_channels * in_channels); // [k][oc][ic]

                for (int oc = 0; oc < out_channels; ++oc)
                {{
                    const int k_idx = ker_k_base + oc * in_channels + ic; // [k][oc][ic]
                    outputs[out_base + oc] += val * kernels[k_idx];
                }}
            }}
        }}
    }}

    // activation
    const int total = out_length * out_channels;
    for (int i = 0; i < total; ++i)
    {{
        activation_function(outputs[i], outputs[i], alpha);
    }}
}}
""",
        "Conv2D": f"""
template <typename Scalar, int out_channels, int out_height, int out_width, typename ActivationFunc>
inline void Conv2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar *__restrict weights, const Scalar *__restrict biases, int in_channels, int in_height, int in_width, int kernel_height, int kernel_width, int stride_height, int stride_width, int padding_height, int padding_width, ActivationFunc activation_function, Scalar alpha) noexcept
{{
 
    Scalar sum_buf[out_channels];
    const int input_row_stride = in_width * in_channels;
    const int weights_per_khkw = in_channels * out_channels;  
    const int weights_per_kh = kernel_width * weights_per_khkw;

    for (int oh = 0; oh < out_height; ++oh)
    {{
        const int h_origin = oh * stride_height - padding_height;

        for (int ow = 0; ow < out_width; ++ow)
        {{
            for (int oc = 0; oc < out_channels; ++oc) {{
                sum_buf[oc] = biases[oc];
            }}
            const int w_origin = ow * stride_width - padding_width;

            const int kh_min = std::max(0,       -h_origin);
            const int kh_max = std::min(kernel_height, in_height - h_origin);
            const int kw_min = std::max(0,       -w_origin);
            const int kw_max = std::min(kernel_width, in_width - w_origin);

            for (int kh = kh_min; kh < kh_max; ++kh)
            {{
                const int in_h = h_origin + kh;
                const int input_row_offset = in_h * input_row_stride;
                const int weight_kh_offset = kh * weights_per_kh;

                for (int kw = kw_min; kw < kw_max; ++kw)
                {{
                    const int in_w = w_origin + kw;
                    const int input_base = input_row_offset + in_w * in_channels;

                    const int weight_base = weight_kh_offset + (kw * weights_per_khkw);

                    for (int ic = 0; ic < in_channels; ++ic)
                    {{
                        const Scalar input_val = inputs[input_base + ic];
                        const Scalar *w_ptr = weights + weight_base + ic * out_channels;

                        
                        for (int oc = 0; oc < out_channels; ++oc) {{
                            sum_buf[oc] += input_val * w_ptr[oc];
                        }}
                    }}
                }}
            }}

            Scalar *out_pixel_ptr = outputs + ((oh * out_width + ow) * out_channels);
            for (int oc = 0; oc < out_channels; ++oc) {{
                activation_function(out_pixel_ptr[oc], sum_buf[oc], alpha);
            }}
        }}
    }}
}}
""",
        "Conv2DTranspose": f"""
template <typename Scalar, int out_channels, int out_height, int out_width, typename ActFun>
inline void Conv2DTranspose_{base_file_name}(
    Scalar* __restrict outputs,                 
    const Scalar* __restrict inputs,            
    const Scalar* __restrict weights,           
    const Scalar* __restrict biases,           
    int in_channels,
    int in_height, int in_width,
    int kernel_h,  int kernel_w,
    int stride_h,  int stride_w,
    int pad_h,     int pad_w,
    ActFun activation_function, Scalar alpha) noexcept
{{
    // init bias
    for (int oh = 0; oh < out_height; ++oh) 
    {{
        for (int ow = 0; ow < out_width; ++ow) 
        {{
            const int out_base = (oh * out_width + ow) * out_channels;
            if (biases) 
            {{
                for (int oc = 0; oc < out_channels; ++oc)
                    outputs[out_base + oc] = biases[oc];
            }} 
            else 
            {{
                for (int oc = 0; oc < out_channels; ++oc)
                    outputs[out_base + oc] = Scalar(0);
            }}
        }}
    }}

    // scatter add
    for (int ih = 0; ih < in_height; ++ih) 
    {{
        const int oh_base = ih * stride_h - pad_h;

        for (int iw = 0; iw < in_width; ++iw) 
        {{
            const int ow_base = iw * stride_w - pad_w;

            for (int ic = 0; ic < in_channels; ++ic) 
            {{
                const Scalar v = inputs[( (ih * in_width + iw) * in_channels ) + ic];

                //[kh][kw][oc][ic]   
                for (int kh = 0; kh < kernel_h; ++kh) 
                {{
                    const int oh = oh_base + kh;
                    if (oh < 0 || oh >= out_height) continue;

                    for (int kw = 0; kw < kernel_w; ++kw) 
                    {{
                        const int ow = ow_base + kw;
                        if (ow < 0 || ow >= out_width) continue;

                        const int out_pix_base = ((oh * out_width + ow) * out_channels);
                        const int w_spatial_base = (kh * kernel_w + kw) * out_channels;

                        for (int oc = 0; oc < out_channels; ++oc) 
                        {{
                            const int w_base_oc = (w_spatial_base + oc) * in_channels;   
                            outputs[out_pix_base + oc] += v * weights[w_base_oc + ic];
                        }}
                    }}
                }}
            }}
        }}
    }}
    

    // pointwise activation
    for (int oh = 0; oh < out_height; ++oh) 
    {{
        for (int ow = 0; ow < out_width; ++ow) 
        {{
            const int out_base = (oh * out_width + ow) * out_channels;
            for (int oc = 0; oc < out_channels; ++oc) 
            {{
                const Scalar z = outputs[out_base + oc];
                activation_function(outputs[out_base + oc], z, alpha);
            }}
        }}
    }}
}}
""",
        "Conv3D": f"""
template <typename Scalar, int out_channels, int out_depth, int out_height, int out_width, typename ActFun>
inline void Conv3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases,
                   int in_channels, int in_depth, int in_height, int in_width,
                   int kernel_depth, int kernel_height, int kernel_width, int stride_depth, int stride_height, int stride_width,
                   int padding_depth, int padding_height, int padding_width,
                   ActFun activation_function, Scalar alpha) noexcept
{{
    for (int oc = 0; oc < out_channels; ++oc)
    {{
        for (int od = 0; od < out_depth; ++od)
        {{
            for (int oh = 0; oh < out_height; ++oh)
            {{
                for (int ow = 0; ow < out_width; ++ow)
                {{
                    Scalar sum = 0;
                    for (int ic = 0; ic < in_channels; ++ic)
                    {{
                        for (int kd = 0; kd < kernel_depth; ++kd)
                        {{
                            for (int kh = 0; kh < kernel_height; ++kh)
                            {{
                                
                                for (int kw = 0; kw < kernel_width; ++kw)
                                {{
                                    int in_d = od * stride_depth - padding_depth + kd;
                                    int in_h = oh * stride_height - padding_height + kh;
                                    int in_w = ow * stride_width - padding_width + kw;
                                    if (in_d >= 0 && in_d < in_depth &&
                                        in_h >= 0 && in_h < in_height &&
                                        in_w >= 0 && in_w < in_width)
                                    {{
                                        int input_index = ((in_d * in_height * in_width * in_channels) +
                                                           (in_h * in_width * in_channels) +
                                                           (in_w * in_channels) + ic);
                                        int weight_index = (((((kd * kernel_height + kh) * kernel_width + kw) * in_channels + ic) * out_channels) + oc);
                                        sum += inputs[input_index] * weights[weight_index];
                                    }}
                                }}
                            }}
                        }}
                    }}
                    sum += biases[oc];
                    int output_index = ((od * out_height * out_width * out_channels) +
                                        (oh * out_width * out_channels) +
                                        (ow * out_channels) + oc);
                    activation_function(outputs[output_index], sum, alpha);
                }}
            }}
        }}
    }}
}}
""",
        "Conv3DTranspose": f"""
template <typename Scalar, int out_channels, int out_depth, int out_height, int out_width, typename ActFun>
inline void Conv3DTranspose_{base_file_name}(
    Scalar* __restrict outputs,                  
    const Scalar* __restrict inputs,             
    const Scalar* __restrict weights,           
    const Scalar* __restrict biases,            
    int in_channels,
    int in_depth,  int in_height,  int in_width,
    int kernel_d,  int kernel_h,   int kernel_w,
    int stride_d,  int stride_h,   int stride_w,
    int pad_d,     int pad_h,      int pad_w,
    ActFun activation_function, Scalar alpha) noexcept
{{
    // init bias
    for (int od = 0; od < out_depth; ++od) 
    {{
        for (int oh = 0; oh < out_height; ++oh) 
        {{
            for (int ow = 0; ow < out_width; ++ow) 
            {{
                const int out_base = ((od * out_height + oh) * out_width + ow) * out_channels;
                if (biases) 
                {{
                    for (int oc = 0; oc < out_channels; ++oc)
                        outputs[out_base + oc] = biases[oc];
                }} 
                else 
                {{
                    for (int oc = 0; oc < out_channels; ++oc)
                        outputs[out_base + oc] = Scalar(0);
                }}
            }}
        }}
    }}

    // scatter add
    for (int id = 0; id < in_depth; ++id) 
    {{
        const int od_base = id * stride_d - pad_d;

        for (int ih = 0; ih < in_height; ++ih) 
        {{
            const int oh_base = ih * stride_h - pad_h;

            for (int iw = 0; iw < in_width; ++iw) 
            {{
                const int ow_base = iw * stride_w - pad_w;

                for (int ic = 0; ic < in_channels; ++ic) 
                {{
                    const Scalar v = inputs[(((id * in_height + ih) * in_width + iw) * in_channels) + ic];

                    for (int kd = 0; kd < kernel_d; ++kd) 
                    {{
                        const int od = od_base + kd;
                        if (od < 0 || od >= out_depth) continue;

                        for (int kh = 0; kh < kernel_h; ++kh) 
                        {{
                            const int oh = oh_base + kh;
                            if (oh < 0 || oh >= out_height) continue;

                            for (int kw = 0; kw < kernel_w; ++kw) 
                            {{
                                const int ow = ow_base + kw;
                                if (ow < 0 || ow >= out_width) continue;

                                const int out_pix_base = (((od * out_height + oh) * out_width + ow) * out_channels);

                                for (int oc = 0; oc < out_channels; ++oc) 
                                {{
                                    const int w_k_base = (((((kd * kernel_h + kh) * kernel_w + kw) * out_channels) + oc) * in_channels);
                                    outputs[out_pix_base + oc] += v * weights[w_k_base + ic];
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}

    // pointwise activation
    for (int od = 0; od < out_depth; ++od) 
    {{
        for (int oh = 0; oh < out_height; ++oh) 
        {{
            for (int ow = 0; ow < out_width; ++ow) 
            {{
                const int out_base = ((od * out_height + oh) * out_width + ow) * out_channels;
                for (int oc = 0; oc < out_channels; ++oc) 
                {{
                    const Scalar z = outputs[out_base + oc];
                    activation_function(outputs[out_base + oc], z, alpha);
                }}
            }}
        }}
    }}
}}
""",
        "DepthwiseConv1D": f"""
template <typename Scalar, typename ActFun>
inline void DepthwiseConv1D_{base_file_name}(Scalar * __restrict outputs,
                                             const Scalar * __restrict inputs,
                                             const Scalar * __restrict weights,
                                             const Scalar * __restrict biases,
                                             int out_channels, int out_length,
                                             int in_channels,  int in_length,
                                             int kernel_size,  int stride,
                                             int padding,
                                             ActFun activation_function, Scalar alpha) noexcept
{{
    for (int c = 0; c < in_channels; ++c)
    {{
        for (int ol = 0; ol < out_length; ++ol)
        {{
            Scalar sum = 0;
            for (int k = 0; k < kernel_size; ++k)
            {{
                const int il = ol * stride - padding + k;
                if (il >= 0 && il < in_length)
                {{
                    const int input_index  = il * in_channels + c;      
                    const int weight_index = k  * in_channels + c;      
                    sum += inputs[input_index] * weights[weight_index];
                }}
            }}

            sum += biases[c];

            const int output_index = ol * in_channels + c;               
            activation_function(outputs[output_index], sum, alpha);
        }}
    }}
}}
""",
        "DepthwiseConv2D": f"""
template <typename Scalar, typename ActFun>
inline void DepthwiseConv2D_{base_file_name}(Scalar* __restrict outputs,
                                             const Scalar* __restrict inputs,
                                             const Scalar* __restrict weights,
                                             const Scalar* __restrict biases,
                                             int out_channels, int out_height, int out_width,
                                             int in_channels, int in_height, int in_width,
                                             int kernel_height, int kernel_width,
                                             int stride_height, int stride_width,
                                             int padding_height, int padding_width,
                                             int depth_multiplier,
                                             ActFun activation_function, Scalar alpha) noexcept
{{
    // out_channels should equal in_channels * depth_multiplier
    const int outC = in_channels * depth_multiplier;
    (void)out_channels; // optional: assert(out_channels == outC)

    for (int ic = 0; ic < in_channels; ++ic)
    {{
        for (int m = 0; m < depth_multiplier; ++m)
        {{
            const int oc = ic * depth_multiplier + m;

            for (int oh = 0; oh < out_height; ++oh)
            {{
                const int ih0 = oh * stride_height - padding_height;

                for (int ow = 0; ow < out_width; ++ow)
                {{
                    const int iw0 = ow * stride_width - padding_width;

                    Scalar sum = 0;

                    for (int kh = 0; kh < kernel_height; ++kh)
                    {{
                        const int ih = ih0 + kh;
                        if (ih < 0 || ih >= in_height) continue;

                        for (int kw = 0; kw < kernel_width; ++kw)
                        {{
                            const int iw = iw0 + kw;
                            if (iw < 0 || iw >= in_width) continue;
                            const int in_idx = (ih * in_width + iw) * in_channels + ic;
                            const int w_idx =
                                (((kh * kernel_width + kw) * in_channels) + ic) * depth_multiplier + m;

                            sum += inputs[in_idx] * weights[w_idx];
                        }}
                    }}
                    sum += biases ? biases[oc] : Scalar(0);
                    const int out_idx = (oh * out_width + ow) * outC + oc;
                    activation_function(outputs[out_idx], sum, alpha);
                }}
            }}
        }}
    }}
}}
""",
        "SeparableConv1D": f"""
template <typename Scalar>
inline void DepthwiseForsSeparableConv1D(Scalar *__restrict outputs, const Scalar *__restrict inputs, const Scalar *__restrict weights, const Scalar *__restrict biases,
                                  int out_length,
                                  int in_channels, int in_length,
                                  int kernel_size, int stride,
                                  int padding) noexcept
{{
    for (int c = 0; c < in_channels; ++c)
    {{
        for (int ol = 0; ol < out_length; ++ol)
        {{
            Scalar sum = 0;
            for (int k = 0; k < kernel_size; ++k)
            {{
                int in_l = ol * stride - padding + k;
                if (in_l >= 0 && in_l < in_length)
                {{
                    int input_index = (in_l * in_channels) + c;
                    int weight_index = k * in_channels + c;
                    sum += inputs[input_index] * weights[weight_index];
                }}
            }}
            sum += biases[c];
            int output_index = (ol * in_channels) + c;
            outputs[output_index] = sum;
        }}
    }}
}}

template <typename Scalar, int out_channels, int out_length, typename ActFun>
inline void SeparableConv1D_{base_file_name}(
    Scalar *__restrict outputs,
    const Scalar *__restrict inputs,
    const Scalar *__restrict depthwise_weights,
    const Scalar *__restrict pointwise_weights,
    const Scalar *__restrict biases,
    int in_channels, int in_length,
    int kernel_size,
    int stride,
    int padding,
    ActFun activation_function, Scalar alpha) noexcept
{{
    std::vector<Scalar> depthwise_output(out_length * in_channels, 0);
    std::vector<Scalar> zero_bias(in_channels, 0);
    DepthwiseForsSeparableConv1D(
        depthwise_output.data(), inputs, depthwise_weights, zero_bias.data(), out_length,
        in_channels, in_length,
        kernel_size,
        stride,
        padding);
    for (int oc = 0; oc < out_channels; ++oc)
    {{
        for (int i = 0; i < out_length; ++i)
        {{
            Scalar sum = 0;
            
            for (int ic = 0; ic < in_channels; ++ic)
            {{
                int index = i * in_channels + ic;
                int weight_index = ic * out_channels + oc;
                sum += depthwise_output[index] * pointwise_weights[weight_index];
            }}
            sum += biases[oc];
            int output_index = i * out_channels + oc;
            activation_function(outputs[output_index], sum, alpha);
        }}
    }}
}}
""",
        "SeparableConv2D": f"""
template <typename Scalar>
inline void DepthwiseForsSeparableConv2D(Scalar *__restrict outputs, const Scalar *__restrict inputs, const Scalar *__restrict weights, const Scalar *__restrict biases,
                                  int out_height, int out_width,
                                  int in_channels, int in_height, int in_width,
                                  int kernel_height, int kernel_width, int stride_height, int stride_width,
                                  int padding_height, int padding_width) noexcept
{{
    for (int c = 0; c < in_channels; ++c)
    {{
        for (int oh = 0; oh < out_height; ++oh)
        {{
            for (int ow = 0; ow < out_width; ++ow)
            {{
                Scalar sum = 0;
                for (int kh = 0; kh < kernel_height; ++kh)
                {{
                    
                    for (int kw = 0; kw < kernel_width; ++kw)
                    {{
                        int in_h = oh * stride_height - padding_height + kh;
                        int in_w = ow * stride_width - padding_width + kw;
                        if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width)
                        {{
                            int input_index = (in_h * in_width * in_channels) + (in_w * in_channels) + c;
                            int weight_index = (kh * kernel_width + kw) * in_channels + c;
                            sum += inputs[input_index] * weights[weight_index];
                        }}
                    }}
                }}
                sum += biases[c];
                int output_index = ((oh * out_width + ow) * in_channels) + c;
                outputs[output_index] = sum;
            }}
        }}
    }}
}}

template <typename Scalar, int out_channels, int out_height, int out_width, typename ActFun>
inline void SeparableConv2D_{base_file_name}(
    Scalar *__restrict outputs,
    const Scalar *__restrict inputs,
    const Scalar *__restrict depthwise_weights,
    const Scalar *__restrict pointwise_weights,
    const Scalar *__restrict biases,
    int in_channels, int in_height, int in_width,
    int kernel_height, int kernel_width,
    int stride_height, int stride_width,
    int padding_height, int padding_width,
    ActFun activation_function, Scalar alpha) noexcept
{{
    std::vector<Scalar> depthwise_output(out_height * out_width * in_channels, 0);
    std::vector<Scalar> zero_bias(in_channels, 0);
    DepthwiseForsSeparableConv2D(
        depthwise_output.data(), inputs, depthwise_weights, zero_bias.data(), out_height, out_width,
        in_channels, in_height, in_width,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_height, padding_width);
    for (int oc = 0; oc < out_channels; ++oc)
    {{
        for (int i = 0; i < out_height * out_width; ++i)
        {{
            Scalar sum = 0;
            
            for (int ic = 0; ic < in_channels; ++ic)
            {{
                int index = i * in_channels + ic;
                int weight_index = ic * out_channels + oc;
                sum += depthwise_output[index] * pointwise_weights[weight_index];
            }}
            sum += biases[oc];
            int output_index = i * out_channels + oc;
            activation_function(outputs[output_index], sum, alpha);
        }}
    }}
}}
""",
        "ConvLSTM2D": f"""
template<typename Scalar, typename ActFun>
inline void ConvLSTM2D_{base_file_name}(Scalar* __restrict outputs,
                       const Scalar* __restrict inputs,
                       const Scalar* __restrict kernel,
                       const Scalar* __restrict recurrent_kernel,
                       const Scalar* __restrict bias,
                       int time_steps,
                       int in_channels,
                       int in_height,
                       int in_width,
                       int filters,
                       int kernel_height,
                       int kernel_width,
                       int stride_height,
                       int stride_width,
                       int padding_height,
                       int padding_width,
                       ActFun activation_function,
                       activationFunction<Scalar> recurrent_activation_function,
                       Scalar alpha) noexcept
{{
    // hidden + cell state buffers
    std::vector<Scalar> h_state(filters * in_height * in_width, Scalar(0));
    std::vector<Scalar> c_state(filters * in_height * in_width, Scalar(0));
    int spatial_size = in_height * in_width;

    for (int t = 0; t < time_steps; ++t) {{
        // slice for time step t
        const Scalar* x_t = inputs + t * in_channels * spatial_size;

        // gate buffers
        std::vector<Scalar> i_gate(filters * spatial_size, 0);
        std::vector<Scalar> f_gate(filters * spatial_size, 0);
        std::vector<Scalar> g_gate(filters * spatial_size, 0);
        std::vector<Scalar> o_gate(filters * spatial_size, 0);

        // --- Input convolution for all 4 gates ---
        for (int oc = 0; oc < filters; ++oc) {{
            for (int oh = 0; oh < in_height; ++oh) {{
                for (int ow = 0; ow < in_width; ++ow) {{
                    int out_idx = oc + filters * (ow + in_width * oh);
                    Scalar sum_i = 0, sum_f = 0, sum_g = 0, sum_o = 0;
                    for (int ic = 0; ic < in_channels; ++ic) {{
                        for (int kh = 0; kh < kernel_height; ++kh) {{
                            
                            for (int kw = 0; kw < kernel_width; ++kw) {{
                                int ih = oh * stride_height - padding_height + kh;
                                int iw = ow * stride_width  - padding_width  + kw;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {{
                                    int in_idx = ic + in_channels * (iw + in_width * ih);
                                    int k_base = ((kh * kernel_width + kw) * in_channels + ic);
                                    // kernel layout: [kh,kw,in_channels,4*filters]
                                    sum_i += x_t[in_idx] * kernel[(k_base * 4 + 0) * filters + oc];
                                    sum_f += x_t[in_idx] * kernel[(k_base * 4 + 1) * filters + oc];
                                    sum_g += x_t[in_idx] * kernel[(k_base * 4 + 2) * filters + oc];
                                    sum_o += x_t[in_idx] * kernel[(k_base * 4 + 3) * filters + oc];
                                }}
                            }}
                        }}
                    }}
                    i_gate[out_idx] = sum_i;
                    f_gate[out_idx] = sum_f;
                    g_gate[out_idx] = sum_g;
                    o_gate[out_idx] = sum_o;
                }}
            }}
        }}

        // --- Recurrent convolution and bias add ---
        for (int oc = 0; oc < filters; ++oc) {{
            for (int oh = 0; oh < in_height; ++oh) {{
                for (int ow = 0; ow < in_width; ++ow) {{
                    int idx = oc + filters * (ow + in_width * oh);
                    Scalar rec_i = 0, rec_f = 0, rec_g = 0, rec_o = 0;
                    for (int kc = 0; kc < filters; ++kc) {{
                        for (int kh = 0; kh < kernel_height; ++kh) {{
                            
                            for (int kw = 0; kw < kernel_width; ++kw) {{
                                int ih = oh * stride_height - padding_height + kh;
                                int iw = ow * stride_width  - padding_width  + kw;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {{
                                    int h_idx = kc + filters * (iw + in_width * ih);
                                    int rk_base = ((kh * kernel_width + kw) * filters + kc);
                                    rec_i += h_state[h_idx] * recurrent_kernel[(rk_base * 4 + 0) * filters + oc];
                                    rec_f += h_state[h_idx] * recurrent_kernel[(rk_base * 4 + 1) * filters + oc];
                                    rec_g += h_state[h_idx] * recurrent_kernel[(rk_base * 4 + 2) * filters + oc];
                                    rec_o += h_state[h_idx] * recurrent_kernel[(rk_base * 4 + 3) * filters + oc];
                                }}
                            }}
                        }}
                    }}
                    i_gate[idx] += rec_i + bias[oc];
                    f_gate[idx] += rec_f + bias[filters + oc];
                    g_gate[idx] += rec_g + bias[2 * filters + oc];
                    o_gate[idx] += rec_o + bias[3 * filters + oc];
                }}
            }}
        }}

        // --- Activation + state update + write outputs ---
        for (int idx = 0; idx < filters * spatial_size; ++idx) {{
            recurrent_activation_function(i_gate[idx], i_gate[idx], alpha);
            recurrent_activation_function(f_gate[idx], f_gate[idx], alpha);
            activation_function(g_gate[idx], g_gate[idx], alpha);
            recurrent_activation_function(o_gate[idx], o_gate[idx], alpha);

            // cell state
            c_state[idx] = f_gate[idx] * c_state[idx] + i_gate[idx] * g_gate[idx];
            // hidden state
            Scalar c_act;
            activation_function(c_act, c_state[idx], alpha);
            h_state[idx] = o_gate[idx] * c_act;

            // write out
            outputs[t * filters * spatial_size + idx] = h_state[idx];
        }}
    }}
}}
""",
    }

    pooling_functions = {
        "MaxPooling1D": f"""
template <typename Scalar, int pool_size, int stride>
inline void MaxPooling1D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_length, int channels) noexcept
{{
    int out_length = (in_length - pool_size) / stride + 1;
    for (int c = 0; c < channels; ++c)
    {{
        for (int o = 0; o < out_length; ++o)
        {{
            Scalar max_val = inputs[(o * stride * channels) + c];
            
            for (int p = 0; p < pool_size; ++p)
            {{
                int idx = ((o * stride + p) * channels) + c;
                if (inputs[idx] > max_val)
                {{
                    max_val = inputs[idx];
                }}
            }}
            outputs[o * channels + c] = max_val;
        }}
    }}
}}
""",
        "MaxPooling2D": f"""
template <typename Scalar, int pool_height, int pool_width, int stride_h, int stride_w>
inline void MaxPooling2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_height, int in_width, int channels) noexcept
{{
    int out_height = (in_height - pool_height) / stride_h + 1;
    int out_width = (in_width - pool_width) / stride_w + 1;
    for (int c = 0; c < channels; ++c)
    {{
        for (int oh = 0; oh < out_height; ++oh)
        {{
            for (int ow = 0; ow < out_width; ++ow)
            {{
                Scalar max_val = -std::numeric_limits<Scalar>::infinity();
                for (int ph = 0; ph < pool_height; ++ph)
                {{
                    
                    for (int pw = 0; pw < pool_width; ++pw)
                    {{
                        int in_h = oh * stride_h + ph;
                        int in_w = ow * stride_w + pw;
                        int idx = (in_h * in_width * channels) + (in_w * channels) + c;
                        if (inputs[idx] > max_val)
                        {{
                            max_val = inputs[idx];
                        }}
                    }}
                }}
                int out_idx = (oh * out_width * channels) + (ow * channels) + c;
                outputs[out_idx] = max_val;
            }}
        }}
    }}
}}
""",
        "MaxPooling3D": f"""
template <typename Scalar, int pool_depth, int pool_height, int pool_width, int stride_depth, int stride_height, int stride_width>
inline void MaxPooling3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_depth, int in_height, int in_width, int channels) noexcept
{{
    int out_depth = (in_depth - pool_depth) / stride_depth + 1;
    int out_height = (in_height - pool_height) / stride_height + 1;
    int out_width = (in_width - pool_width) / stride_width + 1;
    for (int c = 0; c < channels; ++c) {{
        for (int d = 0; d < out_depth; ++d) {{
            for (int h = 0; h < out_height; ++h) {{
                for (int w = 0; w < out_width; ++w) {{
                    Scalar max_val = inputs[(((d * stride_depth * in_height + h * stride_height) * in_width + w * stride_width) * channels) + c];
                    for (int pd = 0; pd < pool_depth; ++pd) {{
                        for (int ph = 0; ph < pool_height; ++ph) {{
                            
                            for (int pw = 0; pw < pool_width; ++pw) {{
                                int idx = ((((d * stride_depth + pd) * in_height + (h * stride_height + ph)) * in_width + (w * stride_width + pw)) * channels) + c;
                                if (inputs[idx] > max_val) {{
                                    max_val = inputs[idx];
                                }}
                            }}
                        }}
                    }}
                    outputs[(((d * out_height + h) * out_width + w) * channels) + c] = max_val;
                }}
            }}
        }}
    }}
}}
""",
        "AvgPooling1D": f"""
template <typename Scalar, int pool_size, int stride>
inline void AvgPooling1D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_length, int channels) noexcept
{{
    int out_length = (in_length - pool_size) / stride + 1;
    for (int c = 0; c < channels; ++c)
    {{
        for (int o = 0; o < out_length; ++o)
        {{
            Scalar sum = 0;
            
            for (int p = 0; p < pool_size; ++p)
            {{
                int idx = ((o * stride + p) * channels) + c;
                sum += inputs[idx];
            }}
            outputs[o * channels + c] = sum / pool_size;
        }}
    }}
}}
""",
        "AvgPooling2D": f"""
template <typename Scalar, int pool_height, int pool_width, int stride_h, int stride_w>
inline void AvgPooling2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_height, int in_width, int channels) noexcept
{{
    int out_height = (in_height - pool_height) / stride_h + 1;
    int out_width = (in_width - pool_width) / stride_w + 1;
    for (int c = 0; c < channels; ++c)
    {{
        for (int oh = 0; oh < out_height; ++oh)
        {{
            for (int ow = 0; ow < out_width; ++ow)
            {{
                Scalar sum = 0;
                for (int ph = 0; ph < pool_height; ++ph)
                {{
                    
                    for (int pw = 0; pw < pool_width; ++pw)
                    {{
                        int in_h = oh * stride_h + ph;
                        int in_w = ow * stride_w + pw;
                        int idx = (in_h * in_width * channels) + (in_w * channels) + c;
                        sum += inputs[idx];
                    }}
                }}
                int out_idx = (oh * out_width * channels) + (ow * channels) + c;
                outputs[out_idx] = sum / (pool_height * pool_width);
            }}
        }}
    }}
}}
""",
        "AvgPooling3D": f"""
template <typename Scalar, int pool_depth, int pool_height, int pool_width, int stride_depth, int stride_height, int stride_width>
inline void AvgPooling3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_depth, int in_height, int in_width, int channels) noexcept
{{
    int out_depth = (in_depth - pool_depth) / stride_depth + 1;
    int out_height = (in_height - pool_height) / stride_height + 1;
    int out_width = (in_width - pool_width) / stride_width + 1;
    for (int c = 0; c < channels; ++c) {{
        for (int d = 0; d < out_depth; ++d) {{
            for (int h = 0; h < out_height; ++h) {{
                for (int w = 0; w < out_width; ++w) {{
                    Scalar sum = 0;
                    for (int pd = 0; pd < pool_depth; ++pd) {{
                        for (int ph = 0; ph < pool_height; ++ph) {{
                            
                            for (int pw = 0; pw < pool_width; ++pw) {{
                                int idx = ((((d * stride_depth + pd) * in_height + (h * stride_height + ph)) * in_width + (w * stride_width + pw)) * channels) + c;
                                sum += inputs[idx];
                            }}
                        }}
                    }}
                    outputs[(((d * out_height + h) * out_width + w) * channels) + c] = sum / (pool_depth * pool_height * pool_width);
                }}
            }}
        }}
    }}
}}   
""",
        "GlobalMaxPooling1D": f"""
template <typename Scalar>
inline void GlobalMaxPooling1D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_length, int channels) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        Scalar max_val = -std::numeric_limits<Scalar>::infinity();
        
        for (int i = 0; i < in_length; ++i)
        {{
            int idx = (i * channels) + c;
            if (inputs[idx] > max_val)
            {{
                max_val = inputs[idx];
            }}
        }}
        outputs[c] = max_val;
    }}
}}
""",
        "GlobalMaxPooling2D": f"""
template <typename Scalar>
inline void GlobalMaxPooling2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_height, int in_width, int channels) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        Scalar max_val = -std::numeric_limits<Scalar>::infinity();
        for (int h = 0; h < in_height; ++h)
        {{
            
            for (int w = 0; w < in_width; ++w)
            {{
                int idx = (h * in_width * channels) + (w * channels) + c;
                if (inputs[idx] > max_val)
                {{
                    max_val = inputs[idx];
                }}
            }}
        }}
        outputs[c] = max_val;
    }}
}}
""",
        "GlobalMaxPooling3D": f"""
template <typename Scalar>
inline void GlobalMaxPooling3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_depth, int in_height, int in_width, int channels) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        Scalar max_val = -std::numeric_limits<Scalar>::infinity();
        for (int d = 0; d < in_depth; ++d)
        {{
            for (int h = 0; h < in_height; ++h)
            {{
                
                for (int w = 0; w < in_width; ++w)
                {{
                    int idx = ((d * in_height * in_width * channels) + (h * in_width * channels) + (w * channels) + c);
                    if (inputs[idx] > max_val)
                    {{
                        max_val = inputs[idx];
                    }}
                }}
            }}
        }}
        outputs[c] = max_val;
    }}
}}
""",
        "GlobalAvgPooling1D": f"""
template <typename Scalar>
inline void GlobalAvgPooling1D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_length, int channels)
{{
    for (int ic = 0; ic < channels; ic++)
    {{
        Scalar sum = 0;

        for (int i = 0; i < in_length; i++)
        {{
            int idx = (i * channels) + ic;
            sum += inputs[idx];
        }}
        outputs[ic] = sum / (in_length);
    }}
}}
""",
        "GlobalAvgPooling2D": f"""
template <typename Scalar>
inline void GlobalAvgPooling2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_height, int in_width, int channels) noexcept
{{
    // Compute global average per channel.
    for (int c = 0; c < channels; ++c)
    {{
        Scalar sum = 0;
        for (int h = 0; h < in_height; ++h)
        {{
            
            for (int w = 0; w < in_width; ++w)
            {{
                int idx = (h * in_width * channels) + (w * channels) + c;
                sum += inputs[idx];
            }}
        }}
        outputs[c] = sum / (in_height * in_width);
    }}
}}
""",
        "GlobalAvgPooling3D": f"""
template <typename Scalar>
inline void GlobalAvgPooling3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_depth, int in_height, int in_width, int channels) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        Scalar sum = 0;
        for (int d = 0; d < in_depth; ++d)
        {{
            for (int h = 0; h < in_height; ++h)
            {{
                
                for (int w = 0; w < in_width; ++w)
                {{
                    int idx = ((d * in_height * in_width * channels) + (h * in_width * channels) + (w * channels) + c);
                    sum += inputs[idx];
                }}
            }}
        }}
        outputs[c] = sum / (in_depth * in_height * in_width);
    }}
}}
""",
    }

    try:
        current_activations = set(activation_functions)
        current_activations = {
            ("tanhCustom" if act == "tanh" else act)
            for act in current_activations
            if act is not None and act != "Activation"
        }
        current_activations.add("linear")

        cpp_lambda = """"""

        ## SET ACTIVATION FUNCTION ##
        for act in current_activations:
            if act in lambda_functions:
                cpp_lambda += lambda_functions[act]
            ## CUSTOM ACTIVATION FUNCTION ##
            else:
                custom_lambda = lambda_functions["custom_act"].format(act_name=act)
                cpp_lambda += custom_lambda

        # deduplicate layer_type list
        unique_layer_types = {lt for lt in layer_type if lt is not None}

        for type in unique_layer_types:
            if type in preprocessing_functions:
                cpp_code += preprocessing_functions[type]
            if type in dense_function:
                cpp_code += dense_function[type]
            if type in reshape_functions:
                cpp_code += reshape_functions[type]
            if type in normalization_functions:
                cpp_code += normalization_functions[type]
            if type in convolution_functions:
                cpp_code += convolution_functions[type]
            if type in pooling_functions:
                cpp_code += pooling_functions[type]
    except ValueError as e:
        print(f"__Error__ in build_model.py -> {e}")

    return cpp_code, cpp_lambda
