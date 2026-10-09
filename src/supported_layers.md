<!-- 
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
-->

## 1. **Core, Reshape, Regularization Layers**

These are the fundamental layers used in almost every neural network.

- **Dense**: Fully connected layer.

- **Rescaling**: Rescale layers by a new range between [0, 1] or [-1, 1]

- **Activation**: Applies an activation function.

- **LeakyReLU**, **ELU**, **ReLU**, **PReLU**: Activation functions as their own layer.

- **Dropout**, **SpatialDropout1D**, **SpatialDropout2D**, **SpatialDropout3D**: Randomly sets input units to 0 during training to prevent overfitting.

- **Flatten**: Flattens the input without affecting the batch size.

- **Reshape**: Reshapes an output to a certain shape.

*Note:*

**Dense** does not support `use_bias=False`. **ReLU** as its own layer does not support `max_value`, `negative_slope`, or `threshold`. **Softmax** as its own layer is not supported, use `Activation('softmax')` or `activation='softmax'` instead. **Reshape** does not support a `-1` dimension in `target_shape`.

*Example:*

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Reshape

model = Sequential([
    Input(input_shape=(784,)),
    Rescaling(scale=1/value_std, offset=-value_mean/value_std),
    Dense(128, input_shape=(784,)),
    Activation('relu'),
    Dropout(0.2),
    Dense(64),
    Activation('relu'),
    Flatten(),
    Reshape((8, 8, 1)),
    Dense(10, activation='softmax')
])
```

## 2. **Convolutional Layers**

Used primarily for processing grid-like data such as images.

- **Conv1D, Conv2D, Conv3D**: Convolution layers for 1D, 2D, and 3D inputs.

- **Conv1DTranspose, Conv2DTranspose, and Conv3DTranspose**: Transposed convolution (deconvolution).

- **SeparableConv1D, SeparableConv2D**: 1D and 2D Separable convolution.

- **DepthwiseConv1D, DepthwiseConv2D**: 1D and 2D Depthwise convolution.

*Note:*

None of the convolution layers support `data_format='channels_first'` or a `dilation_rate` other than 1. **Conv1D, Conv2D, Conv3D** do not support `groups` other than 1, and **Conv1D** does not support `padding='causal'`. **Conv1DTranspose, Conv2DTranspose, Conv3DTranspose** do not support `use_bias=False` or `output_padding`. **SeparableConv1D, SeparableConv2D** do not support `depth_multiplier` other than 1 or `use_bias=False`. **DepthwiseConv1D** does not support `depth_multiplier` other than 1.

*Example:*

```python
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Conv1D, Conv3D, Conv2DTranspose, DepthwiseConv2D
from tensorflow.keras.models import Sequential

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    SeparableConv2D(64, (3, 3), activation='relu'),
    Conv1D(32, 3, activation='relu', input_shape=(64, 64)),
    Conv3D(16, (3, 3, 3), activation='relu', input_shape=(16, 64, 64, 3)),
    Conv2DTranspose(32, (3, 3), activation='relu'),
    DepthwiseConv2D((3, 3), activation='relu')
])
```

## 3. **Normalization Layers**

Helps in stabilizing and speeding up the training process.

- **BatchNormalization**: Normalizes the activations of the previous layer at each batch.

- **LayerNormalization**: Normalizes across the features instead of the batch.

- **UnitNormalization**: Normalize the layer by the inputs 2 norm.

- **GroupNormalization**: Normalize the layer by the groups.

*Note:*

None of the normalization layers support an `axis` other than the default `-1`. **BatchNormalization**, **LayerNormalization**, and **GroupNormalization** do not support `center=False` or `scale=False`. **LayerNormalization** does not support `rms_scaling`.

*Example:*

```python
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, GroupNormalization, UnitNormalization, Dense, Activation
from tensorflow.keras.models import Sequential

model = Sequential([
    Dense(64, input_shape=(100,)),
    BatchNormalization(),
    Activation('relu'),
    LayerNormalization(),
    GroupNormalization(groups=8),
    UnitNormalization(),
    Dense(10, activation='softmax')
])
```

## 4. **Pooling Layers**

Used to reduce the spatial dimensions (width, height) of the input.

- **MaxPooling1D, MaxPooling2D, MaxPooling3D**: Max pooling operations.

- **AveragePooling1D, AveragePooling2D, AveragePooling3D**: Average pooling operations.

- **GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D**: Global max pooling operations.

- **GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D**: Global average pooling operations.

*Note:*

None of the pooling layers support `data_format='channels_first'`. **MaxPooling1D, MaxPooling2D, MaxPooling3D** and **AveragePooling1D, AveragePooling2D, AveragePooling3D** do not support `padding='same'`.

*Example:*

```python
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.models import Sequential

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    GlobalAveragePooling2D()
])
```

## 5. **Activation Functions**

Down below is all activation functions supported so far.

1. relu
1. sigmoid
1. tanh
1. leakyrelu
1. linear
1. elu
1. selu
1. swish
1. prelu
1. silu
1. gelu
1. softmax
1. mish
1. softplus
1. exponential
1. celu

*Note:*

`leakyrelu` and `prelu` are not supported as `activation='name'`, only through the **LeakyReLU** and **PReLU** layers. `elu` and `celu` do not support an `alpha` other than 1.0, for `elu` use the **ELU** layer instead. `gelu` does not support `approximate=True`. `softmax` does not support an axis other than the last. Any activation function not listed must be supplied with `--custom_activation` (Example in ***../tutorials/05_advanced_mlp/***).
