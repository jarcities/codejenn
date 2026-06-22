## 1. **Core, Reshape, Regularization Layers**

These are the fundamental layers used in almost every neural network.

- **Dense**: Fully connected layer.
- **Rescale**: Rescale layers by a new range between [0, 1] or [-1, 1]
- **Activation**: Applies an activation function.
- **Dropout**, **SpatialDropout1D**, **SpatialDropout2D**, **SpatialDropout3D**: Randomly sets input units to 0 during training to prevent overfitting.
- **Flatten**: Flattens the input without affecting the batch size.
- **Reshape**: Reshapes an output to a certain shape.

**Example:**
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
<!-- - **ConvLSTM2D**: Convolutional LSTM. -->

**Example:**
```python
from tensorflow.keras.layers import Conv2D, SeparableConv2D, ConvLSTM2D, Conv1D, Conv3D, Conv2DTranspose, DepthwiseConv2D
from tensorflow.keras.models import Sequential

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    SeparableConv2D(64, (3, 3), activation='relu'),
    ConvLSTM2D(64, (3, 3), activation='relu', return_sequences=True),
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

**Example:**
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
- **AvgPooling1D, AvgPooling2D, AvgPooling3D**: Average pooling operations.
- **GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D**: Global max pooling operations.
- **GlobalAvgPooling1D, GlobalAvgPooling2D, GlobalAvgPooling3D**: Global average pooling operations.

**Example:**
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