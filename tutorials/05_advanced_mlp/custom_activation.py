import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras.utils import register_keras_serializable


@register_keras_serializable()
def custom_activation(x):
    return (keras.activations.sigmoid(x) * 5) - 1


# FOR COPY AND PASTING IN C++ CODE
"""
    auto custom_activation = +[](Scalar& output, Scalar input, Scalar index /*can use "alpha" for index*/) noexcept
    {
        output = (1.0 / (1.0 + std::exp(-input))) * 5.0 - 1.0;
    };
"""
