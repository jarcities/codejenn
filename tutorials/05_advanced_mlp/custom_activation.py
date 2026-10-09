import keras
from keras.utils import register_keras_serializable


@register_keras_serializable()
def scaled_sigmoid(x):
    return (keras.activations.sigmoid(x) * 5) - 1


@register_keras_serializable()
def bent_identity(x):
    return (keras.ops.sqrt(x * x + 1) - 1) / 2 + x


#C++ RENDITION OF EACH ACTIVATION FUNCTION ABOVE
C_FUNCTIONS = {
    "scaled_sigmoid": r"""
        output = (Scalar(1) / (Scalar(1) + std::exp(-input))) * Scalar(5) - Scalar(1);
    """,
    "bent_identity": r"""
        const Scalar root = std::sqrt(input * input + Scalar(1));
        output = (root - Scalar(1)) / Scalar(2) + input;
    """,
}
