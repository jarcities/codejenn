#### NOTHING IN THIS DIRECTORY IS GENERATED YET, FOLLOW THE STEPS BELOW TO GENERATE EVERYTHING:

1. **advanced_mlp.py** is the python script used to build an advanced mlp using the sequential model class. This model was not normalized and uses two custom activation functions, `scaled_sigmoid` and `bent_identity`, which are imported from **custom_activation.py**. Run `./run.sh` or `bash run.sh` in the terminal to run the bash script. This tutorial uses the **Tensorflow** backend, so it must be installed.

1. The bash script will first run the python script **advanced_mlp.py** which will generate **advanced_mlp.h5**, the saved neural net. It will then run **CodeJeNN** to code generate **advanced_mlp.hpp**.

1. Because the `--custom_activation` flag is on with the name of the python script attached (without the **.py**), the custom activation script named **custom_activation.py** was also supplied for the code generation process. This script holds every custom activation function the model uses, each one serialized with `@register_keras_serializable()`. **MOREOVER**, the C++ rendition of each activation function is at the bottom of the **custom_activation.py** file in a dictionary named `C_FUNCTIONS`, where each key is the name of the python activation function and each value is the C++ code that sets `output` from `input` (use `Scalar` as the data type). **CodeJeNN** reads this dictionary and writes each activation function into the generated **.hpp** file, so nothing has to be copied over by hand. If an activation function is missing from `C_FUNCTIONS`, **CodeJeNN** will print an error and will not generate the **.hpp** file.

1. Because the `--debug` flag was on, **DEBUG_*** python and C++ files were generated as well and are used to test the inference and layer outputs for **advanced_mlp.hpp**. This means that every layer of both the Keras model and C++ model will output all values of that layer, and the C++ model will also output the final output. Every layer's output should match up to machine precision based on the `--bit` defined in **run.sh**.

1. The bash script then runs `clang++ DEBUG_advanced_mlp.cpp` to create the binary **a.out** for the test file. It runs `./a.out` and `python3 DEBUG_advanced_mlp.py` and saves what each one prints to **DEBUG_cpp.txt** and **DEBUG_keras.txt**.

1. Finally, the bash script compares both files, where the maximum absolute error, mean absolute error, and relative error of all values are calculated for each layer and printed out. Layers that do not compute anything during inference, such as dropout, are not printed by the C++ model and are marked as such.

1. To start from the beginning, run `./clean.sh` and restart from step 1.
