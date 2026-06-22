#### EVERYTHING IN THIS DIRECTORY IS ALREADY GENERATED EXCEPT BINARIES/EXECUTABLES:

1. **advanced_mlp.py** is the python script used to build an advanced mlp using the sequential model class. This models was not normalized. Run `python3 advanced_mlp.py` in the terminal to run the python script.

1. The python script will generate **advanced_mlp.h5** which is the saved neural net. 

1. Because the `--custom_activation` is on with the activation name attached, a custom activation script named **custom_activation.py** was also supplied for the code generation process. **ADDITIONALLY**, in the **.hpp** file, the C++ rendition of that activation function was added in the activation function section near the bottom.

1. **CodeJeNN** is then used to generate **advanced_mlp.hpp**. Run `./generate.sh` to code generate the trained model. Note: the **main.py** path is in relation to where this directory is directly located in the github repo.

1. Because the `--debug` flag was on, **DEBUG_** python and C++ files were generated as well and is used to test the inference and layer outputs for **advanced_mlp.hpp**. This means that every layer of both the Keras model and C++ model will output the first 10 values of that layer as well as the final output. Every layers output should match.

1. Run `clang++ DEBUG_simple_mlp.cpp` or `g++ DEBUG_simple_mlp.cpp` to create the binary for the test file. Then run `./a.out` in the terminal. Then run `python DEBUG_simple_mlp.py` to generate the output that Keras gives. Then compare both.

1. If you want to try it yourself, you may run `./clean.sh` and restart from step 1.