#### EVERYTHING IN THIS DIRECTORY IS ALREADY GENERATED EXCEPT BINARIES/EXECUTABLES:

1. **simple_mlp.py** is the python script used to build a simple mlp using the sequential model class. The models input and output was normalized using the standard deviation and mean. Run `python3 simple_mlp.py` in the terminal to run the python script.

1. The python script will generate **simple_mlp.h5** which is the saved neural net. It will also generate the input and output normalization parameters in **.npy** files that will be also used to code generate the model.

1. **CodeJeNN** is then used to generate **simple_mlp.hpp**. Run `./generate.sh` to code generate the trained model. Note: the **main.py** path is in relation to where this directory is directly located in the github repo.

1. Because the `--debug` flag was on, **DEBUG_** python and C++ files were generated as well and is used to test the inference and layer outputs for **simple_mlp.hpp**. This means that every layer of both the Keras model and C++ model will output the first 10 values of that layer as well as the final output. Every layers output should match.

1. Run `clang++ DEBUG_simple_mlp.cpp` or `g++ DEBUG_simple_mlp.cpp` to create the binary for the test file. Then run `./a.out` in the terminal. Then run `python DEBUG_simple_mlp.py` to generate the output that Keras gives. Then compare both.

1. If you want to try it yourself, you may run `./clean.sh` and restart from step 1.