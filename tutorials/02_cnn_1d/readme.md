#### EVERYTHING IN THIS DIRECTORY IS ALREADY GENERATED EXCEPT BINARIES/EXECUTABLES:

1. **cnn_1d.py** is the python script used to build a 1d cnn using the functional model class. The models input and output was normalized using the max and min. Run `python3 cnn_1d.py` in the terminal to run the python script.

1. The python script will generate **cnn_1d.keras** which is the saved neural net. It will also generate the input and output normalization parameters in **.npy** files that will be also used to code generate the model.

1. **CodeJeNN** is then used to generate **cnn_1d.hpp**. Run `./generate.sh` to code generate the trained model. Note: the **main.py** path is in relation to where this directory is directly located in the github repo.

1. Because the `--debug` flag was on, **DEBUG_** python and C++ files were generated as well and is used to test the inference and layer outputs for **cnn_1d.hpp**. This means that every layer of both the Keras model and C++ model will output the first 10 values of that layer as well as the final output. Every layers output should match.

1. Run `clang++ DEBUG_cnn_1d.cpp` or `g++ DEBUG_cnn_1d.cpp` to create the binary for the test file. Then run `./a.out` in the terminal. Then run `python DEBUG_cnn_1d.py` to generate the output that Keras gives. Then compare both.

1. If you want to try it yourself, you may run `./clean.sh` and restart from step 1.