#### NOTHING IN THIS DIRECTORY IS GENERATED YET, FOLLOW THE STEPS BELOW TO GENERATE EVERYTHING:

1. **cnn_3d.py** is the python script used to build a 3d cnn using the functional model class. The models input and output was normalized using the standard deviation and mean. Run `./run.sh` or `bash run.sh` in the terminal to run the bash script. This tutorial uses the **Tensorflow** backend, so it must be installed.

1. The bash script will first run the python script **cnn_3d.py** which will generate **cnn_3d.keras** which is the saved neural net. It will also generate the input and output normalization parameters in **.npy** files that will be also used to code generate the model.

1. It will then run **CodeJeNN** to code generate **cnn_3d.hpp**. Note: the **main.py** path is in relation to where this directory is directly located in the github repo. This tutorial uses `--bit=32` because the **Tensorflow** `MaxPooling3D` layer does not support 64 bit.

1. Because the `--debug` flag was on, **DEBUG_** python and C++ files were generated as well and is used to test the inference and layer outputs for **cnn_3d.hpp**. This means that every layer of both the Keras model and C++ model will output all values of that layer as well as the final output. Every layers output should match up to machine precision based on the `--bit` defined in **run.sh**.

1. The bash script then runs `clang++ DEBUG_cnn_3d.cpp` to create the binary **a.out** for the test file. It runs `./a.out` and `python3 DEBUG_cnn_3d.py` and saves what each one prints to **DEBUG_cpp.txt** and **DEBUG_keras.txt**.

1. Finally, the bash script compares both files, where the maximum absolute error, mean absolute error, and relative error of all values are calculated for each layer and for the final output, and printed out. Layers that do not compute anything during inference, such as dropout, are not printed by the C++ model and are marked as such.

1. To start from the beginning, run `./clean.sh` and restart from step 1.
