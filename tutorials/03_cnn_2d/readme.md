#### EVERYTHING IN THIS DIRECTORY IS ALREADY GENERATED EXCEPT BINARIES/EXECUTABLES:

1. **cnn_2d.py** is the python script used to build a 2d cnn using the functional model class. The models input and output was normalized using the max and min. Run `python3 cnn_2d.py` in the terminal to run the python script.

1. The python script will generate **cnn_2d.keras** which is the saved neural net. It will also generate the input and output normalization parameters in **.npy** files that will be also used to code generate the model.

1. **CodeJeNN** is then used to generate **cnn_2d.hpp**. Run `./generate.sh` to code generate the trained model. Note: the **main.py** path is in relation to where this directory is directly located in the github repo.

1. In this tutorial, the `--debug` flag is not used.

1. If you want to try it yourself, you may run `./clean.sh` and restart from step 1.