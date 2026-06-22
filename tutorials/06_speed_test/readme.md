## Profile Test

This tutorial examines the inference speed of a CNN and an MLP using **CodeJeNN** and **Keras**.
During inference, **Keras** utilizes optimized C++ tensor operations. To ensure a fair comparison, both the CNN and MLP are executed in eager mode rather than graph mode. In graph mode, the model is traced and converted into a computational graph for faster execution, instead of sequentially evaluating each operation.
When JIT is enabled with graph mode, the traced graph is further compiled and passed to XLA (Accelerated Linear Algebra) when using the TensorFlow backend, or to TorchDynamo when using the PyTorch backend.
Therefore, for fairness, both **CodeJeNN** and the **Keras** models are executed on a single thread using one CPU core.

Both CNN and MLP have the exact same workflow. **run.sh** and **clean.sh** are bash scripts to run and reset the code/data for testing pure inference speed.

1. First, **train.py** trains the model to be code generated, the shape or input and output size can be tuned as well as the number of samples.

    * The user can tune the important following options: `NUM_SAMPLES`, `INPUT_DIM`, and `EPOCHS`. There are other options but these are the most important.

1. Second, **data.py** creates synthetic data (completely random data) to be used for inference. The synthetic data should match the shape of the models input/output and the number of samples.

1. Third, **CodeJeNN** code generates the model into C++.

1. Finally, both **test.py** and **test.cpp** test the inference on 10,000 loops with each loop corresponding to a brand new data sample from **data.py**. The data is preloaded and switched each loop. The times will be posted once the code finishes.

    * **test.py** contains a `JIT` options which the user may use for a heavily optimized inferenced **Keras** model for comparison.

    * **test.py** also contains which backend you wish to use, **Pytorch** or **Tensorflow**. Both have different JIT logic.

## Compiling

* This test was ran on a macbook m1 pro with an m1 pro chip. 

* The code was compiled with gnc gcc compilers installed via homebrew NOT macOS native clang compiler. 

* The code was also ran in a conda environment that addionally installed `conda install -c conda-forge llvm-openmp`.

* The CNN test case uses pytorch backend and the MLP test case uses tensorflow backend.

## Notes

* In `cnn/` there are two options where the user may train a small CNN or a much bigger CNN taking more memory. This option allows users to see how big a CNN can get for inference showing that the significant scaling of learnable parameters—particularly in the `Conv2DTranspose` layer and the large subsequent `Dense` layer—shifts the bottleneck from computation to memory bandwidth. Since **CodeJeNN** produces inline C++ models with large static weight arrays, the CPU cache can be overwhelmed, resulting in slower inference due to the overhead of memory transfer from RAM.