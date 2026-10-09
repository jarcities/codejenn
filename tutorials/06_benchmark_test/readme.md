## Benchmark Test

This benchmark tests the inference speed and accuracy of **CodeJeNN** against **Keras** using four models, two MLPs and two CNNs, `small_mlp/`, `big_mlp/`, `small_cnn/`, and `big_cnn/`. The small and big models show how inference scales as the model takes more memory.

* **Speed** ⮕ The total wall clock time to run inference on 10,000 data samples, one sample at a time. **CodeJeNN** is compared to **Keras** with both the **Tensorflow** and **Pytorch** backend, in eager mode and when heavily optimized. Each test is ran 30 times and the mean, standard deviation, and median are posted.

* **Accuracy** ⮕ The maximum absolute error, mean absolute error, and relative error between **CodeJeNN** and **Keras** for all values of each layer, using a single input.

For a fair comparison, everything is ran on a single thread using one CPU core, in 32 bit.

## How To Run

All four models have the exact same workflow and ran through a shell.

1. Assuming everything has been in followed in `codejenn/readme.md` and `codejenn/src/api-core/readme.md`, the next two steps are independent of each other and test two different aspects.

1. **Speed**: first run `./run.sh | tee output.txt`. For each backend, the bash script will:

    1. Run **train.py** to train the model. The user can tune the important following options: `NUM_SAMPLES`, `INPUT_DIM`, and `EPOCHS`.

    1. Run **CodeJeNN** to code generate the model into C++.

    1. Run **data.py** to create training data (at random) to be used for inference.

    1. Run **test.py** for **Keras** in eager mode, **test.py** with the `--jit` option for the optimized **Keras** model, and **test.cpp** for **CodeJeNN**. Each one is ran 30 times (`RUNS` in **run.sh**).

    1. Post the mean, standard deviation, and median of the times once the code finishes, along with the machine info.

1. **Accuracy**: first run `./run_.sh`. Either run with `tensorflow` or `torch` backend (`BACKEND` in **run_.sh**), it does not make much of a difference. The bash script will:

    1. Run **train.py** to train the model, then run **CodeJeNN** to code generate the model into C++ with the `--debug` flag on. The `--model_image` flag is also on, which requires installing `graphviz`.

    1. Run **DEBUG_model.cpp** and **DEBUG_model.py** once on the same single input. Every layer of both the C++ model and Keras model will output all values of that layer, saved to **DEBUG_cpp.txt** and **DEBUG_keras.txt**.

    1. Post the errors for each layer. Layers that do not compute anything during inference, such as dropout and flatten, are not printed by the C++ model and are marked as such.

1. To start from the beginning, run `./clean.sh`.

## Benchmark Reported in Cited Paper

The following table lists out the complete software and hardware configuration for both the benchmark test case and the CFD neural network implementation reported in the paper cited above for `tutorials/06_benchmark_test/` and `tutorials/07_cfd_implementation.zip`.

| Configuration | Details |
| :--- | :--- |
| **Hardware Configuration** | |
| Machine Info. | Apple MacBook Pro |
| Processor Info. | Apple M4 Pro (12 Cores, 8 Performance, 4 Efficiency) |
| Memory Info. | 24 GB Unified Memory |
| **Software Configuration** | |
| C++ Compiler | `GCC 15.2.0` (g++-15) |
| Compiler Flags | `-O3`, `-ffast-math`, `-fopenmp`, `-march=native` |
| Python Runtimes | `Python 3.12.12` (TensorFlow), `Python 3.14.2` (PyTorch)` |
| ML Frameworks | `Keras 3.13.2`, `TensorFlow 2.20.0`, `PyTorch 2.10.0` |
| **Execution & Optimization Configuration** | |
| Target Device | CPU (`float32` precision) |
| Thread Allocation | Single-threaded (`OMP_NUM_THREADS=1`, `TF=1`, `Torch=1`) |
| TensorFlow Optimizations | `Frozen graph` → `tf.function(jit_compile=True)` |
| PyTorch Optimizations | `torch.jit.trace` → `freeze` → `optimize_for_inference` |
| **Hardware Configuration for CFD Testcase** | |
| Machine Info. | Compute Cluster |
| Processor Info. | AMD EPYC 7702 64-Core Processor |
| Memory Info. | 270 GB Memory |

