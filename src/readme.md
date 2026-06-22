<!-- 
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
-->

## DIRECTORY CONTENTS
  * `api-core/` ⮕ Directory that holds all scripts to extract, load, and code generate any model in a directory where your neural nets are stored.

  * `bin/` ⮕ Default directory where genereated header files are created (the user can link another directory).

  * `dump_model/` ⮕ Default dump directory that the user can dump any trained model and link during the code generation process (the user can link another directory).

  * `generate.sh` & `clean.sh` ⮕ Default bash scripts used to generate and/or clean c++ files.

## CODE GENERATION STEPS
1. First create a trained neural nets using Keras and save in the supported file extensions `.h5` or `.Keras` (there is not advantage of using one over the other).

    * If necessary, save normalization/standardization parameters in a `.npy` file. (Example in ***dump_model/***)

    * **CodeJeNN** will find `.h5` or `.Keras` files in the directoy linked and skip all the other files so linking the directoy would suffice.

1. Put all models wanting to be code generated in the ***dump_model/*** directory, or whatever directory the user wish to link.

1. Link necessary flags or change the bash script `generate.sh` as stated below:

    1. `--input` ⮕ (REQUIRED) Path to directory that holds any and all trained models to be code generated, or use **dump_model/**.

    1. `--output` ⮕ (REQUIRED) Path to directory to save all generated header files, or use **bin/**.

    1. `--backend` ⮕ (OPTIONAL) Which DL framework to use for Keras api. **CodeJeNN** supports **PyTorch**, **Tensorflow**, or **Jax**, and will default to **Tensorflow**. For **PyTorch**, put `torch`, for **Tensorflow** put `tensorflow` and for **Jax** put `jax`.

    1. `--bit` ⮕ (OPTIONAL) Specifies the numerical precision (data type bit-width), either 32-bit or 64-bit. Defaults to 32-bit (float) if not specified.

    1. `--custom_activation` ⮕  (OPTIONAL)  If your model has a serialized custom activation function, attach the name of the activation function. (Example in ***tutorials/05_advanced_mlp/*** and more information below) 

    1. `--debug` ⮕ (OPTIONAL) If there are inference errors with the generated neural net, the user can compare what each layer outputs and see what was extracted between CodeJeNN and Keras. (Example in ***tutorials/***)

    1. `--model_image` ⮕ (OPTIONAL) Requires installing `graphviz` and `svn` allowing visualization image of neural net.

        ```bash
        python3 \
            ./api-core/main.py \
            --input="./dump_model" \
            --output="./bin" \
            --backend="tensorflow" \
            --bit=32 \
            --custom_activation="act_fun" \
            --debug \
            --model_image 
        ```

1. Run **generate.sh** (type `./generate.sh` in terminal/shell).

1. Check for any errors once generation is complete.

## About CodeJeNN (i)

* CodeJeNN code generations a train neural net stored on a **.Keras**,a **.h5** file. Keras was chosen because of its portability. Because the NN parameters, hyperparameters and the architecture itself is stored on the file, **CodeJeNN** only needs that file to code generate.

* You link the input directory with all trained models (infinite amount of **.Keras** or **.h5** files up to the user) wanting to be code generated as well as the output directory to save the files. Additionally, the data type bit is optional, but only accepts 32 (float) or 64 (double).

    * ***dump_model/*** is the default linked dump directory to place trained models. **bin/** is the default output directory for generated models.

* The generated header file and "predict" function will be named after the file name of the trained model that it was code generated from.

* The neural net layers that CodeJeNN supports to build a given model can be found in ***../tutorials/supported_layers.md***. 

* CodeJeNN also supports **custom activation functions** as said before and a full example is in ***../tutorials/05_advanced_mlp/***. Three things must happen. 

    1. In the training script, the user uses a custom activation function.

    1. Then in a seperate python file, the custom activation function must be supplied and serialized with the name of the file matching the name of the activation function.

    1. Finally, after code generation, the user must supply the C++ rendition of the activation function within the **.hpp** file. Print statements are included if the user forgets to implement.

* CodeJeNN is not perfect. Thus, extensive error handling as well as a `--debug` flag are used to help the user zero in on the code generation and inference errors. From testing, most of the time, errors are due to the user activation of CodeJeNN.

    * If such errors occurs, the user is very, and I mean very encouraged to add and update the logic and submit a pull request. 

## CodeJeNN Model Architecture (ii)

* CodeJeNN suports CNNs and MLPs. CodeJeNN only supports single input and output sequential models or functional models, more information can be found in the **../tutorials/** directory. Single input and single output data features refers to any scalar, any array, or any 2d or 3d matrix. So long as the data features are packed together. To truly understand how complex and deep these architectures can be code generated, the user is referred to the **../tutorials/** directory.

    * Sequential models or functional models refers to how the user can build an MLP or CNN using Keras. You can stack different types of layers on top whether they are Dense, Conv2D, Activations, Normalization layers and etc. 

* Unfortunately, at the time of writing, Keras is currently implementing API for GNNs.

* Once the model is code generated, the header file structure is as follows (percentage denotes space taken in the header file):
    1. import std libraries **>1%**
    1. propagation step functions (Dense(), UnitNormalization(), Conv2DTranspose, etc...) **~10%**
    1. function header template **>1%**
    1. arrays of layer parameters (weights, baises, strides, kernels, gamma, etc...) **~85%**
    1. functions calls to drive through the each layer of the model. **~5%**

## Algorithm Complexity (iii)

* **CodeJeNN** generates C++-interpreted neural networks for computational physics applications, especially for runtime performance.

* **Vectorization and parallelization** within the generated C++ code are avoided because outside of that, the users codebase is assumed to be already vectorized and parallelized thus no threads are left and implementation of such actions can result in slower inference and slower code overall.

* However, CodeJeNN tries to inline as much as possible by using templating, lambda functions, and less cache space being used. **Inlining and efficiency** are prioritized using techniques such as:

    * `constexpr` static arrays and passing variables by `const` reference or pointer to minimize memory usage.

    * Layer propagation functions (`Conv1D`, `Conv2D`, `Dense`, `LayerNormalization`) are heavily inlined, with memory pre-allocated where possible.

    * Activation functions are passed as lambda expressions and treated as template parameters to maximize inlining, separate from layer propagation logic.

* Loop optimization via variadic templates, which increases compile time and cache usage but significantly improves runtime has been considered. Although it is useful, for how large these weight/bias arrays can get, it would be too much cache space being used. 

* The user is also encouraged to optimize the code generated neural net as much as possible too. 

* Because everything is inlined, and all data structures are serialized, then memory can be an issue, it is up to the user to reduce the model in any way possible.

## Normalization/Standardization (iv)

CodeJeNN supports normalization/standardization of inputs and outputs.

* The normalization/standardization values must be stored in a **.npy** file.

* The normalization/standardization file must be named the following depending on how the user wish to normalize:

    * `input_max.npy` `input_min.npy` `output_max.npy` `output_min.npy`
    
    * `input_std.npy` `intput_mean.npy` `output_std.npy` `output_mean.npy`

* An example of input and output normalization/standardization is in `../tutorials/04_cnn_3d/`. 

## Testing Generated Model (v)

* As said before, if `--debug` flag is on, a **.cpp** and **.py** test file prepended with `DEBUG_` will be generated into the output directory so the user can compare the C++ NN output with the Python Keras output. 

* The **.hpp** file itself will generate print statements printing the first 10 values of each layer that is then used in the **.cpp** file.

* Similarly, the **.py** script will also use Keras' `predict` function within the `extractor` class to print out the first 10 values of each layer.

* Running both the **.cpp** and **.py**, the user may compare layer outputs to see which layer created discrepencies with the final output.

* The input will be based on expected input shape extracted during code generation and the input values are in ascending order from 0 up to the input shape.

* With the `--debug` flag on, during code generation, the model summary from Keras API will be printed. Then what **CodeJeNN** extracted will also be printed. This allows the user to compare and see any incorrect information being extracted.

* Note: prior testing has shown that the layer where the discrepency occurs may not be the issue but from the layer before it even if the prior layer outputs match. This can be from mismatches in layer shape sizes being passed to incorrect padding. Verify that the output is correct!

## Example & Tutorials (vi)
A simple example in **dump_model/** has been laid out. Just open a terminal/shell in the **src/** directory, MAKE SURE YOU HAVE INSTALLED ALL DEPENDICIES FROM **requirements.txt**, link the correct directorys in **generate.sh**, enter in `bash generate.sh` or `./generate.sh` in the terminal/shell, and the code generation process will execute.

There are of course more examples in ***../tutorials/***.