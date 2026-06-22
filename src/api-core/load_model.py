"""
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
from keras.models import load_model


def loadModel(file_path, base_file_name, custom_activation):
    # ==============================================================
    # load a keras model from .h5 or .keras using keras and tf with
    # combos of compile and custom objects options.
    # ==============================================================
    file_name, file_extension = os.path.splitext(file_path)
    if file_extension not in (".h5", ".keras"):
        raise ValueError(f"Unsupported file type: {file_extension}")

    #######################
    ## CUSTOM ACTIVATION ##
    #######################
    act_fun = {}
    custom_objects = {}
    if custom_activation is not None:
        try:
            import importlib

            try:
                module = importlib.import_module(custom_activation)
                act_fun = getattr(module, custom_activation)
            except Exception:
                import importlib.util

                module_dir = os.path.dirname(file_path)
                module_path = os.path.join(module_dir, f"{custom_activation}.py")
                if os.path.exists(module_path):
                    spec = importlib.util.spec_from_file_location(
                        custom_activation, module_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    act_fun = getattr(module, custom_activation)
                else:
                    raise
        except Exception as e:
            print(
                f"\n__Error__ Cannot import custom activation '{custom_activation}' -> {e}"
            )
    if act_fun:
        custom_objects[custom_activation] = act_fun

    errors = []

    # try 1
    try:
        model = load_model(file_path, custom_objects=custom_objects, compile=False)
        return model, file_extension
    except Exception as e:
        errors.append(f"__Error__ Keras compile=False, custom_objects -> {e}")

    # try 2
    try:
        model = load_model(file_path, compile=False)
        return model, file_extension
    except Exception as e:
        errors.append(f"__Error__ Keras compile=False, no custom_objects -> {e}")

    # try 3
    try:
        model = load_model(file_path, custom_objects=custom_objects)
        return model, file_extension
    except Exception as e:
        errors.append(f"__Error__ Keras default compile, custom_objects -> {e}")

    # try 4
    try:
        model = load_model(file_path)
        return model, file_extension
    except Exception as e:
        errors.append(f"__Error__ Keras default compile, no custom_objects -> {e}")

    # all is lost
    error_message = "\n".join(errors)
    raise RuntimeError(
        f"__Error__ All attempts to load the model failed for {file_name} -> \n{error_message}"
    )
