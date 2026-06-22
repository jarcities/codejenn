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
import numpy as np


def normParam(model_dir, debug, dtype=None):
    input_scale = []
    output_scale = []
    input_shift = []
    output_shift = []
    which_norm = {}

    # Use the model directory directly to look for normalization files
    base_dir = model_dir

    # expected .npy file names
    npy_files = {
        "input_min": os.path.join(base_dir, "input_min.npy"),
        "input_max": os.path.join(base_dir, "input_max.npy"),
        "output_min": os.path.join(base_dir, "output_min.npy"),
        "output_max": os.path.join(base_dir, "output_max.npy"),
        "input_std": os.path.join(base_dir, "input_std.npy"),
        "input_mean": os.path.join(base_dir, "input_mean.npy"),
        "output_std": os.path.join(base_dir, "output_std.npy"),
        "output_mean": os.path.join(base_dir, "output_mean.npy"),
    }

    # check if files exist (prioritize std/mean over min/max)
    input_std_mean = os.path.exists(npy_files["input_std"]) and os.path.exists(
        npy_files["input_mean"]
    )
    output_std_mean = os.path.exists(npy_files["output_std"]) and os.path.exists(
        npy_files["output_mean"]
    )
    input_min_max = os.path.exists(npy_files["input_min"]) and os.path.exists(
        npy_files["input_max"]
    )
    output_min_max = os.path.exists(npy_files["output_min"]) and os.path.exists(
        npy_files["output_max"]
    )

    if input_std_mean or output_std_mean:
        # use std/mean normalization
        if input_std_mean:
            input_std = np.load(npy_files["input_std"])
            input_mean = np.load(npy_files["input_mean"])
            if dtype is not None:
                input_std = input_std.astype(dtype)
                input_mean = input_mean.astype(dtype)
            input_scale = input_std
            input_shift = input_mean
            if debug:
                which_norm["input"] = "std/mean"
        if output_std_mean:
            output_std = np.load(npy_files["output_std"])
            output_mean = np.load(npy_files["output_mean"])
            if dtype is not None:
                output_std = output_std.astype(dtype)
                output_mean = output_mean.astype(dtype)
            output_scale = output_std
            output_shift = output_mean
            if debug:
                which_norm["output"] = "std/mean"

    else:
        # use min/max normalization
        if input_min_max:
            input_min = np.load(npy_files["input_min"])
            input_max = np.load(npy_files["input_max"])
            if dtype is not None:
                input_min = input_min.astype(dtype)
                input_max = input_max.astype(dtype)
            input_scale = input_max - input_min
            input_shift = input_min
            if debug:
                which_norm["input"] = "max/min"

        if output_min_max:
            output_min = np.load(npy_files["output_min"])
            output_max = np.load(npy_files["output_max"])
            if dtype is not None:
                output_min = output_min.astype(dtype)
                output_max = output_max.astype(dtype)
            output_scale = output_max - output_min
            output_shift = output_min
            if debug:
                which_norm["output"] = "max/min"

    # convert empty lists to None
    if isinstance(input_scale, list) and len(input_scale) == 0:
        input_scale = None
    if isinstance(output_scale, list) and len(output_scale) == 0:
        output_scale = None
    if isinstance(input_shift, list) and len(input_shift) == 0:
        input_shift = None
    if isinstance(output_shift, list) and len(output_shift) == 0:
        output_shift = None

    return input_scale, input_shift, output_scale, output_shift, which_norm
