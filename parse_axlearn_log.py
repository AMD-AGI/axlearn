# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
A script to parse out metrics during training from logs
"""

import glob
import os
import re

import pandas as pd
from absl import app, flags, logging

flags.DEFINE_list(
    "logs",
    None,
    (
        "The paths or folders of log files generated from Axlearn. "
        "The log files must use `log` as file extension."
    ),
    short_name="l",
    required=True,
)
flags.DEFINE_string(
    "output_folder",
    os.getcwd(),
    "The output folder of parsed metrics",
    short_name="o",
    required=False,
)
flags.DEFINE_bool(
    "overwrite_output_file",
    False,
    "Whether to overwrite previous output file if exists.",
    short_name="w",
)
FLAGS = flags.FLAGS


METRICS = {
    "Max training state size (partitioned)": "Max training state size \(partitioned\)",
    "Total HBM memory": "Total HBM memory",
    "FLOPS": "FLOPS",
    "The total memory traffic": "The total memory traffic",
    "Average step time": "Average step time",
    "PP": "\nmesh_shape\[0\]",
    "DP": "\nmesh_shape\[1\]",
    "EP": "\nmesh_shape\[2\]",
    "FSDP": "\nmesh_shape\[3\]",
    "SP": "\nmesh_shape\[4\]",
    "TP": "\nmesh_shape\[5\]",
}
MESH_AXES = ["PP", "DP", "EP", "FSDP", "SP", "TP"]


def parse_metrics_from_text(text: str, metric_patterns: dict) -> dict:
    out = dict()
    for metric_name, pattern in metric_patterns.items():
        match = pattern.search(text)
        if match:
            out[metric_name] = match.group(1)

    return out


def parse_mesh_axes_from_log_file_name(file_name: str) -> "Optional[str]":
    """
    Best effort to parse out used mesh axes from log file name
    Format: ..._<axis_code><scale>_<axis_code><scale>_...
    Example: ..._p1_d2_e1_f-1_s2_m2... -> DP/FSDP/SP/TP
    """
    axis_code_to_parallism = {
        "p": "PP",
        "d": "DP",
        "e": "EP",
        "f": "FSDP",
        "s": "SP",
        "m": "TP",
    }
    axis_codes = list(axis_code_to_parallism.keys())

    file_name = os.path.basename(file_name).split(".", 1)[0]
    out = [
        axis_code_to_parallism[ele[0]]
        for ele in file_name.split("_")
        if (len(ele) == 2 or len(ele) == 3)
        and ele[0] in axis_codes
        and ele[1:].replace("-", "").isdecimal()
        and int(ele[1:]) != 1
    ]

    return "/".join(out) if out else None


def main(argv):
    del argv

    # metric patterns
    metric_patterns = {
        k: re.compile(f"{v}:\s*([-+]?\d*\.?\d+)", re.MULTILINE) for k, v in METRICS.items()
    }

    # get all log files
    file_paths = []
    for file_or_folder in FLAGS.logs:
        if os.path.isdir(file_or_folder):
            file_paths.extend(
                [
                    os.path.abspath(f)
                    for f in glob.glob(os.path.join(file_or_folder, "**", "*.log"), recursive=True)
                ]
            )
        else:
            file_paths.append(os.path.abspath(file_or_folder))

    # parse metrics
    out = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            log_text = f.read()

        metrics = parse_metrics_from_text(log_text, metric_patterns)
        axis_scales = [metrics.pop(axis) for axis in MESH_AXES if axis in metrics]
        logging.info(f"Log File: {file_path} | Metrics: {metrics}")

        if len(axis_scales) == len(MESH_AXES):
            metrics["Mesh Axes"] = "/".join(
                [axis for axis, scale in zip(MESH_AXES, axis_scales) if int(scale) != 1]
            )
        else:
            metrics["Mesh Axes"] = parse_mesh_axes_from_log_file_name(file_path)

        metrics["log_file_path"] = file_path

        out.append(metrics)

    # output metrics (append if output file exists)
    out = pd.DataFrame(out)
    output_path = os.path.join(FLAGS.output_folder, "axlearn_log_metrics.csv")
    if os.path.exists(output_path) and not FLAGS.overwrite_output_file:
        prev_out = pd.read_csv(output_path)
        out = pd.concat([prev_out, out])

    out.to_csv(output_path, index=False)


if __name__ == "__main__":

    app.run(main)
