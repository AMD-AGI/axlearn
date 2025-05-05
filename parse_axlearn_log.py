import os
import glob
import re

import pandas as pd
from absl import flags, app, logging


flags.DEFINE_list(
    "logs",
    None,
    "The paths or folders of log files generated from Axlearn. The log files must use `log` as file extension.",
    required=True,
)
flags.DEFINE_string(
    "output_folder",
    os.getcwd(),
    "The output folder of parsed metrics",
    required=False,
)
flags.DEFINE_bool(
    "overwrite_output_file",
    False,
    "Whether to overwrite previous output file if exists.",    
)
FLAGS = flags.FLAGS


METRICS = [
    "Max training state size \(partitioned\)",
    "Total HBM memory",
    "FLOPS",
    "The total memory traffic",
    "Average step time",    
]


def parse_metrics_from_text(text: str, metric_patterns: dict) -> dict:
    out = dict()        
    for metric_name, pattern in metric_patterns.items():
        match = pattern.search(text)        
        if match:            
            out[metric_name] = match.group(1)
            
    return out


def main(argv):
    del argv

    # metric patterns
    metric_patterns = {k: re.compile(rf"{k}:\s*(\d+\.\d+)") for k in METRICS}

    # get all log files
    file_paths = []
    for file_or_folder in FLAGS.logs:
        if os.path.isdir(file_or_folder):
            file_paths.extend([os.path.abspath(f) for f in glob.glob(os.path.join(file_or_folder, "**", "*.log"), recursive=True)])
        else:
            file_paths.append(os.path.abspath(file_or_folder))

    # parse metrics
    out = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            log_text = f.read()

        metrics = parse_metrics_from_text(log_text, metric_patterns)
        logging.info(f"Log File: {file_path} | Metrics: {metrics}")

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
    
        
                

