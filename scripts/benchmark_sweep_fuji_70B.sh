#!/bin/bash

set -x

export TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")


# BS=16
PER_NODE_BATCH_SIZE=16 bash scripts/benchmark_70B_fuji_mi350.sh

# BS=32
PER_NODE_BATCH_SIZE=32 bash scripts/benchmark_70B_fuji_mi350.sh

# BS=48
PER_NODE_BATCH_SIZE=48 bash scripts/benchmark_70B_fuji_mi350.sh

# # BS=56 (OOM)
# PER_NODE_BATCH_SIZE=56 bash scripts/benchmark_70B_fuji_mi350.sh

# # BS=64 (OOM)
# PER_NODE_BATCH_SIZE=64 bash scripts/benchmark_70B_fuji_mi350.sh


# # Get HIPBLAST log for GEMM tuning (optional)
# export HIPBLASLT_LOG_MASK=64

# # BS=16
# export HIPBLASLT_LOG_FILE=axlearn_fuji_70b_fsdp8_bs16_mi355x.txt
# PER_NODE_BATCH_SIZE=16 bash scripts/benchmark_70B_fuji_mi350.sh

# # BS=32
# export HIPBLASLT_LOG_FILE=axlearn_fuji_70b_fsdp8_bs32_mi355x.txt
# PER_NODE_BATCH_SIZE=32 bash scripts/benchmark_70B_fuji_mi350.sh

# # BS=48
# export HIPBLASLT_LOG_FILE=axlearn_fuji_70b_fsdp8_bs48_mi355x.txt
# PER_NODE_BATCH_SIZE=48 bash scripts/benchmark_70B_fuji_mi350.sh