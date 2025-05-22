#!/bin/bash

# How to run? (under Axlearn repo's root)
# In Slurm interactive mode: srun --ntasks=4 bash -c 'SLRUM_JOB_NAME=axlearn bash run_axlearn_multinodes.sh'
# For sbatch: sbatch sbatch_job.sh "<flags-if-any> bash run_axlearn_multinodes.sh"

START_TIME=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StartTime=\K\S+')
START_TIME=$(echo "$START_TIME" | sed 's/T/-/; s/:/-/g')

WORKDIR="${WORKDIR:=$(pwd)}"
EXAMPLE_DIR="${WORKDIR}/example/rocm"
IMAGE_NAME="${IMAGE_NAME:=rocm/jax-training:maxtext-v25.5}"
# IMAGE_NAME="${IMAGE_NAME:=rocm/jax-private:rocm6.3.1-jax0.4.35-py3.10.15_cs}"

source ${EXAMPLE_DIR}/set_slurm_multinodes_parameters_if_missing.sh
SLURM_JOB_NAME="${SLURM_JOB_NAME:=axlearn-eval}"
BATCH_SZIE_BASE="${BATCH_SZIE_BASE:=16}"
BATCH_SIZE="${BATCH_SIZE:=$((SLURM_NNODES*BATCH_SZIE_BASE))}"
XLA_FLAGS="${XLA_FLAGS:=--xla_gpu_graph_level=0 --xla_gpu_autotune_level=0 --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false}"
LOG_OUTPUT_FOLDER_SUFFIX=""

docker rm -f $SLURM_JOB_NAME || true

docker run -dit \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --cap-add=IPC_LOCK \
    --volume /dev/infiniband:/dev/infiniband \
    --tmpfs /dev/shm:size=200G \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    -v $HOME/.ssh:/root/.ssh \
    -v $HOME:$HOME \
    -e HF_HOME=/hf_cache \
    --name $SLURM_JOB_NAME \
    $IMAGE_NAME > /dev/null

docker exec \
    -w $WORKDIR \
    $SLURM_JOB_NAME \
    bash -c '
        pip install -q -e ".[core]"
    '

docker exec \
    -e SLURM_NNODES=$SLURM_NNODES \
    -e SLURM_NODEID=$SLURM_NODEID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e HEAD_NODE=$HEAD_NODE \
    -e TIMESTAMP=$START_TIME \
    -e BATCH_SIZE=$BATCH_SIZE \
    -e XLA_FLAGS="$XLA_FLAGS" \
    -e LOG_OUTPUT_FOLDER="nodes-${SLURM_NNODES}-bs-${BATCH_SIZE}-${LOG_OUTPUT_FOLDER_SUFFIX}" \
    -w $EXAMPLE_DIR \
    $SLURM_JOB_NAME \
    bash -c '
        lspci | grep 'Broadcom.*NetXtreme-E' && echo "Found Broadcom card. Installing Broadcom InfiniBand driver" && bash install_broadcom_ib.sh
        lspci | grep 'Mellanox.*ConnectX' && echo "Found Mellanox card. Installing Mellanox InfiniBand driver" && bash install_mellanox_ib.sh
        # pip install -q -e ".[core]"
        bash mesh_axes_tests_on_70B_multi_nodes.sh
    '
