#!/bin/bash

# How to run?
# In Slurm interactive mode, under Axlearn repo's root: srun --ntasks=4 bash -c 'SLRUM_JOB_NAME=axlearn bash run_axlearn_multinodes.sh'

START_TIME=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StartTime=\K\S+')
START_TIME=$(echo "$START_TIME" | sed 's/T/-/; s/:/-/g')

WORKDIR="${WORKDIR:=$(pwd)}"
IMAGE_NAME="${IMAGE_NAME:=rocm/jax-training:maxtext-v25.5}"
# IMAGE_NAME="${IMAGE_NAME:=rocm/jax-private:rocm6.3.1-jax0.4.35-py3.10.15_cs}"

source set_slurm_multinodes_parameters_if_missing.sh
SLURM_JOB_NAME="${SLURM_JOB_NAME:=axlearn-eval}"
BATCH_SZIE_BASE="${BATCH_SZIE_BASE:=16}"
BATCH_SIZE="${BATCH_SIZE:=$((SLURM_NNODES*BATCH_SZIE_BASE))}"


docker rm -f $SLURM_JOB_NAME | true

docker run -dit \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    -v /home/amd-shared-home/.cache/huggingface:/hf_cache \
    -v $HOME/.ssh:/root/.ssh \
    -v $HOME:$HOME \
    -e HF_HOME=/hf_cache \
    --name $SLURM_JOB_NAME \
    $IMAGE_NAME > /dev/null

docker exec \
    -e SLURM_NNODES=$SLURM_NNODES \
    -e SLURM_NODEID=$SLURM_NODEID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e HEAD_NODE=$HEAD_NODE \
    -e TIMESTAMP=$START_TIME \
    -e BATCH_SIZE=$BATCH_SIZE \
    -e LOG_OUTPUT_FOLDER="nodes-${SLURM_NNODES}-bs-${BATCH_SIZE}-with-driver" \
    -w $WORKDIR \
    $SLURM_JOB_NAME \
    bash -c '
        bash install_broadcomm_rdma.sh
        pip install -e ".[core]"
        # pip install einops  # rocm/jax-private:rocm6.3.1-jax0.4.35-py3.10.15_cs
        bash mesh_axes_tests_on_70B_multi_nodes.sh
    '
