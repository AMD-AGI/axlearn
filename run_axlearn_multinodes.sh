#!/bin/bash

# How to run?
# In Slurm interactive mode, under Axlearn repo's root: srun bash -c 'SLRUM_JOB_NAME=axlearn bash run_axlearn_multinodes.sh'

WORKDIR="${WORKDIR:=$(pwd)}"
IMAGE_NAME="${IMAGE_NAME:=rocm/jax-training:maxtext-v25.5}"

source set_slurm_multinodes_parameters_if_missing.sh
SLURM_JOB_NAME="${SLURM_JOB_NAME:=axlearn-eval}"
BATCH_SIZE="${BATCH_SIZE:=$((SLURM_NNODES*8))}"


docker rm -f $SLURM_JOB_NAME | true

docker run -dit \
    --device /dev/dri --device /dev/kfd \
    --device=/dev/infiniband \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    --shm-size 128G \
    -e HF_HOME=/hf_cache \
    -v /home/amd-shared-home/.cache/huggingface:/hf_cache \
    -v $HOME/.ssh:/root/.ssh \
    -v $HOME:$HOME \
    --name $SLURM_JOB_NAME \
    $IMAGE_NAME > /dev/null

docker exec \
    -e SLURM_NNODES=$SLURM_NNODES \
    -e SLURM_NODEID=$SLURM_NODEID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e HEAD_NODE=$HEAD_NODE \
    -e BATCH_SIZE=$BATCH_SIZE \
    -e WORKDIR=$WORKDIR \
    $SLURM_JOB_NAME \
    bash -c '
        sudo apt install iproute2 -y
        sudo apt install -y linux-headers-"$(uname -r)" libelf-dev
        sudo apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev
        cd $WORKDIR
        pip install ".[core]"
        bash mesh_axes_tests_on_70B_multi_nodes.sh
    '
