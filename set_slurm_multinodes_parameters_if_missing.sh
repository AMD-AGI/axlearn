#!/bin/bash

if [[ -z "$SLURM_JOB_NODELIST" ]]; then
    echo "SLURM_JOB_NODELIST has to be provided."
    exit 1
fi
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))

if [[ -z "$SLURM_NNODES" ]]; then
    export SLURM_NNODES=${#nodes[@]}

    echo "Set SLURM_NNODES"
fi

if [[ -z "$HEAD_NODE" ]]; then
    export HEAD_NODE=${nodes[0]}

    echo "Set HEAD_NODE"
fi

if [[ -z "$SLURM_NODEID" ]]; then
    for i in "${!nodes[@]}"; do
        [[ "${nodes[$i]}" == "${hostname}" ]] && export SLURM_NODEID=$i
    done

    echo "Set SLURM_NODEID"
fi

echo "Hostname: $(hostname)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "HEAD_NODE: $HEAD_NODE"
