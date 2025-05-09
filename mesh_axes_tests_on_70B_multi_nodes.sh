#!/bin/bash

if [[ -z "$SLURM_NNODES" ]]; then
    echo "SLURM_NNODES is unknown. Please set SLURM_NNODES."
fi

if [[ -z "$SLURM_NODEID" ]]; then
    echo "SLURM_NODEID is unknown. Please set SLURM_NODEID."
fi

if [[ -z "$HEAD_NODE" ]]; then
    echo "HEAD_NODE is unknown. Please set HEAD_NODE."
fi

echo "hostname: $(hostname) SLURM_NNODES: $SLURM_NNODES, SLURM_NODEID: $SLURM_NODEID, HEAD_NODE: $HEAD_NODE"

export LOG_OUTPUT_FOLDER=nodes-4
export BATCH_SIZE=64

# FSDP only (baseline)
MESH_DCN_FSDP=$SLURM_NNODES bash launch_70B_multi_nodes.sh

# DP/FSDP (baseline)
MESH_DCN_DATA=-1 bash launch_70B_multi_nodes.sh

# 3 axes (DCN: SLURM_NNODES, INI: [2,4]/-1) (<data>*<expert>*<fsdp> <= batch_size)
MESH_PIPELINE=2 MESH_DCN_DATA=-1 MESH_FSDP=-1 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=4 MESH_DCN_DATA=-1 MESH_FSDP=-1 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=1 MESH_DCN_DATA=-1 MESH_FSDP=2 MESH_SEQ=-1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=1 MESH_DCN_DATA=-1 MESH_FSDP=4 MESH_SEQ=-1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=1 MESH_DCN_DATA=-1 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=1 MESH_DCN_DATA=-1 MESH_FSDP=4 MESH_SEQ=1 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
MESH_DCN_PIPELINE=-1 MESH_DATA=2 MESH_FSDP=-1 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
MESH_DCN_PIPELINE=-1 MESH_DATA=4 MESH_FSDP=-1 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
