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

echo "SLURM_NNODES: $SLURM_NNODES, SLURM_NODEID: $SLURM_NODEID, HEAD_NODE: $HEAD_NODE"

# FSDP only (baseline)
bash launch_70B_multi_nodes.sh

# DP/FSDP (baseline)
MESH_DATA=$SLURM_NNODES bash launch_70B_multi_nodes.sh

# 3 axes (SLURM_NNODES/2/-1) (<data>*<expert>*<fsdp> <= batch_size)
MESH_PIPELINE=1 MESH_DATA=$SLURM_NNODES MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=2 MESH_DATA=$SLURM_NNODES MESH_FSDP=-1 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=2 MESH_DATA=$SLURM_NNODES MESH_FSDP=1 MESH_SEQ=-1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=2 MESH_DATA=$SLURM_NNODES MESH_FSDP=1 MESH_SEQ=1 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=$SLURM_NNODES MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=-1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=$SLURM_NNODES MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=$SLURM_NNODES MESH_DATA=1 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=1 MESH_DATA=$SLURM_NNODES MESH_FSDP=2 MESH_SEQ=-1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=$SLURM_NNODES MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=2 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
