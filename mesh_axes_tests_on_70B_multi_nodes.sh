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

# baseline
# error: mesh axes ('data', 'expert', 'fsdp') (of total size 16), but 16 does not evenly divide 8
# MESH_DATA=2 MESH_FSDP=4 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=4 bash launch_70B_multi_nodes.sh

# 3 axes (2/2/4)
MESH_PIPELINE=2 MESH_DATA=2 MESH_FSDP=4 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=2 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=4 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=2 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=1 MESH_MODEL=4 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=4 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=4 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=4 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=2 MESH_SEQ=4 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=4 bash launch_70B_multi_nodes.sh
MESH_PIPELINE=1 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=2 MESH_MODEL=4 bash launch_70B_multi_nodes.sh