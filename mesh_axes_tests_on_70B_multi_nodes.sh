echo "SLURM_NODEID: $SLURM_NODEID"

export HEAD_NODE=tw045
export PROCESS_ID=$SLURM_NODEID

# baseline
# error: mesh axes ('data', 'expert', 'fsdp') (of total size 16), but 16 does not evenly divide 8
# MESH_DATA=2 MESH_FSDP=4 bash launch_70B_multi_nodes.sh

# error: NCCL operation ncclGroupEnd() failed: invalid usage (run with NCCL_DEBUG=WARN for details). Last NCCL warning(error) log entry (may be unrelated) 'MSCCL: number of channels available (2) less than required (4)'
MESH_DATA=2 MESH_FSDP=2 MESH_MODEL=4 bash launch_70B_multi_nodes.sh  

# # PP + one axis
# MESH_PIPELINE=-1 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=-1 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=-1 MESH_DATA=1 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=-1 MESH_DATA=1 MESH_FSDP=1 MESH_SEQ=1 MESH_MODEL=2 bash launch_70B_multi_nodes.sh

# # FSDP + one axis
# MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=-1 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=-1 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=1 MESH_FSDP=-1 MESH_SEQ=2 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=1 MESH_FSDP=-1 MESH_SEQ=1 MESH_MODEL=2 bash launch_70B_multi_nodes.sh

# # TP + one axis
# MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=1 MESH_SEQ=1 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=1 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=1 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=-1 bash launch_70B_multi_nodes.sh

# 3 axes
# MESH_PIPELINE=2 MESH_DATA=2 MESH_FSDP=4 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=2 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=4 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=2 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=1 MESH_MODEL=4 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=4 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=4 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=4 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=2 MESH_SEQ=4 MESH_MODEL=1 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=4 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=4 bash launch_70B_multi_nodes.sh
# MESH_PIPELINE=1 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=2 MESH_MODEL=4 bash launch_70B_multi_nodes.sh