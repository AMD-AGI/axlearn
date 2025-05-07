# FSDP only (baseline)
bash launch_70B_single_node.sh

# DP only (baseline)
MESH_DATA=-1 MESH_FSDP=1 bash launch_70B_single_node.sh

# DP/FSDP (baseline)
MESH_DATA=2 MESH_FSDP=-1 bash launch_70B_single_node.sh

# 3 axes
MESH_PIPELINE=2 MESH_DATA=2 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=1 bash launch_70B_single_node.sh
MESH_PIPELINE=2 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=1 bash launch_70B_single_node.sh
MESH_PIPELINE=2 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=1 MESH_MODEL=2 bash launch_70B_single_node.sh
MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=2 MESH_MODEL=1 bash launch_70B_single_node.sh
MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=2 bash launch_70B_single_node.sh
MESH_PIPELINE=2 MESH_DATA=1 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=2 bash launch_70B_single_node.sh
MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=2 MESH_SEQ=2 MESH_MODEL=1 bash launch_70B_single_node.sh
MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=2 MESH_SEQ=1 MESH_MODEL=2 bash launch_70B_single_node.sh
MESH_PIPELINE=1 MESH_DATA=2 MESH_FSDP=1 MESH_SEQ=2 MESH_MODEL=2 bash launch_70B_single_node.sh
MESH_PIPELINE=1 MESH_DATA=1 MESH_FSDP=2 MESH_SEQ=2 MESH_MODEL=2 bash launch_70B_single_node.sh
