#!/bin/bash
#################################
# NCCL Setup
#################################
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export NCCL_IB_TC=41
export NCCL_IB_SL=0
export NCCL_CHECKS_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_CROSS_NIC=0

# set NCCL_IB_HCA based on the DISCOVERED_NIC
if [ "${DISCOVERED_NIC:-}" = 'mlx' ]; then
  export NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7
elif [ "${DISCOVERED_NIC:-}" = 'ionic' ]; then
  export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
  # check available NCCL_SOCKET_IFNAME: ifconfig / ip addr (currently set for Pollara/ionic)
  export NCCL_SOCKET_IFNAME=enp49s0f0np0
elif [ "${DISCOVERED_NIC:-}" = 'bcm' ]; then
#  export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
#  export NCCL_SOCKET_IFNAME=ib0
  export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
else
  echo "DISCOVERED_NIC is not set or unknown. Please set DISCOVERED_NIC to 'mlx', 'ionic', or 'bcm'."
  exit 1
fi

export NCCL_IB_GID_INDEX=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=2


#################################
# Jax and TransformerEngine Setup
#################################
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.975
export HIP_FORCE_DEV_KERNARG=1
export TF_CPP_MIN_LOG_LEVEL="2"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_EXT_ASM=1
export NVTE_CK_ASM_ATOMIC_FP32=0
export NVTE_CK_ASM_NO_COEX=0
export NVTE_CK_ASM_RTZ_CVT=1
export NVTE_CK_BWD_V3=1
export NVTE_CK_V3_RTZ_CVT=2
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_CK_IS_V3_SPEC=1
export NVTE_CK_HOW_V3_BF16_CVT=2



#################################
# Experiment Settings
#################################
export EXP_NAME="${EXP_NAME:=amd_70B_fuji_bs32_fsdp8}"
export MAX_STEP="${MAX_STEP:=80}"
export STEP_TO_CAL_AVG_STEP_TIME="${STEP_TO_CAL_AVG_STEP_TIME:=5}"
export NUM_LAYERS="${NUM_LAYERS:=80}"
export PER_NODE_BATCH_SIZE="${PER_NODE_BATCH_SIZE:=32}"

# Multi-node settings
export NUM_PROCESSES="${NUM_PROCESSES:=1}"
export PROCESS_ID="${PROCESS_ID:=0}"
export HEAD_NODE="${HOSTNAME}"
export DISTRIBUTED_COORDINATOR=$HEAD_NODE:1234
export BATCH_SIZE=$((NUM_PROCESSES * PER_NODE_BATCH_SIZE))

# Mesh settings
export MESH_PIPELINE="${MESH_PIPELINE:=1}"
export MESH_DATA="${MESH_DATA:=1}"
export MESH_EXPERT="${MESH_EXPERT:=1}"
export MESH_FSDP="${MESH_FSDP:=8}" # FSDP intra-node
export MESH_SEQ="${MESH_SEQ:=1}"
export MESH_MODEL="${MESH_MODEL:=1}"
export MESH_DCN_PIPELINE="${MESH_DCN_PIPELINE:=1}"
export MESH_DCN_DATA="${MESH_DCN_DATA:=-1}" # DDP inter-node
export MESH_DCN_EXPERT="${MESH_DCN_EXPERT:=1}"
export MESH_DCN_FSDP="${MESH_DCN_FSDP:=1}"
export MESH_DCN_SEQ="${MESH_DCN_SEQ:=1}"
export MESH_DCN_MODEL="${MESH_DCN_MODEL:=1}"
echo "MESH_PIPELINE=$MESH_PIPELINE MESH_DATA=$MESH_DATA MESH_EXPERT=$MESH_EXPERT MESH_FSDP=$MESH_FSDP MESH_SEQ=$MESH_SEQ MESH_MODEL=$MESH_MODEL"

# Pallas Settings
export FORCE_PALLAS="${FORCE_PALLAS:=0}" # default is no Pallas, default pallas configuration is best performing in most cases
export GPU_BLOCK_Q="${GPU_BLOCK_Q:=64}"
export GPU_BLOCK_K="${GPU_BLOCK_K:=32}"
export NUM_WARPS="${NUM_WARPS:=2}"
export NUM_STAGES="${NUM_STAGES:=1}"

# XLA Flags
export XLA_FLAGS="${XLA_FLAGS:=--xla_gpu_graph_level=0 --xla_gpu_autotune_level=0 --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false}"
export XLA_FLAGS=$XLA_FLAGS" --xla_dump_to="$EXP_DIR"/xla_dump" # for Profiling logs


export EXP_DIR=$(pwd)"/amd_experiments/logs/"$EXP_NAME
mkdir -p $EXP_DIR
echo "Logging to: ${EXP_DIR}"


#################################
# Tuned GEMMs
#################################
# using Tuned GEMMs from hipblaslt by default, will exit if TUNED_GEMMS=1 and no tuned GEMMs found
export TUNED_GEMMS="${TUNED_GEMMS:=1}"

# automatically retrieves the tuned GEMMs file based on the mesh configuration
build_mesh_string() {
   local result="bs${PER_NODE_BATCH_SIZE}"
   [[ $MESH_MODEL -gt 1 ]] && result+="_tp${MESH_MODEL}"
   [[ $MESH_SEQ -gt 1 ]] && result+="_sp${MESH_SEQ}"
   [[ $MESH_EXPERT -gt 1 ]] && result+="_ep${MESH_EXPERT}"
   [[ $MESH_FSDP -gt 1 ]] && result+="_fsdp${MESH_FSDP}"
   echo "$result"
}
tuned_gemms_dir='./amd_experiments/tuned_gemms/'
model_settings="mi325x_fuji_70b" # for moe this would be "mi325x_moe"
tuned_gemms_path="${tuned_gemms_dir}gemm_${model_settings}_$(build_mesh_string).txt"

if [[ $TUNED_GEMMS -eq 1 ]]; then
    if [[ ! -f "$tuned_gemms_path" ]]; then
        echo "No tuned GEMMs for $(build_mesh_string)! you can still run training by setting TUNED_GEMMS=0"
        exit 0
    else
        export HIPBLASLT_TUNING_OVERRIDE_FILE="$tuned_gemms_path"
    fi
else
    echo "using default xla picked GEMMs"
fi




python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash-single-host \
  --trainer_dir="${EXP_DIR}" --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu \
  --max_step $MAX_STEP \
  --step_to_cal_avg_step_time $STEP_TO_CAL_AVG_STEP_TIME \
  --num_layers $NUM_LAYERS \
  --batch_size $BATCH_SIZE \
  --mesh_selector="amd-mi300-single-node" \
  --mesh_pipeline $MESH_PIPELINE \
  --mesh_data $MESH_DATA \
  --mesh_expert $MESH_EXPERT \
  --mesh_fsdp $MESH_FSDP \
  --mesh_seq $MESH_SEQ \
  --mesh_model $MESH_MODEL \
  --mesh_dcn_pipeline $MESH_DCN_PIPELINE \
  --mesh_dcn_data $MESH_DCN_DATA \
  --mesh_dcn_expert $MESH_DCN_EXPERT \
  --mesh_dcn_fsdp $MESH_DCN_FSDP \
  --mesh_dcn_seq $MESH_DCN_SEQ \
  --mesh_dcn_model $MESH_DCN_MODEL \
  --force_pallas $FORCE_PALLAS \
  --gpu_block_q $GPU_BLOCK_Q \
  --gpu_block_k $GPU_BLOCK_K \
  --num_warps $NUM_WARPS \
  --num_stages $NUM_STAGES \
  --log_dir "${EXP_DIR}" \
  --trainer_dir "${EXP_DIR}" \
  --trace_at_steps="20" \
  2>&1 | tee ${EXP_DIR}/output${PROCESS_ID}.log




