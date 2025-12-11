#!/bin/bash
set -x


#################################
# Jax and TransformerEngine Setup
#################################
export GPU_MAX_HW_QUEUES=2
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
export MAX_STEPS="${MAX_STEPS:=80}"
export STEP_TO_CAL_AVG_STEP_TIME="${STEP_TO_CAL_AVG_STEP_TIME:=20}"
export NUM_LAYERS="${NUM_LAYERS:=80}"
export PER_NODE_BATCH_SIZE="${PER_NODE_BATCH_SIZE:=32}"

# Multi-node settings
export NUM_PROCESSES="${NUM_PROCESSES:=1}"
export PROCESS_ID="${PROCESS_ID:=0}"
export HEAD_NODE="${HEAD_NODE:=$HOSTNAME}"
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


# XLA Flags
export XLA_FLAGS="${XLA_FLAGS:=--xla_gpu_graph_level=0 --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false}"
#export XLA_FLAGS=$XLA_FLAGS" --xla_dump_to="$EXP_DIR"/xla_dump" # for Profiling logs


export EXP_DIR=$(pwd)"/amd_experiments/logs/"$EXP_NAME
mkdir -p $EXP_DIR
echo "Logging to: ${EXP_DIR}"


CONFIG=fuji-70B-v2-flash

python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=$CONFIG \
  --trainer_dir="${EXP_DIR}" --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu \
  --max_step $MAX_STEPS \
  --trainer_log_every_n_steps $STEP_TO_CAL_AVG_STEP_TIME \
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
  --log_dir "${EXP_DIR}" \
  --trainer_dir "${EXP_DIR}" \
  2>&1 | tee ${EXP_DIR}/output${PROCESS_ID}.log
#   --trace_at_steps="20" \