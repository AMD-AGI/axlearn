export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=0 --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.975
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
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

export LOG_OUTPUT_FOLDER="${LOG_OUTPUT_FOLDER:=$(pwd)}"
export STEP_TIMEOUT="${STEP_TIMEOUT:=360}"
export MAX_STEP="${MAX_STEP:=40}"
export STEP_TO_CAL_AVG_STEP_TIME="${STEP_TO_CAL_AVG_STEP_TIME:=20}"
export NUM_LAYERS="${NUM_LAYERS:=80}"
export BATCH_SIZE="${BATCH_SIZE:=16}"
export MESH_PIPELINE="${MESH_PIPELINE:=1}"
export MESH_DATA="${MESH_DATA:=1}"
export MESH_EXPERT="${MESH_EXPERT:=1}"
export MESH_FSDP="${MESH_FSDP:=-1}"
export MESH_SEQ="${MESH_SEQ:=1}"
export MESH_MODEL="${MESH_MODEL:=1}"

export FORCE_PALLAS="${FORCE_PALLAS:=0}"
export GPU_BLOCK_Q="${GPU_BLOCK_Q:=32}"
export GPU_BLOCK_K="${GPU_BLOCK_K:=32}"
export NUM_WARPS="${NUM_WARPS:=2}"
export NUM_STAGES="${NUM_STAGES:=1}"

if [ $FORCE_PALLAS -eq 1 ]; then
  LOG_PATH=${LOG_OUTPUT_FOLDER}/output_$(date +%s)_p${MESH_PIPELINE}_d${MESH_DATA}_e${MESH_EXPERT}_f${MESH_FSDP}_s${MESH_SEQ}_m${MESH_MODEL}_L${NUM_LAYERS}_pallas_bq${GPU_BLOCK_Q}_bk${GPU_BLOCK_K}.log
else
  LOG_PATH=${LOG_OUTPUT_FOLDER}/output_$(date +%s)_p${MESH_PIPELINE}_d${MESH_DATA}_e${MESH_EXPERT}_f${MESH_FSDP}_s${MESH_SEQ}_m${MESH_MODEL}_L${NUM_LAYERS}.log
fi

mkdir -p /tmp/gpt_c4_test
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash-single-host \
  --trainer_dir=/tmp/gpt_c4_test --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu \
  --trainer_watchdog_timeout_seconds $STEP_TIMEOUT \
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
  --force_pallas $FORCE_PALLAS \
  --gpu_block_q $GPU_BLOCK_Q \
  --gpu_block_k $GPU_BLOCK_K \
  --num_warps $NUM_WARPS \
  --num_stages $NUM_STAGES \
  &> ${LOG_PATH}
