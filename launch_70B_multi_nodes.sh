export NCCL_IB_TC=41
export NCCL_IB_SL=0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET  # for NCCL debug, use COLL for collectives
# export NCCL_DEBUG_FILE=llama3-70b.%h.%p.log
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=ens51np0
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
# export XLA_FLAGS="--xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=0 --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=False"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=0 --xla_gpu_enable_latency_hiding_scheduler=TRUE --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_all_gather_combine_by_dim=FALSE --xla_gpu_memory_limit_slop_factor=95"
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

export TF_COORDINATION_SERVICE_BARRIER_TIMEOUT_SECONDS=300
export NUM_PROCESSES="${NUM_PROCESSES:=$SLURM_NNODES}"
export PROCESS_ID="${PROCESS_ID:=$SLURM_NODEID}"
export HEAD_NODE=$HEAD_NODE
export DISTRIBUTED_COORDINATOR=$HEAD_NODE:12345

export LOG_OUTPUT_FOLDER="${LOG_OUTPUT_FOLDER:=$(pwd)}"
export STEP_TIMEOUT="${STEP_TIMEOUT:=3600}"
export MAX_STEP="${MAX_STEP:=40}"
export STEP_TO_CAL_AVG_STEP_TIME="${STEP_TO_CAL_AVG_STEP_TIME:=20}"
export NUM_LAYERS="${NUM_LAYERS:=80}"
export BATCH_SIZE="${BATCH_SIZE:=8}"
export MESH_PIPELINE="${MESH_PIPELINE:=1}"
export MESH_DATA="${MESH_DATA:=1}"
export MESH_EXPERT="${MESH_EXPERT:=1}"
export MESH_FSDP="${MESH_FSDP:=-1}"
export MESH_SEQ="${MESH_SEQ:=1}"
export MESH_MODEL="${MESH_MODEL:=1}"
export MESH_DCN_PIPELINE="${MESH_DCN_PIPELINE:=1}"
export MESH_DCN_DATA="${MESH_DCN_DATA:=1}"
export MESH_DCN_EXPERT="${MESH_DCN_EXPERT:=1}"
export MESH_DCN_FSDP="${MESH_DCN_FSDP:=1}"
export MESH_DCN_SEQ="${MESH_DCN_SEQ:=1}"
export MESH_DCN_MODEL="${MESH_DCN_MODEL:=1}"

mkdir -p /tmp/gpt_c4_test
mkdir -p $LOG_OUTPUT_FOLDER
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
  --mesh_dcn_pipeline $MESH_DCN_PIPELINE \
  --mesh_dcn_data $MESH_DCN_DATA \
  --mesh_dcn_expert $MESH_DCN_EXPERT \
  --mesh_dcn_fsdp $MESH_DCN_FSDP \
  --mesh_dcn_seq $MESH_DCN_SEQ \
  --mesh_dcn_model $MESH_DCN_MODEL \
  &> ${LOG_OUTPUT_FOLDER}/output_$(date +%s)_p${MESH_PIPELINE}_d${MESH_DATA}_e${MESH_EXPERT}_f${MESH_FSDP}_s${MESH_SEQ}_m${MESH_MODEL}.${PROCESS_ID}.log
