export NCCL_IB_TC=41
export NCCL_IB_SL=0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET # for NCCL debug, use COLL for collectives
# export NCCL_DEBUG_FILE=llama3-70b-64N-short-run.%h.%p.log
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9  #### should be detected automatically.
export NCCL_SOCKET_IFNAME=rdma0 #### should be detected automatically.
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TF_CPP_VMODULE=hlo_rematerialization=1,gpu_compiler=1,gpu_compiler=2,gpu_compiler=3,gpu_compiler=4,gpu_compiler=5
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
# export XLA_FLAGS="--xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=5 --xla_gpu_enable_latency_hiding_scheduler=true"
export XLA_FLAGS="--xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=5 --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=137438953472 --xla_gpu_enable_all_gather_combine_by_dim=FALSE"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.975
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_EXT_ASM=1
export NVTE_CK_ASM_ATOMIC_FP32=0 #
export NVTE_CK_ASM_NO_COEX=0
export NVTE_CK_ASM_RTZ_CVT=1
export NVTE_CK_BWD_V3=1 #
export NVTE_CK_V3_RTZ_CVT=2
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0  # try setting this to 1 if we get NAN loss in training
export NVTE_CK_IS_V3_SPEC=1 # try setting this to 0 if we get NAN loss in training
export NVTE_CK_HOW_V3_BF16_CVT=2 # may want to try 1 and 0 if we get NAN loss in training


export DISTRIBUTED_COORDINATOR=$HEAD_NODE:12345
export NUM_PROCESSES=2


# XLA_FLAGS=--xla_dump_to=/tmp/xla_dump; \
mkdir -p /tmp/gpt_c4_test; \
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash-single-host \
  --trainer_dir=/home/axlearn/2-node-70B-without-latency-hidding-scheduler  --trace_at_steps=0,3,6  --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu
