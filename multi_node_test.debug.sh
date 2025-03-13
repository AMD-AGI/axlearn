export NCCL_IB_TC=41
export NCCL_IB_TIMEOUT=17
export NCCL_IB_SL=0
# export NCCL_SOCKET_IFNAME=ens51f0np0
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple

# NCCL debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL


export HSA_FORCE_FINE_GRAIN_PCIE=1
export TF_CPP_VMODULE=hlo_rematerialization=1,gpu_compiler=1,gpu_compiler=2,gpu_compiler=3,gpu_compiler=4,gpu_compiler=5
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=5"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
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


export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# XLA_FLAGS=--xla_dump_to=/tmp/xla_dump; \
mkdir -p /tmp/gpt_c4_test; \
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash-single-host \
  --trainer_dir=/home/axlearn/2-node-70B-without-latency-hidding-scheduler  --trace_at_steps=0,3,6  --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu
