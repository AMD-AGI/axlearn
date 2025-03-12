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
export XLA_FLAGS=" --xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=5 --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=137438953472 --xla_gpu_enable_all_gather_combine_by_dim=FALSE"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

export DISTRIBUTED_COORDINATOR=$HEAD_NODE:12345
export NUM_PROCESSES=2


export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# XLA_FLAGS=--xla_dump_to=/tmp/xla_dump; \
mkdir -p /tmp/gpt_c4_test; \
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash-single-host \
  --trainer_dir=/home/axlearn/2-node-70B-without-latency-hidding-scheduler  --trace_at_steps=0,3,6  --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu
