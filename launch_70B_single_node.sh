export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH # required for hipblaslt

# export XLA_FLAGS="--xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=5 --xla_gpu_enable_latency_hiding_scheduler=true"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.975
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=false"

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
 

# XLA_FLAGS=--xla_dump_to=/tmp/xla_dump; \
mkdir -p /tmp/gpt_c4_test; \
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash-single-host \
  --trainer_dir=/tmp/gpt_c4_test --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu \
  --mesh_selector="gpu-remat-test" \
  --trainer_dir="/home/mingyyan/single-node-70B/"  \
  --trace_at_steps=0,3,6 