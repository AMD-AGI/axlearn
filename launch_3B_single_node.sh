export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH # required for hipblaslt

# export XLA_FLAGS="--xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0 --xla_gpu_autotune_level=5 --xla_gpu_enable_latency_hiding_scheduler=true"
export XLA_FLAGS="--xla_gpu_enable_cublaslt=true --xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0 --xla_gpu_graph_level=0 --xla_gpu_enable_latency_hiding_scheduler=true"
# export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_gpu_dump_llvmir=true --xla_dump_to=/work/hlo_dumps_training_3b --xla_dump_hlo_pass_re=.* --xla_gpu_enable_cublaslt=true --xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.975
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
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
 

# XLA_FLAGS=--xla_dump_to=/tmp/xla_dump; \
mkdir -p /tmp/gpt_c4_test; \
# /work/rocprofiler-sdk-build/bin/rocprofv3 -s --output-format csv pftrace -d /work/profiling_data -- 
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=fuji-3B-v3-flash-single-host \
  --trainer_dir=/tmp/gpt_c4_test --data_dir=gs://axlearn-public/tensorflow_datasets --jax_backend=gpu --mesh_selector="gpu-remat-test" --trainer_dir="/work/single-node-3B/" --trace_at_steps=0,3,6 
