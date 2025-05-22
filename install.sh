#!/bin/bash
export JAX_BRANCH=rocm-jaxlib-v0.5.0-waves_per_eu-2
export XLA_BRANCH=rocm-jaxlib-v0.5.0-waves_per_eu
#export XLA_BRANCH=rocm-jaxlib-v0.5.0_aiss_ws64
export ROCM_VERSION=6.3.1
export PATH=/opt/rocm-$ROCM_VERSION/bin:/opt/rocm-$ROCM_VERSION/lib/llvm/bin:$PATH
export ROCM_PATH=/opt/rocm-$ROCM_VERSION
export HIP_PATH=/opt/rocm-$ROCM_VERSION
export LLVM_PATH=/opt/rocm-$ROCM_VERSION/lib/llvm
cd /home
python3 -m pip uninstall jax jaxlib jax-rocm60-pjrt jax-rocm60-plugin -y
python3 -m pip install numpy wheel build ninja patchelf
git clone https://github.com/ROCm/xla.git -b ${XLA_BRANCH}
# git clone https://github.com/yaomingamd/xla.git -b ${XLA_BRANCH}
git clone https://github.com/ROCm/jax.git -b ${JAX_BRANCH}
cd jax
python3 ./build/build.py build --wheels=jaxlib,jax-rocm-plugin,jax-rocm-pjrt --rocm_version=60 \
     --local_xla_path="/home/xla/" --rocm_path="/opt/rocm-$ROCM_VERSION" --rocm_amdgpu_targets=gfx942 \
     --clang_path="/opt/rocm-$ROCM_VERSION/lib/llvm/bin/clang"
python setup.py bdist_wheel