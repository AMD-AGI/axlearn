## AXLearn + ROCm 2026
- Updated axlearn main to work with ROCm 7 and JAX 6.2 (dec. 10, 2025 axlearn main)
- Using JAX 0.6.2 + ROCm 7.0: `docker.io/rocm/pyt-megatron-lm-jax-nightly-private:jax_rocm7.0_jax_0.6.2_latest`
- Evaluated on MI350X

### Get Started
1. Update start_docker.sh to use the correct path and then run `bash start_docker.sh` which will enter the environment and install axlearn.
2. Run simple benchmarking with `bash scripts/benchmark_70B_fuji_mi350.sh` (no TE support yet)


### See original README.md: [ORIGINAL_README.md](ORIGINAL_README.md)
