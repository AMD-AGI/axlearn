## AXLearn + ROCm 2026
- Updated axlearn main to work with ROCm 7 and JAX 6.2 (dec. 10, 2025 axlearn main)
- Using JAX 0.6.2 + ROCm 7.0: `rocm/pyt-megatron-lm-jax-nightly-private:jax_rocm7.0_20250930`
- Evaluated on MI350X

### Get Started
1. Update start_docker.sh to use the correct path and then run `bash start_docker.sh` which will enter the environment and install axlearn.
2. Run simple benchmarking with `bash scripts/benchmark_70B_fuji_mi350.sh` ~~(no TE support yet)~~ TE 2.2 supported (issues on TE 2.4)


### See original README.md: [ORIGINAL_README.md](ORIGINAL_README.md)
