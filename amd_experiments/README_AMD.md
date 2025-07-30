# Instructions for reproducing AMD Experiments
## Environment set-up 
* Checkout the correct repo + branch for AMD
* For dense models use this PR of the public repo:
```
git clone git@github.com:apple/axlearn.git
git fetch origin pull/TODO-ADD-PR
branch git checkout TODO-ADD-PR
```
* For MoE models use the shared private repo `AMD-AIG-AIMA/axlearn_dev`: 
```
git clone git@github.com:AMD-AIG-AIMA/axlearn_dev.git && cd axlearn_dev    
git checkout eval/mesh-shape-configs-testing-moe-rocm-jax-maxtext-training-private-20250604-te2.1-fp8-tf2.18.1
```

- move the `amd_experiments` directory we shared with you to the `axlearn` or `axlearn_dev` (for MoE) parent directory
  
- Log-in to docker-hub (reach out for docker-hub token credentials) 
```
docker login  -u ADD_USER -p ADD_TOKEN
```
- run `bash ./amd_experiments/start_docker.sh`
	- This script downloads and starts a container of `rocm/jax-maxtext-training-private:maxtext-v25.6` 
	- Detects and installs the correct drivers for the Network Interface Card (NIC) detected on the cluster (Pollara/Mellanox/Broadcom)
	- Installs axlearn and additional dependencies

---
## Dense model experiments

The experiments for the dense models are based off this PR: TODO add PR<br>
We study the 70B Fuji dense model under different configurations. The main script for the Fuji 70B model is
```
bash amd_experiments/benchmark_70B_fuji.sh
```
By default, it runs on a 1) single node with 2) FSDP parallelism 3) Tuned-GEMMs if available 4) `ROCM TransfomerEngine` Attention and 5) runs with **per-node** batch-size of 32. Below we list different modifications of this base script:

##### Multi-node scaling
We don't assume the cluster is equipped with SLURM (or NFS) but can also share slurm based multi-node launcher setup if interested. 
   
* To extend the script to multi-node, specify the following environment variables 
	1. The number of nodes `NUM_PROCESSES`
	2. The head node `HEAD_NODE` 
	3. process ID (different for each node)`PROCESS_ID`
	   
* **Manual launching**: For example, manually run the environment set-up on 2 nodes, and inside the docker run 
```
NUM_PROCESSES=2 PROCESS_ID=0 HEAD_NODE=MY_NODE_NAME \
bash amd_experiments/benchmark_70B_fuji.sh
``` 
for the first node and on the 2nd node with ```PROCESS_ID=1```

* This Multi-node training reproduces the results presented for multi-node scaling with FSDP, by specifying
```
PER_NODE_BATCH_SIZE=128 MESH_DCN_FSDP=$NUM_PROCESSES bash amd_experiments/benchmark_70B_fuji.sh
```

##### Pallas support for ROCm: Pallas Attention
We add support of [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) which enables a Pallas-based attention implementation as opposed to the default`ROCm TransformerEngine` 

* **Running with Pallas*** To run with Pallas Attention set ```FORCE_PALLAS=1```: 
```
FORCE_PALLAS=1 bash amd_experiments/benchmark_70B_fuji.sh
```
* **Configuring kernel launch parameters for Pallas** we expose the following kernel parameters of Pallas that can be used to best fit the kernel for different hardware. They are each set with default choices that we identified to be most performant and can be overridden as environment variables  
	* `GPU_BLOCK_Q` # block sizes of Query matrix in kernel
	* `GPU_BLOCK_K` # block sizes of key matrix in kernel
	* `NUM_WARPS` # number of simultaneous wraps for execution
	* `NUM_STAGES` # number of stages of execution
example: 
```
FORCE_PALLAS=1 GPU_BLOCK_Q=32 bash amd_experiments/benchmark_70B_fuji.sh
```
##### Mixed Parallelism experiments
In addition to FSDP scaling runs, we ran mixed-parallelism experiments, where we considered Tensor/Sequence Parallelism x FSDP x Data parallelism. In particular we present results for 
TP2-FSDP4-DP4 and SP2-FSDP4-DP4 (DP for inter-node communication) this can be run by running
```
MESH_MODEL=2 MESH_FSDP=4 bash amd_experiments/benchmark_70B_fuji.sh
```
and 
```
MESH_SEQ=2 MESH_FSDP=4 bash amd_experiments/benchmark_70B_fuji.sh
```

##### hipblaslt Tuned GEMMs 
`hipblaslt` tuned-GEMMs are optimized configurations (GEMM kernels and launch parameters) collected for the different workloads above for `hipblaslt` framework. We placed the tuned configurations under `amd_experiments/tuned_gemms`. By setting`TUNED_GEMMS=1` (set true by default) the script will search if the workload (batch size, parallelism setting) has available tuned-GEMMs and use the tuned GEMMs instead of the default ones. tuned GEMMs can be turned off with `TUNED_GEMMS=0` 
```
TUNED_GEMMS=0 bash amd_experiments/benchmark_70B_fuji.sh  
```
Under the hood hipblaslt tuned GEMMs specifies a configuration file to ```HIPBLASLT_TUNING_OVERRIDE_FILE``` which will use it as reference when launching GEMM kernels

---

## MoE model experiments (WIP)

We use the same environment for MoE models, to set-up, follow the instructions above but make sure to use the correct repo + branch combination (see above). we follow a similar structure for the MoE models below:

The experiments for the MoE models are based off 

We study the 44B Envy Mixture of experts model under different configurations. The main script for the Envy model is
```
bash amd_experiments/benchmark_moe_envy.sh
```
By defaults it runs on a 1) single node with 2) Expert parallelism 3) Tuned-GEMMs if available 4) `ROCM TransfomerEngine` Attention and 5) runs with **per-node** batch-size of 16. Below we list different modifications of this base script:
##### Multi-node scaling
Using the same convention as above the model can be launched manually for multi-node.

You can also automate multi-node launching by running
````bash
NODE_1=<node-1-hostname>
NODE_2=<node-2-hostname>
NODE_3=<node-3-hostname>
NODES=($(hostname) $NODE_1 $NODE_2 $NODE_3)
BATCH_SIZE_BASE=16
bash example/rocm/run.sh
````
##### hipblaslt Tuned GEMMs 
We use the same configurations to configure the use of tuned GEMMs











