# AXLearn with ROCm


## Single-Node Training
### How to Run?
1. Launch ROCm image
    ```bash
    IMAGE_NAME=rocm/jax-training:maxtext-v25.5
    docker run -it \
        --ipc=host \
        --network=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --security-opt seccomp=unconfined \
        --group-add video \
        --privileged \
        -v $HOME/.ssh:/root/.ssh \
        --name axlearn_training \
        $IMAGE_NAME
    ```
2. Install AXLearn
    ```bash
    git clone https://github.com/AMD-AIG-AIMA/axlearn.git
    cd axlearn
    pip install ".[core]"
    ```
3. Launch FSDP only training
    ```bash
    cd example/rocm
    bash launch_70B_single_node.sh
    ```
4. For other mesh shape, please check out `mesh_axes_tests_on_70B_single_node.sh`.

### Expected Averaged Step Time
| Mesh Shape ([INI])  | Averaged Step Time (Second) |
| ------------- | ------------- |
| FSDP8​ | 9.85​ |


## Multi-Node Training
**NOTE**: Slurm is expected here.

### How to Run?
```bash
# In Slurm interactive mode:
srun --ntasks=4 bash -c 'SLRUM_JOB_NAME=axlearn bash run_axlearn_multinodes.sh'
# For sbatch:
sbatch sbatch_job.sh "<flags-if-any> bash run_axlearn_multinodes.sh"
```


### Expected Averaged Step Time
We report the expected averaged step time for different mesh shapes on a 4-node MI300X cluster.
- The expected averaged step time is from the last 20 steps among 40 steps in a training run. 
- The total batch size is set to 64 for all shapes.

#### Broadcom InfinityBand

| Mesh Shape ([DCN] / [INI])  | Averaged Step Time (Second) |
| ------------- | ------------- |
| FSDP4 / FSDP8​ | 9.53​ |
| DP4 / FSDP8​ | 10.49​ |
| PP4 / FSDP8​ | OOM​ |
| DP4 / FSDP2, SP4​ | 13.51​ |
| DP4 / FSDP4, SP2​ | 12.43​ |
| DP4 / FSDP2, TP4​ | 16.37​ |
| DP4 / FSDP4, TP2​ | 14.05​ |

#### Mellanox InfinityBand

| Mesh Shape ([DCN] / [INI])  | Averaged Step Time (Second) |
| ------------- | ------------- |
|FSDP4 / FSDP8​ | 9.77​ |
|DP4 / FSDP8​ | 10.7​ |
|PP4 / FSDP8​ | OOM​ |
|DP4 / FSDP2, SP4​ | 13.84​ |
|DP4 / FSDP4, SP2​ | 12.86​ |
|DP4 / FSDP2, TP4​ | 16.65​ |
|DP4 / FSDP4, TP2​ | 14.33​ |


