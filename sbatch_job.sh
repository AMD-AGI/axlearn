#!/bin/bash
#SBATCH --job-name=axlearn
#SBATCH --nodes=4               # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00         # Max time
#SBATCH --output=slurm_log/output_%j.log     # Output log
#SBATCH --error=slurm_log/err_%j.log      # Error log


# clean up hook
cleanup() {
    srun docker stop $(docker ps -q) || true
    srun docker rm -f $SLURM_JOB_NAME || true
    echo "Clean up completed."
}
set -eE
trap 'cleanup; exit 1' SIGTERM SIGUSR1 ERR

# vars
__SCRIPT_AND_ARGS=$@  # gsm8k_task.sh, etc.
if [[ -z "$@" ]]; then
    echo "Please provide the script and arguments to run" >&2
    exit 1
fi

# Stop any existing Docker containers to avoid unwanted interruptions
cleanup

# run
srun bash -c "$__SCRIPT_AND_ARGS"

# clean up when finish
cleanup
