IMAGE_NAME="rocm/jax-maxtext-training-private:maxtext-v25.6"
CONTAINER_NAME="amd_axlearn"
BASH_CMD="bash ./amd_experiments/setup.sh "


docker run -it --rm \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=IPC_LOCK \
    --volume /dev/infiniband:/dev/infiniband \
    --tmpfs /dev/shm:size=2200G \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $PWD:/workspace \
    -v /home:/home \
    --name $CONTAINER_NAME \
    $IMAGE_NAME \
    /bin/bash -c "${BASH_CMD}; exec bash "



