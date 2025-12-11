IMAGE_NAME="rocm/pyt-megatron-lm-jax-nightly-private:jax_rocm7.0_20250930"
CONTAINER_NAME="axlearn_jax6"
AXLEARN_PATH="TODO ADD YOUR PATH HERE"

BASH_CMD="cd ${AXLEARN_PATH}; pip install -e ."


docker run -it \
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
    -v $HOME:$HOME \
    -v /home:/home \
    --name $CONTAINER_NAME \
    $IMAGE_NAME \
    /bin/bash -c "${BASH_CMD}; exec bash "



