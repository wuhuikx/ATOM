docker run -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --network host \
    --shm-size=16G \
    --name vllm_atom_hattie \
    -v /home/hatwu@amd.com:/home/hatwu@amd.com \
    -v /data/models:/data/models \
    -v /dev/shm:/dev/shm \
    -v /mnt/raid0:/mnt/raid0 \
    rocm/vllm-dev:nightly_main_20260118 \
    /bin/bash

# docker exec -it vllm_atom_hattie /bin/bash
#
# tmux new -s mysession
# tmux attach -t mysession
