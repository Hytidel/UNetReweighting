#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

python main.py pipeline=sd-turbo task=sample/t2i/sd-turbo/a_little_girl_holding

