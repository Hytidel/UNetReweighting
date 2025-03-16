#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

python main.py pipeline=sdxl-turbo task=do_importance_probe/t2i/sdxl-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/photo.json"
python main.py pipeline=sdxl-turbo task=do_importance_probe/t2i/sdxl-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/anime.json"
python main.py pipeline=sdxl-turbo task=do_importance_probe/t2i/sdxl-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/concept-art.json"
python main.py pipeline=sdxl-turbo task=do_importance_probe/t2i/sdxl-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/paintings.json"
