#!/bin/bash

# defaults
DEFAULT_GPU_ID=0
DEFAULT_WEIGHT_MATRIX_NAME="default"

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}


# training set
python main.py pipeline=sd-turbo task=sample/t2i/sd-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/anime.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=42 task.task_seed.seed_range_r=42
python main.py pipeline=sd-turbo task=sample/t2i/sd-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/concept-art.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=42 task.task_seed.seed_range_r=42
python main.py pipeline=sd-turbo task=sample/t2i/sd-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/paintings.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=42 task.task_seed.seed_range_r=42
python main.py pipeline=sd-turbo task=sample/t2i/sd-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/photo.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=42 task.task_seed.seed_range_r=42

# test set
python main.py pipeline=sd-turbo task=sample/t2i/sd-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/anime.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=21 task.task_seed.seed_range_r=21
python main.py pipeline=sd-turbo task=sample/t2i/sd-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/concept-art.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=21 task.task_seed.seed_range_r=21
python main.py pipeline=sd-turbo task=sample/t2i/sd-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/paintings.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=21 task.task_seed.seed_range_r=21
python main.py pipeline=sd-turbo task=sample/t2i/sd-turbo/run task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/photo.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=21 task.task_seed.seed_range_r=21
