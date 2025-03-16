#!/bin/bash

# defaults
DEFAULT_GPU_ID=0
DEFAULT_WEIGHT_MATRIX_NAME="default"

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

# weight matrix name
weight_matrix_name=${2:-$DEFAULT_WEIGHT_MATRIX_NAME}


# training set
python main.py pipeline=sdxl-turbo task=sample/t2i/sdxl-turbo/run_scheduled task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/anime.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=42 task.task_seed.seed_range_r=42  task.weight_matrix.weight_matrix_name=${weight_matrix_name}
python main.py pipeline=sdxl-turbo task=sample/t2i/sdxl-turbo/run_scheduled task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/concept-art.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=42 task.task_seed.seed_range_r=42 task.weight_matrix.weight_matrix_name=${weight_matrix_name}
python main.py pipeline=sdxl-turbo task=sample/t2i/sdxl-turbo/run_scheduled task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/paintings.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=42 task.task_seed.seed_range_r=42 task.weight_matrix.weight_matrix_name=${weight_matrix_name}
python main.py pipeline=sdxl-turbo task=sample/t2i/sdxl-turbo/run_scheduled task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/photo.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=42 task.task_seed.seed_range_r=42 task.weight_matrix.weight_matrix_name=${weight_matrix_name}

# test set
python main.py pipeline=sdxl-turbo task=sample/t2i/sdxl-turbo/run_scheduled task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/anime.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=21 task.task_seed.seed_range_r=21 task.weight_matrix.weight_matrix_name=${weight_matrix_name}
python main.py pipeline=sdxl-turbo task=sample/t2i/sdxl-turbo/run_scheduled task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/concept-art.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=21 task.task_seed.seed_range_r=21 task.weight_matrix.weight_matrix_name=${weight_matrix_name}
python main.py pipeline=sdxl-turbo task=sample/t2i/sdxl-turbo/run_scheduled task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/paintings.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=21 task.task_seed.seed_range_r=21 task.weight_matrix.weight_matrix_name=${weight_matrix_name}
python main.py pipeline=sdxl-turbo task=sample/t2i/sdxl-turbo/run_scheduled task.prompt.prompt_json_path="/root/autodl-tmp/zhwang/HPDv2/benchmark/photo.json" task.task.num_inference_step=2 task.task_seed.seed_range_l=21 task.task_seed.seed_range_r=21 task.weight_matrix.weight_matrix_name=${weight_matrix_name}
