#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

python main.py task=cal_importance_ranking/run task.category_root_path="./tmp/importance_probe/run_sdxl_family/anime/sdxl-turbo/step-2_seed-42"
python main.py task=cal_importance_ranking/run task.category_root_path="./tmp/importance_probe/run_sdxl_family/concept-art/sdxl-turbo/step-2_seed-42"
python main.py task=cal_importance_ranking/run task.category_root_path="./tmp/importance_probe/run_sdxl_family/paintings/sdxl-turbo/step-2_seed-42"
python main.py task=cal_importance_ranking/run task.category_root_path="./tmp/importance_probe/run_sdxl_family/photo/sdxl-turbo/step-2_seed-42"
