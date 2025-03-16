#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

python main.py task=cal_importance_ranking/run task.category_root_path="./tmp/importance_probe/run_sdxl_family/anime/sdxl/step-2_seed-42"
python main.py task=cal_importance_ranking/run task.category_root_path="./tmp/importance_probe/run_sdxl_family/concept-art/sdxl/step-2_seed-42"
python main.py task=cal_importance_ranking/run task.category_root_path="./tmp/importance_probe/run_sdxl_family/paintings/sdxl/step-2_seed-42"
python main.py task=cal_importance_ranking/run task.category_root_path="./tmp/importance_probe/run_sdxl_family/photo/sdxl/step-2_seed-42"
