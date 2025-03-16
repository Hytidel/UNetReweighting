#!/bin/bash

# defaults
DEFAULT_GPU_ID=0
DEFAULT_WEIGHT_LOW="0.95"
DEFAULT_WEIGHT_HIGH="1.1"
DEFAULT_REVERSE_RANKING="False"

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

# weight range
weight_low=${2:-$DEFAULT_WEIGHT_LOW}
weight_high=${3:-$DEFAULT_WEIGHT_HIGH}

# reverse ranking
reverse_ranking=${4:-$DEFAULT_REVERSE_RANKING}

# anime
python main.py task=cal_importance_weight/run task.category_root_path="./tmp/importance_probe/run_sd_family/anime/sd-turbo/step-2_seed-42" task.importance_weight.low_importance_weight=${weight_low} task.importance_weight.high_importance_weight=${weight_high} task.reverse_ranking=${reverse_ranking}

# concept-art
python main.py task=cal_importance_weight/run task.category_root_path="./tmp/importance_probe/run_sd_family/concept-art/sd-turbo/step-2_seed-42" task.importance_weight.low_importance_weight=${weight_low} task.importance_weight.high_importance_weight=${weight_high} task.reverse_ranking=${reverse_ranking}

# paintings
python main.py task=cal_importance_weight/run task.category_root_path="./tmp/importance_probe/run_sd_family/paintings/sd-turbo/step-2_seed-42" task.importance_weight.low_importance_weight=${weight_low} task.importance_weight.high_importance_weight=${weight_high} task.reverse_ranking=${reverse_ranking}

# photo
python main.py task=cal_importance_weight/run task.category_root_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42" task.importance_weight.low_importance_weight=${weight_low} task.importance_weight.high_importance_weight=${weight_high} task.reverse_ranking=${reverse_ranking}
