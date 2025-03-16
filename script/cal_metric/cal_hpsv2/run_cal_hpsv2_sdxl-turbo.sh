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
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/run_sdxl_family/sdxl-turbo/step-2_seed-42/anime" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/run_sdxl_family/sdxl-turbo/step-2_seed-42/concept-art" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/run_sdxl_family/sdxl-turbo/step-2_seed-42/paintings" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/run_sdxl_family/sdxl-turbo/step-2_seed-42/photo" task.img_root_path.weight_matrix_name=${weight_matrix_name}

# test set
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/run_sdxl_family/sdxl-turbo/step-2_seed-21/anime" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/run_sdxl_family/sdxl-turbo/step-2_seed-21/concept-art" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/run_sdxl_family/sdxl-turbo/step-2_seed-21/paintings" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/run_sdxl_family/sdxl-turbo/step-2_seed-21/photo" task.img_root_path.weight_matrix_name=${weight_matrix_name}
