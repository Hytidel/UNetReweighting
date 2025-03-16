#!/bin/bash

# defaults
DEFAULT_GPU_ID=0
DEFAULT_WEIGHT_MATRIX_NAME="default"

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

# weight matrix name
weight_matrix_name=${2:-$DEFAULT_WEIGHT_MATRIX_NAME}

#!/bin/bash

# defaults
DEFAULT_GPU_ID=0
DEFAULT_WEIGHT_MATRIX_NAME="default"

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

# weight matrix name
weight_matrix_name=${2:-$DEFAULT_WEIGHT_MATRIX_NAME}

# SD-Turbo
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sd_family/sd-turbo/step-1_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sd_family/sd-turbo/step-2_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sd_family/sd-turbo/step-3_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}

# SD
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sd_family/sd/step-10_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sd_family/sd/step-15_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sd_family/sd/step-20_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}

# SDXL-Turbo
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sdxl_family/sdxl-turbo/step-1_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sdxl_family/sdxl-turbo/step-2_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sdxl_family/sdxl-turbo/step-3_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}

# SDXL
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sdxl_family/sdxl/step-10_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sdxl_family/sdxl/step-15_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}
python main.py task=cal_metric/run_cal_hpsv2 task.img_root_path.category_root_path="./tmp/sample/sdxl_family/sdxl/step-20_seed-42" task.img_root_path.weight_matrix_name=${weight_matrix_name}
