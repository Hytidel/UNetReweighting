#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "0.98" "1.1" "True"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "0.95" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "0.96" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "0.97" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "0.98" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "0.99" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "1.0" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "1.01" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "1.02" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "1.03" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "1.04" "1.1"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "1.05" "1.1"

./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "0.98" "1.15"
./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "0.98" "1.2"

./script/cal_importance_weight/run_cal_importance_weight_all/sd-turbo.sh ${gpu_id} "0.98" "1.1" "True"
