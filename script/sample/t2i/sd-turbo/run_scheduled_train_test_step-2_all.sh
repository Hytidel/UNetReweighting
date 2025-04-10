#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}


./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "0.95_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "0.96_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "0.97_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "0.98_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "0.99_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "1.0_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "1.01_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "1.02_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "1.03_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "1.04_1.1"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "1.05_1.1"

./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "0.98_1.15"
./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "0.98_1.2"

./script/sample/t2i/sd-turbo/run_scheduled_train_test_step-2.sh ${gpu_id} "0.98_1.1_rev"
