#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

python main.py task=plot/scatter_fid_lpips task.data.pruning_res_dict_path="./tmp/sample/sd_family/sd-turbo/step-2_seed-42/some_cut_up_fruit/png/sd-turbo.yaml" task.data.split="training" task.data.num_skipped_block=1
python main.py task=plot/scatter_fid_lpips task.data.pruning_res_dict_path="./tmp/sample/sd_family/sd-turbo/step-2_seed-42/some_cut_up_fruit/png/sd-turbo.yaml" task.data.split="training" task.data.num_skipped_block=2

python main.py task=plot/scatter_fid_lpips task.data.pruning_res_dict_path="./tmp/sample/sd_family/sd-turbo/step-2_seed-21/some_cut_up_fruit/png/sd-turbo.yaml" task.data.split="test" task.data.num_skipped_block=1
python main.py task=plot/scatter_fid_lpips task.data.pruning_res_dict_path="./tmp/sample/sd_family/sd-turbo/step-2_seed-21/some_cut_up_fruit/png/sd-turbo.yaml" task.data.split="test" task.data.num_skipped_block=2
