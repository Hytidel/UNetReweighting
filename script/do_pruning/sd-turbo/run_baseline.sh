#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

# skip-1
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[0]
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[1]
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[2]
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[3]
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[4]
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[5]
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[6]

# skip-2
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[0,6]
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[1,5]
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix="static" task.skipping_strategy.skip_block_idx_list=[2,4]
