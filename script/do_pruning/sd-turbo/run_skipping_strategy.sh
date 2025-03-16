#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

# skip-1
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-1/skip-1_0.yaml"
# python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-1/skip-1_1.yaml"
# python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-1/skip-1_2.yaml"
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-1/skip-1_3.yaml"

# skip-2
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-2/skip-2_0.yaml"
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-2/skip-2_1.yaml"
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-2/skip-2_2.yaml"
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-2/skip-2_3.yaml"
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-2/skip-2_4.yaml"
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-2/skip-2_5.yaml"
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-2/skip-2_6.yaml"
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-2/skip-2_7.yaml"
python main.py pipeline=sd-turbo task=do_pruning/sd-turbo task.skipping_strategy.load_weight_threshold_matrix=True task.skipping_strategy.weight_threshold_matrix_path="./tmp/importance_probe/run_sd_family/photo/sd-turbo/step-2_seed-42/some_cut_up_fruit/skipping_strategy/skip-2/skip-2_8.yaml"
