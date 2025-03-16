#!/bin/bash

# defaults
DEFAULT_GPU_ID=0

# GPU ID
gpu_id=${1:-$DEFAULT_GPU_ID}
export CUDA_VISIBLE_DEVICES=${gpu_id}

python main.py pipeline=sd task.task.num_inference_step=10 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix=False
python main.py pipeline=sd task.task.num_inference_step=10 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=0 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=10 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=1 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=10 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=2 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=10 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=3 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=10 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=4 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=10 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=5 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=10 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=6 task.weight_matrix.static_weight_matrix.block_weight=1.1

python main.py pipeline=sd task.task.num_inference_step=15 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix=False
python main.py pipeline=sd task.task.num_inference_step=15 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=0 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=15 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=1 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=15 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=2 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=15 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=3 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=15 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=4 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=15 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=5 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=15 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=6 task.weight_matrix.static_weight_matrix.block_weight=1.1

python main.py pipeline=sd task.task.num_inference_step=20 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix=False
python main.py pipeline=sd task.task.num_inference_step=20 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=0 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=20 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=1 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=20 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=2 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=20 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=3 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=20 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=4 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=20 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=5 task.weight_matrix.static_weight_matrix.block_weight=1.1
python main.py pipeline=sd task.task.num_inference_step=20 task.task.batch_size=5 task.save_sample.num_sample_per_prompt=20 task=sample/t2i/sd/template task.task_seed.random_seed=False task.weight_matrix.load_weight_matrix="static" task.weight_matrix.static_weight_matrix.block_idx_list=6 task.weight_matrix.static_weight_matrix.block_weight=1.1
