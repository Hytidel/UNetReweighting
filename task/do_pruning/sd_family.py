from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.cuda.amp import GradScaler

import numpy as np

import subprocess

from tqdm.auto import tqdm

from matplotlib import pyplot as plt
from PIL import Image

import time

import random

import gc

from pathlib import Path

import concurrent.futures as cf

from util.basic_util import (
    pause, 
    get_global_variable, 
    is_none, 
    get_true_value, 
    get_timestamp
)
from util.torch_util import (
    get_generator, get_optim, get_lr_scheduler, get_criterion, 
    save_model_ckpt, load_model_ckpt, 
    get_selected_state_dict, save_model_state_dict, load_model_state_dict, 
    determine_enable_grad, 
    soft_update_model, 
    get_model_num_param, 
    get_current_lr_list
)
from util.pipeline_util import (
    load_pipeline, load_unet, 
    load_scheduler, 
    get_inference_step_minus_one, 
    img_latent_to_pil
)
from util.yaml_util import (
    load_yaml, save_yaml, 
    convert_numpy_type_to_native_type
)
from util.numpy_util import tsfm_to_2d_matrix

from scheduled_model.do_pruning.pipeline_stable_diffusion import register_pipeline_stable_diffusion
from scheduled_model.unet_2d_condition import register_unet_2d_condition_model
from importance_probe.importance_probe_util import get_weight_matrix
from ..util.prompt_util import (
    preprocess_prompt, 
    get_folder_name
)


# @torch.no_grad()
def do_pruning_sd_family_implement(
    cfg: DictConfig
):
    # ---------= [Basic Global Variables] =---------
    exp_name = get_global_variable("exp_name")
    start_time = get_global_variable("start_time")
    device = get_global_variable("device")
    seed = get_global_variable("seed")
    exp_time_str = f"{exp_name}_{start_time}"

    concurrent_max_worker = get_global_variable("concurrent_max_worker")

    # ---------= [Task] =---------
    logger(f"[Task] Loading started. ")

    height = get_true_value(cfg["task"]["task"]["height"])
    width = get_true_value(cfg["task"]["task"]["width"])
    num_inference_step = get_true_value(cfg["task"]["task"]["num_inference_step"])
    guidance_scale = get_true_value(cfg["task"]["task"]["guidance_scale"])
    prompt = get_true_value(cfg["task"]["task"]["prompt"])
    negative_prompt = get_true_value(cfg["task"]["task"]["negative_prompt"])

    logger(f"    (height, width): ({height}, {width})")
    logger(f"    num_inference_step: {num_inference_step}")
    logger(f"    guidance_scale: {guidance_scale}")
    logger(f"    prompt: {prompt}")
    logger(f"    negative_prompt: {negative_prompt}")

    num_sample_per_prompt = get_true_value(cfg["task"]["task"]["num_sample_per_prompt"])
    batch_size = get_true_value(cfg["task"]["task"]["batch_size"])
    
    logger(f"    num_sample_per_prompt: {num_sample_per_prompt}")
    logger(f"    batch_size: {batch_size}")

    num_sample = num_sample_per_prompt
    num_batch = (num_sample + batch_size - 1) // batch_size

    logger(f"    num_sample: {num_sample}")
    logger(f"    num_batch: {num_batch}")

    logger(
        f"[Task] Loading finished. "
        "\n"
    )

    # ---------= [Pipeline] =---------
    logger(f"[Pipeline] Loading started. ")

    pipeline_type = get_true_value(cfg["pipeline"]["pipeline_type"])
    pipeline_path = get_true_value(cfg["pipeline"]["pipeline_path"])
    torch_dtype = get_true_value(cfg["pipeline"]["torch_dtype"])
    variant = get_true_value(cfg["pipeline"]["variant"])

    pipeline = load_pipeline(
        pipeline_type = pipeline_type, 
        pipeline_path = pipeline_path, 
        torch_dtype = torch_dtype, 
        variant = variant
    )

    # register custom `forward()`
    register_pipeline_stable_diffusion(pipeline)
    
    scheduler_type = get_true_value(cfg["pipeline"]["scheduler"])
    load_scheduler(
        pipeline = pipeline, 
        scheduler_type = scheduler_type
    )

    inference_step_minus_one = get_inference_step_minus_one(scheduler_type)

    pipeline.to(device)  # move to GPU after registration

    # save VRAM by offloading the model to CPU
    # pipeline.enable_model_cpu_offload()

    logger(f"    pipeline: {type(pipeline)}")
    # logger(f"    pipeline: {pipeline}")

    sd_type = "sd"
    if "turbo" in pipeline_path:
        sd_type = "sd-turbo"
    
    num_attn_block = pipeline.unet.num_attn_block

    logger(
        f"[Pipeline] Loading finished. "
        "\n"
    )

    # ---------= [U-Net] =---------
    logger(f"[U-Net] Loading started. ")

    pipeline.stu_unet = load_unet(
        unet_type = "UNet2DConditionModel", 
        pipeline = pipeline
    )

    register_unet_2d_condition_model(pipeline.stu_unet)

    pipeline.stu_unet.to(device)  # move to GPU after registration
    
    logger(
        f"[U-Net] Loading finished. "
        "\n"
    )

    # ---------= [Task Seed] =---------
    logger(f"[Task Seed] Loading started. ")

    random_seed = get_true_value(cfg["task"]["task_seed"]["random_seed"])
    seed_range_l = get_true_value(cfg["task"]["task_seed"]["seed_range_l"])
    seed_range_r = get_true_value(cfg["task"]["task_seed"]["seed_range_r"])

    logger(f"    random_seed: {random_seed}")
    logger(f"    seed_range_l: {seed_range_l}")
    logger(f"    seed_range_r: {seed_range_r}")

    if random_seed:
        if seed_range_l == seed_range_r:
            seed = seed_range_l
            random_seed = False

            seed_list = [
                seed + i % num_sample_per_prompt \
                    for i in range(num_sample)
            ]
        else:
            seed = get_timestamp(to_int = True)
            seed_list = [
                (seed + i % num_sample_per_prompt) % (seed_range_r - seed_range_l + 1) + seed_range_l \
                    for i in range(num_sample)
            ]
    else:
        seed_list = [
            seed + i % num_sample_per_prompt \
                for i in range(num_sample)
            ]

    logger(
        f"[Task Seed] Loading finished. "
        "\n"
    )
    
    # ---------= [Skipping Strategy] =---------
    logger(f"[Skipping Strategy] Loading started. ")

    load_weight_threshold_matrix = get_true_value(cfg["task"]["skipping_strategy"]["load_weight_threshold_matrix"])

    logger(f"    load_weight_threshold_matrix: {load_weight_threshold_matrix}")

    if load_weight_threshold_matrix == "static":
        skip_block_idx_list = get_true_value(cfg["task"]["skipping_strategy"]["skip_block_idx_list"])

        logger(f"    skip_block_idx_list: {skip_block_idx_list}")

        weight_threshold_matrix = np.zeros(
            shape = (num_inference_step, num_attn_block)
        )

        for skip_block_idx in skip_block_idx_list:
            weight_threshold_matrix[:, skip_block_idx] = 1.5
        
        skip_block_idx_str_list = [
            str(skip_block_idx) \
                for skip_block_idx in skip_block_idx_list
        ]
        skip_block_idx_str = ','.join(skip_block_idx_str_list)
        weight_threshold_matrix_name = f"static-{skip_block_idx_str}"
    elif load_weight_threshold_matrix:
        weight_threshold_matrix_path = get_true_value(cfg["task"]["skipping_strategy"]["weight_threshold_matrix_path"])
        weight_threshold_matrix_path = Path(weight_threshold_matrix_path)

        logger(f"    weight_threshold_matrix_path: {weight_threshold_matrix_path}")

        weight_threshold_matrix \
            = load_yaml(weight_threshold_matrix_path)["history_weight_threshold_matrix_list"][-1]

        weight_threshold_matrix_name = weight_threshold_matrix_path.stem
    else:
        weight_threshold_matrix = np.zeros(
            shape = (num_inference_step, num_attn_block)
        )

        weight_threshold_matrix_name = "default"

    logger(f"    weight_threshold_matrix: {weight_threshold_matrix}")

    logger(
        f"[Skipping Strategy] Loading finished. "
        "\n"
    )

    # ---------= [Save Sample] =---------
    logger(f"[Save Sample] Loading started. ")

    save_sample_root_path = get_true_value(cfg["task"]["save_sample"]["save_sample_root_path"])
    save_sample_root_path = Path(save_sample_root_path)

    logger(f"    save_sample_root_path: {save_sample_root_path}")

    logger(
        f"[Save Sample] Loading finished. "
        "\n"
    )

    # ---------= [Do Pruning] =---------
    logger(f"[Do Pruning] Loading started. ")

    num_epoch = get_true_value(cfg["task"]["do_pruning"]["num_epoch"])

    logger(f"    num_epoch: {num_epoch}")

    lr = get_true_value(cfg["task"]["do_pruning"]["finetune"]["lr"])
    optim_type = get_true_value(cfg["task"]["do_pruning"]["finetune"]["optim"])
    criterion_type = get_true_value(cfg["task"]["do_pruning"]["finetune"]["criterion"])

    logger(f"    lr: {lr}")
    logger(f"    optim_type: {optim_type}")
    logger(f"    criterion_type: {criterion_type}")

    save_training_loss_list_in_process = get_true_value(cfg["task"]["do_pruning"]["training_loss_list"]["save_training_loss_list_in_process"])
    save_training_loss_list_epoch_interval = get_true_value(cfg["task"]["do_pruning"]["training_loss_list"]["save_training_loss_list_epoch_interval"])

    logger(f"    save_training_loss_list_in_process: {save_training_loss_list_in_process}")
    logger(f"    save_training_loss_list_epoch_interval: {save_training_loss_list_epoch_interval}")

    save_model_ckpt_start_epoch = get_true_value(cfg["task"]["do_pruning"]["model_ckpt"]["save_model_ckpt_start_epoch"])
    if is_none(save_model_ckpt_start_epoch):
        save_model_ckpt_start_epoch = num_epoch

    logger(f"    save_model_ckpt_start_epoch: {save_model_ckpt_start_epoch}")

    optim = get_optim(
        optim_type = optim_type, 
        model = pipeline.stu_unet, 
        lr = lr
    )

    criterion = get_criterion(criterion_type = criterion_type)

    logger(f"    optim: {optim}")
    logger(f"    criterion: {criterion}")

    logger(
        f"[Do Pruning] Loading finished. "
        "\n"
    )

    # ---------= [All Components Loaded] =---------
    logger(
        f"All components loaded. "
        "\n"
    )

    # ---------= [Preprocess] =---------
    folder_name_list = []  # useless

    prompt = preprocess_prompt(prompt)
    folder_name = get_folder_name(
        prompt = prompt, 
        used_folder_name_list = folder_name_list
    )

    save_sample_root_path = save_sample_root_path / sd_type / folder_name

    if random_seed:
        save_sample_root_path = save_sample_root_path / f"step-{num_inference_step}_seed-{seed_range_l}"
    else:
        save_sample_root_path = save_sample_root_path / f"step-{num_inference_step}_seed-{seed}"

    save_sample_root_path = save_sample_root_path / weight_threshold_matrix_name
    save_sample_root_path = save_sample_root_path / f"{num_sample_per_prompt}_{batch_size}_{num_epoch}"

    save_training_loss_list_root_path = save_sample_root_path / "training_loss_list"
    save_model_ckpt_root_path = save_sample_root_path / "model_ckpt"

    save_inference_process_dict = {
        "noise_pred": False, 
        "latent": False, 
        "pil": True
    }

    task_cfg_dict = {
        "task": {
            "num_sample_per_prompt": num_sample_per_prompt, 
            "batch_size": batch_size, 

            "height": height, 
            "width": width, 

            "num_inference_step": num_inference_step, 

            "guidance_scale": guidance_scale, 

            "prompt": prompt, 
            "negative_prompt": negative_prompt
        }, 
        "skipping_strategy": {
            "load_weight_threshold_matrix": load_weight_threshold_matrix, 
            "weight_threshold_matrix": weight_threshold_matrix
        }, 
        "do_pruning": {
            "num_epoch": num_epoch, 

            "finetune": {
                "lr": lr, 
                "optim": optim_type, 
                "criterion": criterion_type
            }
        }
    }

    if load_weight_threshold_matrix == "static":
        task_cfg_dict["skipping_strategy"]["skip_block_idx_list"] = skip_block_idx_list
    elif load_weight_threshold_matrix:
        task_cfg_dict["skipping_strategy"]["weight_threshold_matrix_path"] = weight_threshold_matrix_path

    task_cfg_dict = convert_numpy_type_to_native_type(task_cfg_dict)
    save_yaml(
        task_cfg_dict, 

        yaml_root_path = save_sample_root_path, 
        yaml_filename = "cfg.yaml"
    )

    # ---------= [AMP] =---------
    grad_scaler = GradScaler(enabled = True)

    torch.autograd.set_detect_anomaly(True)

    # ---------= [Unfreeze Attn Param] =---------
    pipeline.stu_unet.train()

    attn_param_name_list = []

    # unfreeze only params in attn layers
    for name, param in pipeline.stu_unet.named_parameters():
        if "attentions" in name:
            assert param.requires_grad == True

            param.data = param.data.to(torch.float32)

            attn_param_name_list.append(name)
        else:
            param.requires_grad = False
    
    # ---------= [Sample Scheduled] =---------
    def implement_batch(
        batch_idx: int, 

        generator: Union[torch.Generator, List[torch.Generator]]
    ):
        if (batch_idx != num_batch - 1) or (num_sample % batch_size == 0):
            true_batch_size = batch_size
        else:
            true_batch_size = num_sample % batch_size

        (
            batch_training_loss_list, 
            batch_inference_process_dict
        ) = pipeline.forward(
            prompt = [prompt] * true_batch_size, 
            negative_prompt = [negative_prompt] * true_batch_size, 

            height = height, width = width, 
            guidance_scale = guidance_scale, 
            num_inference_steps = num_inference_step, 

            num_images_per_prompt = 1, 
            
            generator = generator, 

            return_dict = False, 

            inference_step_minus_one = inference_step_minus_one, 

            save_inference_process_dict = save_inference_process_dict, 

            # attn reweighting
            # weight_matrix = 1.0, 
            weight_threshold_matrix = weight_threshold_matrix, 

            # fine-tuning
            optim = optim, 
            criterion = criterion, 
            grad_scaler = grad_scaler
        )
        
        # clean up
        torch.cuda.empty_cache()
        gc.collect()

        # batch_training_loss_list.shape = (num_inference_step, )
        return batch_training_loss_list, batch_inference_process_dict

    def implement_epoch(
        epoch_idx: int
    ):
        epoch_training_loss_list = np.zeros(
            shape = (num_inference_step)
        )

        for batch_idx in tqdm(
            range(num_batch), 
            desc = f"[Do Pruning, Epoch {epoch_idx}]"
        ): 
            generator = []

            true_sample_idx_st = batch_idx * batch_size
            if (batch_idx != num_batch - 1) or (num_sample % batch_size == 0):
                true_batch_size = batch_size
            else:
                true_batch_size = num_sample % batch_size

            for seed_idx in range(true_sample_idx_st, true_sample_idx_st + true_batch_size):
                generator.append(
                    get_generator(
                        seed = seed_list[seed_idx], 
                        device = device
                    )
                )

            (
                batch_training_loss_list, 
                batch_inference_process_dict
            ) = implement_batch(
                batch_idx = batch_idx, 

                generator = generator
            )
            batch_training_loss_list = np.asarray(batch_training_loss_list)

            epoch_training_loss_list += batch_training_loss_list

            # clean up
            del batch_training_loss_list
            torch.cuda.empty_cache()
            gc.collect()

            # goto `for batch_idx`
            pass

        epoch_training_loss_list /= num_batch

        # clean up
        del generator
        torch.cuda.empty_cache()
        gc.collect()

        # epoch_training_loss_list.shape = (num_inference_step, )
        return epoch_training_loss_list

    for epoch_idx in tqdm(
        range(num_epoch)
    ):
        # epoch_training_loss_list.shape = (num_inference_step, )
        epoch_training_loss_list = implement_epoch(epoch_idx = epoch_idx)

        logger(f"[Epoch {epoch_idx}]")
        logger(f"    epoch_training_loss_list: {epoch_training_loss_list}")

        # save training loss list
        if save_training_loss_list_in_process or (epoch_idx == num_epoch - 1):
            if ((epoch_idx + 1) % save_training_loss_list_epoch_interval == 0) \
                or (epoch_idx == num_epoch - 1):
                    tmp_dict = {
                        "training_loss_list": epoch_training_loss_list
                    }

                    tmp_dict = convert_numpy_type_to_native_type(tmp_dict)
                    save_yaml(
                        tmp_dict, 

                        yaml_root_path = save_training_loss_list_root_path, 
                        yaml_filename = f"{epoch_idx}.yaml"
                    )

        # save model ckpt
        state_dict = None
        if ((epoch_idx + 1) >= save_model_ckpt_start_epoch) \
            or (epoch_idx == num_epoch - 1):
            state_dict = get_selected_state_dict(
                model = pipeline.stu_unet, 
                selected_param_name_list = attn_param_name_list
            )

            save_model_state_dict(
                state_dict = state_dict, 

                ckpt_root_path = save_model_ckpt_root_path, 
                ckpt_filename = f"{epoch_idx}.pth"
            )

        # clean up
        del epoch_training_loss_list
        torch.cuda.empty_cache()
        gc.collect()

        # goto `for epoch_idx`
        pass
        
    # clean up
    torch.cuda.empty_cache()
    gc.collect()

    # `do_pruning_sd_family_implement()` done
    pass

def do_pruning_sd_family(
    cfg: DictConfig
):
    do_pruning_sd_family_implement(cfg)

    pass
