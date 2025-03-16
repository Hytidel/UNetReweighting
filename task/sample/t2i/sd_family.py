from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.cuda.amp import GradScaler, autocast

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
    load_pipeline, load_scheduler, 
    get_inference_step_minus_one, 
    img_latent_to_pil
)
from util.yaml_util import (
    load_yaml, save_yaml, 
    convert_numpy_type_to_native_type
)
from util.numpy_util import tsfm_to_2d_matrix, cal_matrix_ranking
from util.image_util import save_pil_as_png

from scheduled_model.sample.pipeline_stable_diffusion import register_pipeline_stable_diffusion
from importance_probe.importance_probe_util import get_weight_matrix
from ...util.prompt_util import (
    preprocess_prompt, 
    get_folder_name
)
from ..util.save_task_cfg_dict_util import get_true_task_cfg_dict
from ..util.tsfm_process_list_util import tsfm_pil_list


@torch.no_grad()
def sample_sd_family_implement(
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

    batch_size = get_true_value(cfg["task"]["task"]["batch_size"])
    
    logger(f"    batch_size: {batch_size}")

    logger(
        f"[Task] Loading finished. "
        "\n"
    )

    # ---------= [Pipeline & U-Net] =---------
    logger(f"[Pipeline & U-Net] Loading started. ")

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
    pipeline.enable_model_cpu_offload()

    logger(f"    pipeline: {type(pipeline)}")
    # logger(f"    pipeline: {pipeline}")

    sd_type = "sd"
    if "turbo" in pipeline_path:
        sd_type = "sd-turbo"

    logger(
        f"[Pipeline & U-Net] Loading finished. "
        "\n"
    )

    # ---------= [U-Net] =---------
    logger(f"[U-Net] Loading started. ")

    load_unet_ckpt = get_true_value(cfg["task"]["unet"]["load_unet_ckpt"])

    logger(f"    load_unet_ckpt: {load_unet_ckpt}")

    if load_unet_ckpt:
        unet_ckpt_path = get_true_value(cfg["task"]["unet"]["unet_ckpt_path"])

        logger(f"    unet_ckpt_path: {unet_ckpt_path}")

        load_model_ckpt(
            model = pipeline.unet, 
            ckpt_path = unet_ckpt_path, 
            device = device, 
            strict = False
        )

    logger(
        f"[U-Net] Loading finished. "
        "\n"
    )
    
    # ---------= [Task Seed 1] =---------
    logger(f"[Task Seed 1] Loading started. ")

    random_seed = get_true_value(cfg["task"]["task_seed"]["random_seed"])
    seed_range_l = get_true_value(cfg["task"]["task_seed"]["seed_range_l"])
    seed_range_r = get_true_value(cfg["task"]["task_seed"]["seed_range_r"])

    logger(f"    random_seed: {random_seed}")
    logger(f"    seed_range_l: {seed_range_l}")
    logger(f"    seed_range_r: {seed_range_r}")

    logger(
        f"[Task Seed 1] Loading finished. "
        "\n"
    )

    # ---------= [Weight Matrix] =---------
    logger(f"[Weight Matrix] Loading started. ")

    num_attn_block = pipeline.unet.num_attn_block
    
    logger(f"    num_attn_block: {num_attn_block}")

    load_weight_matrix = get_true_value(cfg["task"]["weight_matrix"]["load_weight_matrix"])
    default_weight_matrix = get_true_value(cfg["task"]["weight_matrix"]["default_weight_matrix"])

    logger(f"    load_weight_matrix: {load_weight_matrix}")
    logger(f"    default_weight_matrix: {default_weight_matrix}")

    weight_matrix = default_weight_matrix
    weight_matrix_name = "default"

    if load_weight_matrix == "static":
        block_idx_list = get_true_value(cfg["task"]["weight_matrix"]["static_weight_matrix"]["block_idx_list"])
        block_idx_list = np.asarray(block_idx_list)
        block_weight = get_true_value(cfg["task"]["weight_matrix"]["static_weight_matrix"]["block_weight"])
        block_weight = np.asarray(block_weight)

        logger(f"    block_idx_list: {block_idx_list}")
        logger(f"    block_weight: {block_weight}")

        weight_matrix = np.ones(
            shape = (num_attn_block, ), 
            dtype = np.float32
        )
        weight_matrix[block_idx_list] = block_weight

        if block_idx_list.ndim == 0:
            block_idx_list_str_list = [str(block_idx_list)]
        else:
            block_idx_list_str_list = [str(block_idx) for block_idx in block_idx_list]
        block_idx_list_str = ",".join(block_idx_list_str_list)
        weight_matrix_name = f"blk-{block_idx_list_str}_weight-{block_weight}"
    elif load_weight_matrix:
        load_weight_matrix_path = get_true_value(cfg["task"]["weight_matrix"]["load_weight_matrix_path"])
        load_weight_matrix_path = Path(load_weight_matrix_path)

        logger(f"    load_weight_matrix_path: {load_weight_matrix_path}")

        weight_matrix = load_yaml(load_weight_matrix_path)["importance_weight_matrix"]
        if not isinstance(weight_matrix, float):
            weight_matrix = np.asarray(weight_matrix)
        
        weight_matrix_name = load_weight_matrix_path.stem

    weight_matrix = get_weight_matrix(
        weight_matrix = weight_matrix, 
        num_inference_step = num_inference_step, 
        num_attn_block = num_attn_block
    )

    logger(f"    weight_matrix: {weight_matrix}")

    logger(
        f"[Weight Matrix] Loading finished. "
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
    if random_seed:
        save_sample_root_path = save_sample_root_path / sd_type / f"step-{num_inference_step}_seed-{seed_range_l}"
    else:
        save_sample_root_path = save_sample_root_path / sd_type / f"step-{num_inference_step}_seed-{seed}"

    category_name = get_true_value(cfg["task"]["save_sample"]["category_name"])
    if not is_none(category_name):
        save_sample_root_path = save_sample_root_path / category_name

    logger(f"    save_sample_root_path: {save_sample_root_path}")
    logger(f"    category_name: {category_name}")

    num_sample_per_prompt = get_true_value(cfg["task"]["save_sample"]["num_sample_per_prompt"])
    sample_start_idx = get_true_value(cfg["task"]["save_sample"]["sample_start_idx"])
    save_process_png = get_true_value(cfg["task"]["save_sample"]["save_process_png"])

    logger(f"    num_sample_per_prompt: {num_sample_per_prompt}")
    logger(f"    sample_start_idx: {sample_start_idx}")
    logger(f"    save_process_png: {save_process_png}")

    logger(
        f"[Save Sample] Loading finished. "
        "\n"
    )

    # ---------= [Preprocess Prompt List] =---------
    prompt_tuple_list = [(0, prompt)] * num_sample_per_prompt

    folder_name_list = []  # useless

    prompt = preprocess_prompt(prompt)
    folder_name = get_folder_name(
        prompt = prompt, 
        used_folder_name_list = folder_name_list
    )

    num_sample = num_sample_per_prompt

    num_batch = (num_sample + batch_size - 1) // batch_size

    logger(f"    num_sample: {num_sample}")
    logger(f"    num_batch: {num_batch}")

    # ---------= [Task Seed 2] =---------
    logger(f"[Task Seed 2] Loading started. ")

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
        f"[Task Seed 2] Loading finished. "
        "\n"
    )

    # ---------= [Save Task Config] =---------
    save_inference_process_dict = {
        "noise_pred": False, 
        "latent": False, 
        "pil": True
    }

    task_cfg_dict = {
        "num_sample_per_prompt": num_sample_per_prompt, 
        "batch_size": batch_size, 

        "height": height, 
        "width": width, 

        "num_inference_step": num_inference_step, 

        "guidance_scale": guidance_scale, 

        "negative_prompt": negative_prompt
    }

    # ---------= [All Components Loaded] =---------
    logger(
        f"All components loaded. "
        "\n"
    )
    
    # ---------= [Sample Scheduled] =---------
    def implement_batch(
        batch_idx: int, 
        generator: Union[torch.Generator, List[torch.Generator]]
    ):
        # batch_prompt_tuple_list.shape = (B, )
        prompt_idx_l = batch_idx * batch_size
        prompt_idx_r = min((batch_idx + 1) * batch_size, num_sample)
        batch_prompt_tuple_list = prompt_tuple_list[prompt_idx_l: prompt_idx_r]

        logger(f"[Batch {batch_idx}] batch_prompt_tuple_list: {batch_prompt_tuple_list}")
        
        # batch_prompt_list.shape = (B, )
        batch_prompt_list = [
            prompt_tuple[1] \
                for prompt_tuple in batch_prompt_tuple_list
        ]

        if (batch_idx != num_batch - 1) or (num_sample % batch_size == 0):
            true_batch_size = batch_size
        else:
            true_batch_size = num_sample % batch_size

        inference_process_dict = pipeline.forward(
            prompt = batch_prompt_list, 
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
            weight_matrix = weight_matrix, 
            weight_threshold_matrix = weight_threshold_matrix
        )
        
        if (batch_idx != num_batch - 1) or (num_sample % batch_size == 0):
            true_batch_size = batch_size
        else:
            true_batch_size = num_sample % batch_size

        # batch_pil_list.shape = (num_inference_step + 1, true_batch_size)
        batch_pil_list = inference_process_dict["pil"]
        # process_pil_list_list.shape = (true_batch_size, num_inference_step + 1)
        process_pil_list_list = tsfm_pil_list(
            pil_list = batch_pil_list, 
            num_inference_step = num_inference_step, 
            batch_size = true_batch_size
        )

        with cf.ThreadPoolExecutor(
            max_workers = concurrent_max_worker
        ) as executor:
            for sample_idx in tqdm(
                range(true_batch_size), 

                desc = f"[Save Sample]"
            ):
                true_sample_idx = batch_idx * batch_size + sample_idx

                prompt_idx = batch_prompt_tuple_list[sample_idx][0]
                folder_name = folder_name_list[prompt_idx]

                # save task cfg at the first sample of a prompt
                if true_sample_idx % num_sample_per_prompt == 0:
                    save_task_cfg_dict_root_path = save_sample_root_path / folder_name
                    tmp_task_cfg_dict = get_true_task_cfg_dict(
                        task_cfg_dict = task_cfg_dict, 
                        prompt = prompt, 
                        seed_list = seed_list[true_sample_idx: min(true_sample_idx + num_sample_per_prompt, num_sample)], 
                        save_task_cfg_dict_root_path = save_task_cfg_dict_root_path
                    )
                    
                    executor.submit(
                        save_yaml, 

                        tmp_task_cfg_dict, 
                        save_task_cfg_dict_root_path, 
                        "cfg.yaml"
                    )

                save_png_root_path \
                    = save_sample_root_path / folder_name / "png" / f"{weight_matrix_name}_{weight_threshold_matrix_name}"
                
                # save png
                for step_idx in range(
                    0 if save_process_png else num_inference_step, 
                    num_inference_step + 1
                ):
                    tmp_pil = process_pil_list_list[sample_idx][step_idx]
                    pil_name = f"{sample_start_idx + true_sample_idx % num_sample_per_prompt}"
                    if step_idx < num_inference_step:
                        pil_name = f"{pil_name}_{step_idx}"

                    executor.submit(
                        save_pil_as_png, 

                        tmp_pil, 
                        save_png_root_path, 
                        f"{pil_name}.png"
                    )

                # goto `for sample_idx`
                pass

        # clean up
        del batch_pil_list, process_pil_list_list
        torch.cuda.empty_cache()
        gc.collect()

        # `implement_batch()` done
        pass

    def implement_epoch(
        epoch_idx: int
    ):
        for batch_idx in tqdm(
            range(num_batch), 
            desc = f"[Sampling SD Family]"
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
            
            implement_batch(
                batch_idx = batch_idx, 

                generator = generator
            )

            # clean up
            del generator
            torch.cuda.empty_cache()
            gc.collect()

            # goto `for batch_idx`
            pass

        # clean up
        torch.cuda.empty_cache()
        gc.collect()

        # `implement_epoch()` done
        pass

    num_epoch = 1
    for epoch_idx in tqdm(
        range(num_epoch)
    ):
        implement_epoch(epoch_idx = epoch_idx)

    # clean up
    torch.cuda.empty_cache()
    gc.collect()

    # `sample_sd_family_implement()` done
    pass

def sample_sd_family(
    cfg: DictConfig
):
    sample_sd_family_implement(cfg)

    pass
