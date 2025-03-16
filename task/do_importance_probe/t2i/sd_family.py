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

from scheduled_model.importance_probe.pipeline_stable_diffusion import register_pipeline_stable_diffusion
from importance_probe.weight_scaling_scheduler import WeightScalingScheduler
from importance_probe.weight_threshold_scheduler import WeightThresholdScheduler
from importance_probe.importance_probe_util import (
    get_weight_matrix, 
    get_weight_max_bias_list, 
    get_noise_pred_loss_threshold_list, 
    save_merged_inference_step_chart, 
    cal_fitness
)
from ...util.prompt_util import (
    preprocess_prompt, 
    get_folder_name
)


@torch.no_grad()
def do_importance_probe_sd_family_implement(
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

    # ---------= [Save Sample] =---------
    logger(f"[Save Sample] Loading started. ")

    num_sample_per_prompt = get_true_value(cfg["task"]["save_sample"]["num_sample_per_prompt"])
    batch_size = get_true_value(cfg["task"]["save_sample"]["batch_size"])
    
    logger(f"    num_sample_per_prompt: {num_sample_per_prompt}")
    logger(f"    batch_size: {batch_size}")

    prompt_idx = get_true_value(cfg["task"]["save_sample"]["prompt_idx"])
    if is_none(prompt_idx):
        prompt_idx = 0

    logger(f"    prompt_idx: {prompt_idx}")

    folder_name = get_true_value(cfg["task"]["save_sample"]["folder_name"])

    prompt_tuple_list = [(prompt_idx, prompt)] * num_sample_per_prompt

    if is_none(folder_name):
        folder_name_list = []  # useless

        prompt = preprocess_prompt(prompt)
        folder_name = get_folder_name(
            prompt = prompt, 
            used_folder_name_list = folder_name_list
        )

    logger(f"    folder_name: {folder_name}")

    save_sample_root_path = get_true_value(cfg["task"]["save_sample"]["save_sample_root_path"])
    save_sample_root_path = Path(save_sample_root_path) / sd_type
    if random_seed:
        save_sample_root_path = save_sample_root_path / f"step-{num_inference_step}_seed-{seed_range_l}"
    else:
        save_sample_root_path = save_sample_root_path / f"step-{num_inference_step}_seed-{seed}"
    save_sample_root_path = save_sample_root_path / folder_name
    
    logger(f"    save_sample_root_path: {save_sample_root_path}")

    num_sample = num_sample_per_prompt
    num_batch = (num_sample + batch_size - 1) // batch_size

    logger(f"    num_sample: {num_sample}")
    logger(f"    num_batch: {num_batch}")

    logger(
        f"[Save Sample] Loading finished. "
        "\n"
    )

    # ---------= [Importance Probe] =---------
    logger(f"[Importance Probe] Loading started. ")

    num_round = get_true_value(cfg["task"]["importance_probe"]["num_round"])
    num_epoch = get_true_value(cfg["task"]["importance_probe"]["num_epoch"])

    logger(f"    num_round: {num_round}")
    logger(f"    num_epoch: {num_epoch}")

    # ---------= [Noise Pred Loss Threshold] =---------
    noise_pred_loss_threshold_strategy = get_true_value(cfg["task"]["importance_probe"]["noise_pred_loss_threshold"]["threshold_strategy"])
    noise_pred_loss_threshold_list = None
    if noise_pred_loss_threshold_strategy == "list":
        noise_pred_loss_threshold_list = get_true_value(cfg["task"]["importance_probe"]["noise_pred_loss_threshold"]["threshold_list"])
    elif noise_pred_loss_threshold_strategy == "linear":
        noise_pred_loss_threshold_st = get_true_value(cfg["task"]["importance_probe"]["noise_pred_loss_threshold"]["threshold_st"])
        noise_pred_loss_threshold_ed = get_true_value(cfg["task"]["importance_probe"]["noise_pred_loss_threshold"]["threshold_ed"])
    else:
        raise ValueError(
            f"Unsupported `noise_pred_loss_threshold_strategy`, got `{noise_pred_loss_threshold_strategy}`. "
        )

    noise_pred_loss_threshold_list = get_noise_pred_loss_threshold_list(
        noise_pred_loss_threshold_strategy = noise_pred_loss_threshold_strategy, 
        noise_pred_loss_threshold_st = noise_pred_loss_threshold_st, 
        noise_pred_loss_threshold_ed = noise_pred_loss_threshold_ed, 
        noise_pred_loss_threshold_list = noise_pred_loss_threshold_list, 
        num_inference_step = num_inference_step
    )

    logger(f"    noise_pred_loss_threshold_strategy: {noise_pred_loss_threshold_strategy}")
    logger(f"    noise_pred_loss_threshold_list: {noise_pred_loss_threshold_list}")

    # ---------= [Weight Max Bias Strategy] =---------
    weight_max_bias_strategy = get_true_value(cfg["task"]["importance_probe"]["weight_max_bias_strategy"]["bias_strategy"])
    weight_max_bias_list = None
    if weight_max_bias_strategy == "list":
        weight_max_bias_list = get_true_value(cfg["task"]["importance_probe"]["weight_max_bias_strategy"]["bias_list"])
    elif weight_max_bias_strategy == "linear":
        weight_max_bias_st = get_true_value(cfg["task"]["importance_probe"]["weight_max_bias_strategy"]["bias_st"])
        weight_max_bias_ed = get_true_value(cfg["task"]["importance_probe"]["weight_max_bias_strategy"]["bias_ed"])
    else:
        raise ValueError(
            f"Unsupported `weight_max_bias_strategy`, got `{weight_max_bias_strategy}`. "
        )

    weight_max_bias_list = get_weight_max_bias_list(
        weight_max_bias_strategy = weight_max_bias_strategy, 
        weight_max_bias_st = weight_max_bias_st, 
        weight_max_bias_ed = weight_max_bias_ed, 
        weight_max_bias_list = weight_max_bias_list, 
        num_inference_step = num_inference_step
    )

    logger(f"    weight_max_bias_strategy: {weight_max_bias_strategy}")
    logger(f"    weight_max_bias_list: {weight_max_bias_list}")

    # ---------= [Weight Threshold Update Strategy] =---------
    weight_threshold_update_strategy = get_true_value(cfg["task"]["importance_probe"]["weight_threshold_update_strategy"]["update_strategy"])
    weight_threshold_update_eps = get_true_value(cfg["task"]["importance_probe"]["weight_threshold_update_strategy"]["update_eps"])

    logger(f"    weight_threshold_update_strategy: {weight_threshold_update_strategy}")
    logger(f"    weight_threshold_update_eps: {weight_threshold_update_eps}")

    sample_accepted_prob_dict = None
    sample_rejected_prob_dict = None

    if weight_threshold_update_strategy in ["hard", "soft", "probability"]:
        if weight_threshold_update_strategy == "probability":
            sample_accepted_prob_dict = get_true_value(cfg["task"]["importance_probe"]["weight_threshold_update_strategy"]["sample_accepted"])
            sample_rejected_prob_dict = get_true_value(cfg["task"]["importance_probe"]["weight_threshold_update_strategy"]["sample_rejected"])
            
            logger(f"    sample_accepted_prob_dict: {sample_accepted_prob_dict}")
            logger(f"    sample_rejected_prob_dict: {sample_rejected_prob_dict}")
    else:
        raise ValueError(
            f"Unsupported `weight_threshold_update_strategy`, got `{weight_threshold_update_strategy}`. "
        )

    # ---------= [Weight Threshold Scheduler] =---------
    num_importance_probe_target = pipeline.unet.num_attn_block
    weight_threshold_scheduler = WeightThresholdScheduler(
        num_inference_step = num_inference_step, 
        num_weight_threshold = num_importance_probe_target, 
        init_weight_threshold = 0.0, 
        min_weight_threshold = 0.0, max_weight_threshold = 1.0
    )

    logger(f"    num_importance_probe_target: {num_importance_probe_target}")
    logger(f"    weight_threshold_scheduler: {weight_threshold_scheduler}")

    update_weight_threshold_matrix_partial = partial(
        weight_threshold_scheduler.update_weight_threshold_matrix, 

        update_strategy = weight_threshold_update_strategy, 
        weight_eps = weight_threshold_update_eps, 

        sample_accepted_prob_dict = sample_accepted_prob_dict, 
        sample_rejected_prob_dict = sample_rejected_prob_dict, 

        # used for `update_strategy = "soft"` 
        num_epoch = num_epoch
    )

    # ---------= [Save Weight Threshold Matrix] =---------
    save_weight_threshold_matrix_list_root_path = save_sample_root_path / "weight_threshold_matrix_list"
    save_last_weight_threshold_matrix_per_epoch = get_true_value(cfg["task"]["importance_probe"]["weight_threshold_matrix"]["save_last_weight_threshold_matrix_per_epoch"])
    save_history_weight_threshold_matrix_list = get_true_value(cfg["task"]["importance_probe"]["weight_threshold_matrix"]["save_history_weight_threshold_matrix_list"])

    logger(f"    save_weight_threshold_matrix_list_root_path: {save_weight_threshold_matrix_list_root_path}")
    logger(f"    save_last_weight_threshold_matrix_per_epoch: {save_last_weight_threshold_matrix_per_epoch}")
    logger(f"    save_history_weight_threshold_matrix_list: {save_history_weight_threshold_matrix_list}")
    
    # ---------= [Save Chart] =---------
    update_chart_in_process = get_true_value(cfg["task"]["importance_probe"]["chart"]["update_chart_in_process"])
    update_chart_epoch_interval = get_true_value(cfg["task"]["importance_probe"]["chart"]["update_chart_epoch_interval"])
    chart_root_path = save_sample_root_path / "chart"
    figsize_per_chart = get_true_value(cfg["task"]["importance_probe"]["chart"]["figsize_per_chart"])
    figsize_per_chart = tuple(figsize_per_chart)
    marker_list = get_true_value(cfg["task"]["importance_probe"]["chart"]["marker_list"])
    num_row = get_true_value(cfg["task"]["importance_probe"]["chart"]["num_row"])
    num_col = get_true_value(cfg["task"]["importance_probe"]["chart"]["num_col"])

    logger(f"    update_chart_in_process: {update_chart_in_process}")
    logger(f"    update_chart_epoch_interval: {update_chart_epoch_interval}")
    logger(f"    chart_root_path: {chart_root_path}")
    logger(f"    figsize_per_chart: {figsize_per_chart}")
    logger(f"    marker_list: {marker_list}")
    logger(f"    (num_row, num_col): ({num_row}, {num_col})")

    # ---------= [Fitness Function] =---------
    # initialize `init_energy` only once
    init_energy = WeightScalingScheduler(
        num_weight = num_importance_probe_target, 
        init_weight = 1.0, 
        min_weight = 0.0, max_weight = 1.0, 
        energy_func = "quadratic_sum"
    ).energy

    logger(f"    init_energy: {init_energy}")

    cal_fitness_partial = partial(
        cal_fitness, 

        init_energy = init_energy, 
        unet = pipeline.unet
    )

    logger(
        f"[Importance Probe] Loading finished. "
        "\n"
    )

    # ---------= [Task Seed 2] =---------
    logger(f"[Task Seed 2] Loading started. ")

    if random_seed:
        if seed_range_l == seed_range_r:
            seed = seed_range_l
            random_seed = False

            seed_list = [seed + i % num_sample_per_prompt for i in range(num_sample)]
        else:
            seed = get_timestamp(to_int = True)
            seed_list = [
                (seed + i % num_sample_per_prompt) % (seed_range_r - seed_range_l + 1) + seed_range_l \
                    for i in range(num_sample)
            ]
    else:
        seed_list = [seed + i % num_sample_per_prompt for i in range(num_sample)]

    logger(
        f"[Task Seed 2] Loading finished. "
        "\n"
    )

    # ---------= [Save Task Config] =---------
    task_cfg_dict = {
        "num_sample_per_prompt": num_sample_per_prompt, 
        "batch_size": batch_size, 

        "height": height, 
        "width": width, 

        "num_inference_step": num_inference_step, 

        "guidance_scale": guidance_scale, 

        "negative_prompt": negative_prompt
    }

    with cf.ThreadPoolExecutor(
        max_workers = concurrent_max_worker
    ) as executor:
        task_cfg_dict["prompt"] = prompt
        task_cfg_dict["seed_list"] = seed_list
        
        executor.submit(
            save_yaml, 

            task_cfg_dict, 
            save_sample_root_path, 
            "cfg.yaml"
        )

    # ---------= [All Components Loaded] =---------
    logger(
        f"All components loaded. "
        "\n"
    )

    # ---------= [Do Importance Probe] =---------
    history_best_weight_matrix = None

    def implement_batch(
        round_idx: int, 
        epoch_idx: int, 
        batch_idx: int, 
        generator: Union[torch.Generator, List[torch.Generator]]
    ):
        nonlocal history_best_weight_matrix

        # batch_prompt_tuple_list.shape = (B, )
        prompt_idx_l = batch_idx * batch_size
        prompt_idx_r = min((batch_idx + 1) * batch_size, num_sample)
        batch_prompt_tuple_list = prompt_tuple_list[prompt_idx_l: prompt_idx_r]

        logger(f"[Round {round_idx}, Epoch {epoch_idx}, Batch {batch_idx}]")
        logger(f"    batch_prompt_tuple_list: {batch_prompt_tuple_list}")
        
        # batch_prompt_list.shape = (B, )
        batch_prompt_list = [
            prompt_tuple[1] \
                for prompt_tuple in batch_prompt_tuple_list
        ]

        if (batch_idx != num_batch - 1) or (num_sample % batch_size == 0):
            true_batch_size = batch_size
        else:
            true_batch_size = num_sample % batch_size

        (
            weight_threshold_matrix, 
            inference_step_sample_accepted_mask
        ) = pipeline.forward(
            prompt = batch_prompt_list, 
            negative_prompt = [negative_prompt] * true_batch_size, 

            height = height, width = width, 
            guidance_scale = guidance_scale, 
            num_inference_steps = num_inference_step, 

            num_images_per_prompt = 1, 
            
            generator = generator, 

            return_dict = False, 

            inference_step_minus_one = inference_step_minus_one, 

            # attn reweighting
            weight_threshold_scheduler = weight_threshold_scheduler, 
            history_best_weight_matrix = history_best_weight_matrix, 
            cal_fitness_partial = cal_fitness_partial, 
            noise_pred_loss_threshold_list = noise_pred_loss_threshold_list, 
            weight_max_bias_list = weight_max_bias_list
        )
        
        # update weight threshold matrix
        update_weight_threshold_matrix_partial(
            weight_threshold_matrix = weight_threshold_matrix, 
            inference_step_sample_accepted_mask = inference_step_sample_accepted_mask, 

            # used for `update_strategy = "soft"` 
            epoch_idx = epoch_idx
        )
        
        # clean up
        del weight_threshold_matrix
        del inference_step_sample_accepted_mask
        torch.cuda.empty_cache()
        gc.collect()

        # `implement_batch()` done
        pass

    def implement_epoch(
        round_idx: int, 
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
                round_idx = round_idx, 
                epoch_idx = epoch_idx, 
                batch_idx = batch_idx, 

                generator = generator
            )

            # clean up
            del generator
            torch.cuda.empty_cache()
            gc.collect()

            # goto `for batch_idx`
            pass

        # save history
        weight_threshold_scheduler._save_history_weight_threshold_matrix()

        # clean up
        torch.cuda.empty_cache()
        gc.collect()

        # `implement_epoch()` done
        pass

    def implement_round(
        round_idx: int
    ):
        nonlocal history_best_weight_matrix
        history_best_weight_matrix = np.random.uniform(
            low = 0.995, high = 1.0, 
            size = (num_inference_step, num_importance_probe_target)
        )

        weight_threshold_scheduler.init(init_weight_threshold = 0.0)

        for epoch_idx in tqdm(
            range(num_epoch)
        ):
            implement_epoch(
                round_idx = round_idx, 
                epoch_idx = epoch_idx
            )

            weight_threshold_matrix = weight_threshold_scheduler.get_weight_threshold_matrix()

            logger(f"[Round {round_idx}, Epoch {epoch_idx}]")
            logger(f"    weight_threshold_matrix: {weight_threshold_matrix}")

            with cf.ThreadPoolExecutor(
                max_workers = concurrent_max_worker
            ) as executor:
                # save `weight_threshold_scheduler.last_weight_threshold_matrix` ckpt
                if save_last_weight_threshold_matrix_per_epoch:
                    save_weight_threshold_matrix_list_root_path = save_sample_root_path / "weight_threshold_matrix_list" / f"round-{round_idx}"
                    
                    executor.submit(
                        weight_threshold_scheduler._save_last_weight_threshold_matrix_as_yaml, 

                        save_weight_threshold_matrix_list_root_path, 
                        f"last_weight_threshold_matrix-{epoch_idx}.yaml"
                    )

                # save chart
                if update_chart_in_process or (epoch_idx == num_epoch - 1):
                    if (epoch_idx % update_chart_epoch_interval == 0) \
                        or (epoch_idx == num_epoch - 1):
                        chart_root_path = save_sample_root_path / "chart"

                        executor.submit(
                            save_merged_inference_step_chart, 

                            weight_threshold_scheduler, 
                            chart_root_path, 
                            f"round-{round_idx}.png", 
                            figsize_per_chart, 
                            marker_list, 
                            num_row, num_col
                        )

        # save `weight_threshold_scheduler.history_weight_threshold_matrix_list` ckpt
        if save_history_weight_threshold_matrix_list:
            save_weight_threshold_matrix_list_root_path = save_sample_root_path / "weight_threshold_matrix_list" / f"round-{round_idx}"
            weight_threshold_scheduler._save_history_weight_threshold_matrix_list_as_yaml(
                yaml_root_path = save_weight_threshold_matrix_list_root_path, 
                yaml_filename = f"history_weight_threshold_matrix_list.yaml"
            )

    for round_idx in range(num_round):
        implement_round(round_idx = round_idx)

    # clean up
    torch.cuda.empty_cache()
    gc.collect()

    # `do_importance_probe_sd_family_implement()` done
    pass

def do_importance_probe_sd_family(
    cfg: DictConfig
):
    do_importance_probe_sd_family_implement(cfg)

    pass
