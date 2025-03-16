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

import shlex
import subprocess

from tqdm.auto import tqdm

from matplotlib import pyplot as plt
from PIL import Image

import time

import copy

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
from util.json_util import load_json

from ..util.save_sample_util import is_data_exist
from ...util.prompt_util import (
    preprocess_prompt, 
    get_folder_name
)


@torch.no_grad()
def do_importance_probe_run_sd_family_implement(
    cfg: DictConfig
):
    # ---------= [Basic Global Variables] =---------
    exp_name = get_global_variable("exp_name")
    start_time = get_global_variable("start_time")
    device = get_global_variable("device")
    seed = get_global_variable("seed")
    exp_time_str = f"{exp_name}_{start_time}"

    concurrent_max_worker = get_global_variable("concurrent_max_worker")

    # ---------= [Pipeline] =---------
    logger(f"[Pipeline] Loading started. ")

    sd_type = get_true_value(cfg["task"]["sd_type"])

    logger(f"    sd_type: {sd_type}")

    logger(
        f"[Pipeline] Loading finished. "
        "\n"
    )

    # ---------= [Task] =---------
    logger(f"[Task] Loading started. ")

    height = get_true_value(cfg["task"]["task"]["height"])
    width = get_true_value(cfg["task"]["task"]["width"])
    num_inference_step = get_true_value(cfg["task"]["task"]["num_inference_step"])
    guidance_scale = get_true_value(cfg["task"]["task"]["guidance_scale"])
    negative_prompt = get_true_value(cfg["task"]["task"]["negative_prompt"])

    logger(f"    (height, width): ({height}, {width})")
    logger(f"    num_inference_step: {num_inference_step}")
    logger(f"    guidance_scale: {guidance_scale}")
    logger(f"    negative_prompt: {negative_prompt}")

    logger(
        f"[Task] Loading finished. "
        "\n"
    )

    # ---------= [Prompt] =---------
    logger(f"[Prompt] Loading started. ")

    prompt_json_path = get_true_value(cfg["task"]["prompt"]["prompt_json_path"])
    prompt_json_path = Path(prompt_json_path)
    prompt_batch_size = get_true_value(cfg["task"]["prompt"]["prompt_batch_size"])
    prompt_set_size = get_true_value(cfg["task"]["prompt"]["prompt_set_size"])

    logger(f"    prompt_json_path: {prompt_json_path}")
    logger(f"    prompt_batch_size: {prompt_batch_size}")
    logger(f"    prompt_set_size: {prompt_set_size}")

    category_name = prompt_json_path.stem

    logger(f"    category_name: {category_name}")

    logger(
        f"[Prompt] Loading finished. "
        "\n"
    )

    # ---------= [Save Sample] =---------
    logger(f"[Save Sample] Loading started. ")

    save_sample_root_path = get_true_value(cfg["task"]["save_sample"]["save_sample_root_path"])
    save_sample_root_path = Path(save_sample_root_path)
    category_root_path = save_sample_root_path / category_name

    logger(f"    save_sample_root_path: {save_sample_root_path}")

    num_sample_per_prompt = get_true_value(cfg["task"]["save_sample"]["num_sample_per_prompt"])
    batch_size = get_true_value(cfg["task"]["save_sample"]["batch_size"])
    
    logger(f"    num_sample_per_prompt: {num_sample_per_prompt}")
    logger(f"    batch_size: {batch_size}")

    logger(
        f"[Save Sample] Loading finished. "
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
        else:
            seed = get_timestamp(to_int = True)

    logger(
        f"[Task Seed] Loading finished. "
        "\n"
    )

    # ---------= [Importance Probe] =---------
    logger(f"[Importance Probe] Loading started. ")

    num_round = get_true_value(cfg["task"]["importance_probe"]["num_round"])
    num_epoch = get_true_value(cfg["task"]["importance_probe"]["num_epoch"])
    update_chart_in_process = get_true_value(cfg["task"]["importance_probe"]["chart"]["update_chart_in_process"])
    num_row = get_true_value(cfg["task"]["importance_probe"]["chart"]["num_row"])
    num_col = get_true_value(cfg["task"]["importance_probe"]["chart"]["num_col"])

    logger(f"    num_round: {num_round}")
    logger(f"    num_epoch: {num_epoch}")
    logger(f"    update_chart_in_process: {update_chart_in_process}")
    logger(f"    num_row: {num_row}")
    logger(f"    num_col: {num_col}")

    logger(
        f"[Importance Probe] Loading finished. "
        "\n"
    )

    # ---------= [Preprocess Prompt List] =---------
    prompt_list = load_json(prompt_json_path)

    random.shuffle(prompt_list)

    prompt_list = prompt_list[: prompt_set_size]

    folder_name_list = []

    prompt_tuple_list = []  # (prompt_idx, prompt)

    for prompt_idx, prompt in enumerate(
        tqdm(
            prompt_list, 

            desc = f"[Check Data Existence]"
        )
    ):
        folder_name = get_folder_name(
            prompt = prompt, 
            used_folder_name_list = folder_name_list
        )

        save_weight_threshold_matrix_list_path \
            = category_root_path / sd_type / f"step-{num_inference_step}_seed-{seed}" / folder_name / "weight_threshold_matrix_list"
        
        if not is_data_exist(
            save_weight_threshold_matrix_list_path = save_weight_threshold_matrix_list_path, 

            num_round = num_round
        ):
            prompt_tuple_list.append((prompt_idx, prompt))

    num_prompt = len(prompt_tuple_list)
    tot_num_prompt = len(prompt_list)
    
    logger(f"    num_prompt / tot_num_prompt: {num_prompt} / {tot_num_prompt}")

    num_batch = (num_prompt + prompt_batch_size - 1) // prompt_batch_size

    logger(f"    num_batch: {num_batch}")

    # ---------= [Save Task Config] =---------
    task_cfg_dict = {
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

    # ---------= [Do Importance Probe] =---------
    negative_prompt = shlex.quote(negative_prompt)

    def implement_batch(
        batch_idx: int
    ):
        batch_prompt_tuple_list = prompt_tuple_list[
            batch_idx * prompt_batch_size: 
            min((batch_idx + 1) * prompt_batch_size, num_prompt)
        ]

        logger(f"[Batch {batch_idx}]")
        logger(f"    batch_prompt_tuple_list: {batch_prompt_tuple_list}")

        with cf.ThreadPoolExecutor(
            max_workers = concurrent_max_worker
        ) as executor:
            for (prompt_idx, prompt) in batch_prompt_tuple_list:
                folder_name = folder_name_list[prompt_idx]
                
                logger(f"    (prompt_idx, prompt): ({prompt_idx}, {prompt})")

                prompt = shlex.quote(prompt)
                
                cmd = [
                    "python", 
                    "main.py", 

                    f"pipeline={sd_type}", 
                    f"task=do_importance_probe/t2i/{sd_type}/template", 

                    f"task.task.prompt={prompt}", 
                    f"task.task.negative_prompt={negative_prompt}", 
                    f"task.task.height={height}", 
                    f"task.task.width={width}", 
                    f"task.task.num_inference_step={num_inference_step}", 
                    f"task.task.guidance_scale={guidance_scale}", 

                    f"task.save_sample.save_sample_root_path={category_root_path}", 
                    f"task.save_sample.num_sample_per_prompt={num_sample_per_prompt}", 
                    f"task.save_sample.batch_size={batch_size}", 
                    f"task.save_sample.prompt_idx={prompt_idx}", 
                    f"task.save_sample.folder_name={folder_name}", 

                    f"task.task_seed.random_seed={random_seed}", 
                    f"task.task_seed.seed_range_l={seed_range_l}", 
                    f"task.task_seed.seed_range_r={seed_range_r}", 

                    f"task.importance_probe.num_round={num_round}", 
                    f"task.importance_probe.num_epoch={num_epoch}", 
                    f"task.importance_probe.chart.update_chart_in_process={update_chart_in_process}", 
                    f"task.importance_probe.chart.num_row={num_row}", 
                    f"task.importance_probe.chart.num_col={num_col}"
                ]

                logger(f"    cmd: {cmd}")

                # subprocess.run(cmd)

                executor.submit(
                    subprocess.run, 

                    cmd
                )

    def implement_epoch(
        epoch_idx: int
    ):
        for batch_idx in range(num_batch):
            implement_batch(batch_idx = batch_idx)

    num_epoch_ = (num_prompt > 0)
    for epoch_idx in tqdm(
        range(num_epoch_)
    ):
        implement_epoch(epoch_idx = epoch_idx)

    # clean up
    torch.cuda.empty_cache()
    gc.collect()

    # `do_importance_probe_run_sd_family_implement()` done
    pass

def do_importance_probe_run_sd_family(
    cfg: DictConfig
):
    do_importance_probe_run_sd_family_implement(cfg)

    pass
