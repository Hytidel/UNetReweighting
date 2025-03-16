from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

from pathlib import Path

import numpy as np

import gc

from util.basic_util import pause, get_global_variable, is_none, get_true_value
from util.yaml_util import load_yaml
from util.plot_util import (
    save_plot, 
    get_scatter
)


# @torch.no_grad()
def scatter_fid_lpips_implement(
    cfg: DictConfig
):
    # ---------= [Basic Global Variables] =---------
    exp_name = get_global_variable("exp_name")
    start_time = get_global_variable("start_time")
    device = get_global_variable("device")
    seed = get_global_variable("seed")
    exp_time_str = f"{exp_name}_{start_time}"

    concurrent_max_worker = get_global_variable("concurrent_max_worker")

    # ---------= [SD Type] =---------
    logger(f"[SD Type] Loading started. ")

    sd_type = get_true_value(cfg["task"]["sd_type"])

    logger(f"    sd_type: {sd_type}")

    logger(
        f"[SD Type] Loading finished. "
        "\n"
    )

    # ---------= [Data] =---------
    logger(f"[Data] Loading started. ")

    pruning_res_dict_path = get_true_value(cfg["task"]["data"]["pruning_res_dict_path"])
    pruning_res_dict_path = Path(pruning_res_dict_path)

    logger(f"    pruning_res_dict_path: {pruning_res_dict_path}")

    split = get_true_value(cfg["task"]["data"]["split"])
    num_skipped_block = get_true_value(cfg["task"]["data"]["num_skipped_block"])

    logger(f"    split: {split}")
    logger(f"    num_skipped_block: {num_skipped_block}")

    duplicate_dict = get_true_value(cfg["task"]["data"]["duplicate_dict"])

    logger(f"    duplicate_dict: {duplicate_dict}")

    logger(
        f"[Data] Loading finished. "
        "\n"
    )

    # ---------= [Scatter] =---------
    logger(f"[Scatter] Loading started. ")

    figsize = get_true_value(cfg["task"]["scatter"]["figsize"])
    figsize = tuple(figsize)
    marker_dict = get_true_value(cfg["task"]["scatter"]["marker_dict"])
    color_dict = get_true_value(cfg["task"]["scatter"]["color_dict"])

    logger(f"    figsize: {figsize}")
    logger(f"    marker_dict: {marker_dict}")
    logger(f"    color_dict: {color_dict}")

    logger(
        f"[Scatter] Loading finished. "
        "\n"
    )

    # ---------= [Save Plot] =---------
    logger(f"[Save Plot] Loading started. ")

    save_plot_root_path = get_true_value(cfg["task"]["save_plot"]["save_plot_root_path"])
    save_plot_root_path = Path(save_plot_root_path)

    logger(f"    save_plot_root_path: {save_plot_root_path}")

    logger(
        f"[Save Plot] Loading finished. "
        "\n"
    )

    # ---------= [All Components Loaded] =---------
    logger(
        f"All components loaded. "
        "\n"
    )

    # ---------= [Plot Scatter] =---------
    pruning_res_dict = load_yaml(pruning_res_dict_path)["pruning_res_dict"]

    skip_split_str = f"skip_{num_skipped_block}"
    skip_duplicate_list = duplicate_dict[skip_split_str]

    def check_duplicate(
        strategy_name: str
    ) -> bool: 
        for duplicate_strategy_pair in skip_duplicate_list:
            if strategy_name == duplicate_strategy_pair[0]:
                return duplicate_strategy_pair[1]

        return None

    split_dict = pruning_res_dict[split]
    
    num_attn_block = 7 if (sd_type in ["sd", "sd-turbo"]) \
        else 5

    point_list = []
    label_list = []

    marker_list = []
    color_list = []

    if num_skipped_block == 1:
        # baselines
        for block_idx in range(num_attn_block):
            strategy_name = f"a.{block_idx}"

            if not (strategy_name in split_dict.keys()):
                break
            
            fid_score = float(split_dict[strategy_name]["FID"])
            lpips_score = float(split_dict[strategy_name]["LPIPS"])
        
            point_list.append((fid_score, lpips_score))
            label_list.append(strategy_name)
            
            marker_list.append(marker_dict["baseline"])
            color_list.append(color_dict["baseline"])
        
        # ours
        strategy_idx = 0

        while True:
            strategy_name = f"b.{strategy_idx}"

            dup_strategy_name = check_duplicate(strategy_name)
            
            if (not dup_strategy_name) and (not (strategy_name in split_dict.keys())):
                break

            if dup_strategy_name:
                fid_score = float(split_dict[dup_strategy_name]["FID"])
                lpips_score = float(split_dict[dup_strategy_name]["LPIPS"])
            else:
                fid_score = float(split_dict[strategy_name]["FID"])
                lpips_score = float(split_dict[strategy_name]["LPIPS"])
        
            point_list.append((fid_score, lpips_score))
            label_list.append(strategy_name)

            if dup_strategy_name:
                marker_list.append(marker_dict["duplicate"])
                color_list.append(color_dict["duplicate"])
            else:
                marker_list.append(marker_dict["ours"])
                color_list.append(color_dict["ours"])

            strategy_idx += 1
    else:
        # baselines
        for block_idx in range(num_attn_block // 2):
            strategy_name = f"c.{block_idx}"

            if not (strategy_name in split_dict.keys()):
                break
            
            fid_score = float(split_dict[strategy_name]["FID"])
            lpips_score = float(split_dict[strategy_name]["LPIPS"])
        
            point_list.append((fid_score, lpips_score))
            label_list.append(strategy_name)

            marker_list.append(marker_dict["baseline"])
            color_list.append(color_dict["baseline"])
        
        # ours
        strategy_idx = 0

        while True:
            strategy_name = f"d.{strategy_idx}"

            dup_strategy_name = check_duplicate(strategy_name)
            
            if (not dup_strategy_name) and (not (strategy_name in split_dict.keys())):
                break

            if dup_strategy_name:
                fid_score = float(split_dict[dup_strategy_name]["FID"])
                lpips_score = float(split_dict[dup_strategy_name]["LPIPS"])
            else:
                fid_score = float(split_dict[strategy_name]["FID"])
                lpips_score = float(split_dict[strategy_name]["LPIPS"])
        
            point_list.append((fid_score, lpips_score))
            label_list.append(strategy_name)

            if dup_strategy_name:
                marker_list.append(marker_dict["duplicate"])
                color_list.append(color_dict["duplicate"])
            else:
                marker_list.append(marker_dict["ours"])
                color_list.append(color_dict["ours"])

            strategy_idx += 1

    sd_name_dict = {
        "sd-turbo": "SD-Turbo", 
        "sd": "SD v2.1", 
        "sdxl-turbo": "SDXL-Turbo", 
        "sdxl": "SDXL", 
    }
    true_sd_name = sd_name_dict[sd_type]

    split_name_dict = {
        "training": "Training", 
        "test": "Test"
    }
    true_split_name = split_name_dict[split]

    plot_title = f"{true_sd_name} {true_split_name} Set (Skip {num_skipped_block})"

    fig, ax = get_scatter(
        figsize = figsize, 

        point_list = point_list, 
        
        label_list = label_list, 

        marker_list = marker_list, 

        color_list = color_list, 
        
        # labels
        plot_x_label = "FID Score", 
        plot_y_label = "LPIPS Score", 

        plot_title = plot_title, 

        show_grid = True, 

        # legend
        show_legend = True, 
        legend_num_col = 2
    )

    save_plot(
        fig = fig, 

        save_plot_root_path = save_plot_root_path / sd_type, 
        save_plot_filename = f"{split}_{num_skipped_block}.png"
    )

    # `scatter_fid_lpips_implement()` done
    pass

def scatter_fid_lpips(
    cfg: DictConfig
):
    scatter_fid_lpips_implement(cfg)

    pass
