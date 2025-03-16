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
    get_bar_chart
)


# @torch.no_grad()
def bar_chart_voting_score_implement(
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

    # ---------= [Image Score Path] =---------
    logger(f"[Image Score Path] Loading started. ")

    importance_score_path = get_true_value(cfg["task"]["importance_score_path"])
    importance_score_path = Path(importance_score_path)

    logger(f"    importance_score_path: {importance_score_path}")

    logger(
        f"[Image Score Path] Loading finished. "
        "\n"
    )

    # ---------= [Num Vote Round] =---------
    logger(f"[Num Vote Round] Loading started. ")

    num_vote_round = get_true_value(cfg["task"]["num_vote_round"])

    logger(f"    num_vote_round: {num_vote_round}")

    logger(
        f"[Num Vote Round] Loading finished. "
        "\n"
    )

    # ---------= [Bar Chart] =---------
    logger(f"[Bar Chart] Loading started. ")

    figsize = get_true_value(cfg["task"]["bar_chart"]["figsize"])
    figsize = tuple(figsize)
    bar_color_list = get_true_value(cfg["task"]["bar_chart"]["bar_color_list"])

    if sd_type in ["sdxl-turbo", "sdxl"]:
        bar_color_list = bar_color_list[1: -1]

    logger(f"    figsize: {figsize}")
    logger(f"    bar_color_list: {bar_color_list}")

    logger(
        f"[Bar Chart] Loading finished. "
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

    # ---------= [Plot Bar Chart] =---------
    voting_score_matrix = load_yaml(importance_score_path)["importance_score_matrix"]
    voting_score_matrix = np.asarray(
        voting_score_matrix, 
        dtype = np.float32
    )

    num_inference_step, num_attn_block = voting_score_matrix.shape

    max_voting_score = num_vote_round * num_attn_block

    x_label_list = [
        f"{i}" \
            for i in range(num_attn_block)
    ]

    sd_name_dict = {
        "sd-turbo": "SD-Turbo", 
        "sd": "SD v2.1", 
        "sdxl-turbo": "SDXL-Turbo", 
        "sdxl": "SDXL", 
    }
    true_sd_name = sd_name_dict[sd_type]

    for step_idx in range(num_inference_step):
        plot_title = f"{true_sd_name} Voting Score (step-{step_idx})"

        fig, ax = get_bar_chart(
            figsize = figsize, 

            y_list = voting_score_matrix[step_idx], 
            y_upper_lim = max_voting_score, 

            x_label_list = x_label_list, 

            bar_color_list = bar_color_list, 

            show_text_annotation = True, 
            
            # labels
            plot_x_label = "Block Index", 
            plot_y_label = "Voting Score", 

            plot_title = plot_title
        )

        save_plot(
            fig = fig, 

            save_plot_root_path = save_plot_root_path / sd_type, 
            save_plot_filename = f"{step_idx}.png"
        )

    # `bar_chart_voting_score_implement()` done
    pass

def bar_chart_voting_score(
    cfg: DictConfig
):
    bar_chart_voting_score_implement(cfg)

    pass
