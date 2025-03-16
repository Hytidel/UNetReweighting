from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

import torch
from torch.nn import functional as F

import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

from pathlib import Path

from util.plot_util import merge_line_chart_list
from util.numpy_util import (
    tsfm_to_1d_array, 
    tsfm_to_2d_matrix
)

from importance_probe.weight_threshold_scheduler import WeightThresholdScheduler


def get_weight_max_bias_list(
    weight_max_bias_strategy: str,  # ["linear", "list"]
    weight_max_bias_st: Optional[float], 
    weight_max_bias_ed: Optional[float], 
    weight_max_bias_list: Optional[Union[float, List[float]]], 
    num_inference_step: Optional[int]
) -> np.ndarray:
    if weight_max_bias_strategy == "linear":
        if num_inference_step == 1:
            if not np.isclose(weight_max_bias_st, weight_max_bias_ed):
                raise ValueError(
                    f"`weight_max_bias_st` must equal to `weight_max_bias_ed` "
                    f"when `num_inference_step = 1`. "
                )
        
        weight_max_bias_list = np.linspace(
            weight_max_bias_st, weight_max_bias_ed, 
            num_inference_step
        )
    elif weight_max_bias_strategy == "list":
        weight_max_bias_list = tsfm_to_1d_array(
            array = weight_max_bias_list, 
            target_length = num_inference_step
        )
    else:
        raise NotImplementedError(
            f"Unsupported `weight_max_bias_strategy`, got `{type(weight_max_bias_strategy)}`."
        )

    return weight_max_bias_list

def get_weight_matrix(
    weight_matrix: Union[float, List[float], List[List[float]], np.ndarray], 
    num_inference_step: int, 
    num_attn_block: int
) -> np.ndarray:
    # weight_matrix.shape = (num_inference_step, num_attn_block)
    weight_matrix = tsfm_to_2d_matrix(
        matrix = weight_matrix, 
        target_shape = (num_inference_step, num_attn_block)
    )

    return weight_matrix

def get_noise_pred_loss_threshold_list(
    noise_pred_loss_threshold_strategy: str,  # ["linear", "list"]
    noise_pred_loss_threshold_st: Optional[float], 
    noise_pred_loss_threshold_ed: Optional[float], 
    noise_pred_loss_threshold_list: Optional[Union[float, List[float]]], 
    num_inference_step: Optional[int]
) -> np.ndarray:
    if noise_pred_loss_threshold_strategy == "linear":
        if num_inference_step == 1:
            if not np.isclose(noise_pred_loss_threshold_st, noise_pred_loss_threshold_ed):
                raise ValueError(
                    f"`noise_pred_loss_threshold_st` must equal to `noise_pred_loss_threshold_ed` "
                    f"when `num_inference_step = 1`. "
                )

        noise_pred_loss_threshold_list = np.linspace(
            noise_pred_loss_threshold_st, noise_pred_loss_threshold_ed, 
            num_inference_step
        )
    elif noise_pred_loss_threshold_strategy == "list":
        noise_pred_loss_threshold_list = tsfm_to_1d_array(
            array = noise_pred_loss_threshold_list, 
            target_length = num_inference_step
        )
    else:
        raise NotImplementedError(
            f"Unsupported `noise_pred_loss_threshold_strategy`, got `{noise_pred_loss_threshold_strategy}`."
        )
    
    if not isinstance(noise_pred_loss_threshold_list, np.ndarray):
        noise_pred_loss_threshold_list = np.asarray(noise_pred_loss_threshold_list)

    return noise_pred_loss_threshold_list

def save_merged_inference_step_chart(
    weight_threshold_scheduler: WeightThresholdScheduler, 
    chart_root_path: str, 
    chart_name: str, 
    figsize: Tuple[float, float], 
    marker_list: List[str], 
    num_row: int, 
    num_col: int, 
):
    history_weight_threshold_matrix_list \
        = weight_threshold_scheduler.history_weight_threshold_matrix_list

    inference_step_weight_threshold_matrix_list \
        = WeightThresholdScheduler.tsfm_history_weight_threshold_matrix_list(history_weight_threshold_matrix_list)
    
    inference_step_chart_list \
        = WeightThresholdScheduler.get_inference_step_chart_list(
            inference_step_weight_threshold_matrix_list = inference_step_weight_threshold_matrix_list, 
            figsize = figsize, 
            marker_list = marker_list
        )
    
    merged_inference_step_chart, _ = merge_line_chart_list(
        chart_list = inference_step_chart_list, 
        figsize = (num_col * figsize[0], num_row * figsize[1]), 
        num_row = num_row, num_col = num_col
    )

    chart_root_path = Path(chart_root_path)
    chart_root_path.mkdir(parents = True, exist_ok = True)

    chart_path = chart_root_path / chart_name
    merged_inference_step_chart.savefig(chart_path)

    plt.close()

def cal_skip_rate(
    weight_list: Union[float, List[float], np.ndarray], 
    weight_threshold_list: Union[float, List[float], np.ndarray], 
    num_attn_block: int
) -> float:
    weight_list = tsfm_to_1d_array(
        array = weight_list, 
        target_length = num_attn_block
    )

    weight_threshold_list = tsfm_to_1d_array(
        array = weight_threshold_list, 
        target_length = num_attn_block
    )

    return sum(weight_list < weight_threshold_list) / num_attn_block

def cal_fitness(
    init_energy: float, 
    weight_scaling_scheduler: "WeightScalingScheduler", 
    weight_list: Union[float, List[float], np.ndarray], 
    weight_threshold_list: Union[float, List[float], np.ndarray], 
    unet: "UNet2DCondition", 
):
    fitness \
        = init_energy / weight_scaling_scheduler.energy \
            + cal_skip_rate(
                weight_list = weight_list, 
                weight_threshold_list = weight_threshold_list, 
                num_attn_block = unet.num_attn_block
            )

    return fitness

@torch.no_grad()
def cal_noise_pred_loss(
    tea_noise_pred: torch.Tensor, 
    stu_noise_pred: torch.Tensor, 
) -> float:
    noise_pred_loss = F.mse_loss(tea_noise_pred, stu_noise_pred)

    return noise_pred_loss
