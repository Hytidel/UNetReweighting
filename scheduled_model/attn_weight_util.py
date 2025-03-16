from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import numpy as np

import os
import shutil

from util.basic_util import pause, get_global_variable, is_none, get_true_value


def cal_attn_weight_matrix(
    # attn_grad_norm_matrix.shape = (num_inference_step, num_attn_block)
    attn_grad_norm_matrix: Union[List[List[float]], np.ndarray], 

    # bias_range
    low_bias: float, 
    high_bias: float,  

    bias_schedule: Optional[str] = "linear", 

    # decay_schedule
    enable_decay: Optional[bool] = True, 
    decay_schedule: Optional[str] = "linear",  # ["linear", "exp"]
    decay_schedule_exp_lambda: Optional[float] = 0.1
) -> np.ndarray:
    if not isinstance(attn_grad_norm_matrix, np.ndarray):
        attn_grad_norm_matrix = np.asarray(attn_grad_norm_matrix)
    
    num_inference_step, num_attn_block = attn_grad_norm_matrix.shape

    # norm to [0, 1]
    normed_attn_grad_norm_matrix \
        = attn_grad_norm_matrix / attn_grad_norm_matrix.max(
            axis = -1, 
            keepdims = True
        )

    # attn_weight_matrix.shape = (num_inference_step, num_attn_block)
    attn_weight_matrix = normed_attn_grad_norm_matrix * (high_bias - low_bias) + low_bias

    # bias decay
    if enable_decay:
        for t in range(num_inference_step):        
            if decay_schedule == "linear":
                attn_weight_matrix[t] *= (num_inference_step - t) / num_inference_step
            elif decay_schedule == "exp":
                attn_weight_matrix[t] *= np.exp(-decay_schedule_exp_lambda * t)
            else:
                raise NotImplementedError(
                    f"Unsupported `decay_schedule`, got {decay_schedule}. "
                )

    attn_weight_matrix += 1.0

    return attn_weight_matrix

def get_attn_weight_matrix_name(
    # bias_range
    low_bias: float, 
    high_bias: float,  

    bias_schedule: Optional[str] = "linear", 

    # decay_schedule
    enable_decay: Optional[bool] = True, 
    decay_schedule: Optional[str] = "linear",  # ["linear", "exp"]
    decay_schedule_exp_lambda: Optional[float] = 0.1
) -> str:
    attn_weight_matrix_name = f"{low_bias}_{high_bias}_bias-{bias_schedule}"
    if enable_decay:
        attn_weight_matrix_name += f"_decay-{decay_schedule}"

        if decay_schedule == "exp":
            attn_weight_matrix_name += f"_lambda-{decay_schedule_exp_lambda}"

    return attn_weight_matrix_name
