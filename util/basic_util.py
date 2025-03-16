from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import torch
import numpy as np
import random

from pathlib import Path

import time

from util.torch_util import get_device


global_variable_dict = {
    # torch
    "device": None, 

    # random
    "seed": None, 

    # basic
    "exp_name": None, 
    "start_time": None, 
    "default_exp_name": None, 
    "add_timestamp": None, 

    # debug
    "enable_pause": None, 

    # concurrent
    "concurrent_max_worker": None
}


def get_time_str():
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    return time_str

def get_global_variable(
    var_name
):
    if not var_name in global_variable_dict:
        raise KeyError(f"Unknown key: {var_name}")
    
    return global_variable_dict[var_name]

def set_global_variable(
    var_name, 
    var_val
):
    if not var_name in global_variable_dict:
        logger(
            f"`{var_name}` is not in `global_variable_dict` before setting. ", 
            log_type = "warning"
        )
    
    global_variable_dict[var_name] = var_val

def pause():
    if not get_global_variable("enable_pause"):
        return

    breakpoint()

def seed_everything(
    seed
):
    set_global_variable("seed", seed)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_global_variable_dict(
    cfg
):
    seed = get_true_value(cfg["seed"])
    seed_everything(seed)

    # ---------= [torch] =---------
    device = get_device()
    set_global_variable("device", device)

    logger(f"Running on device: {device}")

    # ---------= [basic] =---------
    start_time = get_time_str()
    set_global_variable("start_time", start_time)

    default_exp_name = get_true_value(cfg["exp_name"]["default_exp_name"])
    add_timestamp = get_true_value(cfg["exp_name"]["add_timestamp"])
    set_global_variable("default_exp_name", default_exp_name)
    set_global_variable("add_timestamp", add_timestamp)

    exp_name = f"{default_exp_name}_{start_time}" if add_timestamp \
        else default_exp_name
    set_global_variable("exp_name", exp_name)

    enable_pause = get_true_value(cfg["enable_pause"])
    set_global_variable("enable_pause", enable_pause)

    concurrent_max_worker = get_true_value(cfg["concurrent_max_worker"])
    set_global_variable("concurrent_max_worker", concurrent_max_worker)

def is_none(
    var
):
    return (var is None) or (var == "None")

def get_true_value(
    var
):
    return None if is_none(var) \
        else var

def get_timestamp(
    to_int: bool = True
) -> Union[int, float]:
    timestamp = time.time()
    
    if to_int:
        timestamp = int(timestamp)
    
    return timestamp
