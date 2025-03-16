from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from contextlib import contextmanager

from pathlib import Path

import numpy as np


def get_device():
    device = "cuda" if torch.cuda.is_available() \
        else "cpu"

    return device

def get_optim(
    optim_type: str, 
    model: nn.Module, 
    lr: float, 

    # AdamW
    adamw_beta_tuple: Optional[Tuple[float, float]] = (0.9, 0.99)
):
    if optim_type == "Adam":
        optim = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()), 
            lr = lr
        )
    elif optim_type == "AdamW":
        optim = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, model.parameters()), 
            lr = lr, 

            betas = adamw_beta_tuple
        )
    else:
        raise NotImplementedError(
            f"Unsupported optimizer:` {optim_type}`. "
        )

    return optim

def get_lr_scheduler(
    lr_scheduler_type: str, 
    optim, 

    # `ReduceLROnPlateau` param
    mode: str = "min", 
    factor: float = None, 
    patience: int = None, 
    cooldown: int = None, 
    threshold: float = None, 

    verbose: bool = True, 
):
    """
    Args:
        verbose (`bool`, *optional*, defaults to True):
            Set `verbose = True` for `lr_scheduler` to print prompt messages 
            when the learning rate changes. 
    """

    if lr_scheduler_type is None:
        lr_scheduler = None
    elif lr_scheduler_type == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode = mode, 
            factor = factor, 
            patience = patience, 
            cooldown = cooldown, 
            threshold = threshold, 

            verbose = verbose
        )
    else:
        raise NotImplementedError(
            f"Unsupported learning rate scheduler:` {lr_scheduler_type}`. "
        )
    
    return lr_scheduler

def get_criterion(
    criterion_type
):
    if criterion_type == "L1":
        criterion = F.l1_loss
    elif criterion_type in ["L2", "MSE"]:
        criterion = F.mse_loss
    elif criterion_type == "Huber":
        criterion = F.smooth_l1_loss
    else:
        raise NotImplementedError(
            f"Unsupported criterion:` {criterion_type}`. "
        )

    return criterion

def save_model_state_dict(
    state_dict: Dict, 
    ckpt_root_path: str, 
    ckpt_filename: str
):
    ckpt_root_path = Path(ckpt_root_path)
    ckpt_root_path.mkdir(parents = True, exist_ok = True)
    
    ckpt_path = ckpt_root_path / ckpt_filename

    torch.save(
        state_dict, 
        ckpt_path
    )

def save_model_ckpt(
    model, 
    ckpt_root_path: str, 
    ckpt_filename: str
):
    state_dict = model.state_dict()

    save_model_state_dict(
        state_dict, 
        ckpt_root_path, 
        ckpt_filename
    )

def load_model_state_dict(
    state_dict_path: str, 
    device: str
) -> Dict:
    state_dict_path = Path(state_dict_path)
    
    if (state_dict_path is None) or (state_dict_path == "None") \
        or (not state_dict_path.is_file()):
        logger(
            f"State dict `{state_dict_path}` not exists, continue with initial model parameters. ", 
            log_type = "warning"
        )

        return None
    
    state_dict = torch.load(
        state_dict_path, 
        map_location = device
    )

    return state_dict

def load_model_ckpt(
    model, 
    ckpt_path: str, 
    device: str, 
    strict: Optional[bool] = False
):
    ckpt_path = Path(ckpt_path)

    if (ckpt_path is None) or (ckpt_path == "None") \
        or (not ckpt_path.is_file()):
            logger(
                f"Model checkpoint `{ckpt_path}` not exists, continue with initial model parameters. ", 
                log_type = "info"
            )

            return

    state_dict = torch.load(
        ckpt_path, 
        map_location = device
    )

    model.load_state_dict(
        state_dict, 
        strict = strict
    )

    logger(
        f"Loaded model checkpoint `{ckpt_path}`.", 
        log_type = "info"
    )

def get_model_num_param(
    model
):
    model_num_param = sum(
        [
            param.numel() \
                for param in model.parameters()
        ]
    )

    return model_num_param

def get_current_lr_list(
    optim
):
    cur_lr_list = [
        param_group["lr"] \
            for param_group in optim.param_groups
    ]

    return cur_lr_list

def get_generator(
    seed, 
    device
):
    generator = torch.Generator(device).manual_seed(seed)
    
    return generator

def get_selected_state_dict(
    model, 
    selected_param_name_list: List[str]
) -> Dict:
    state_dict = model.state_dict()

    selected_state_dict = {
        name: state_dict[name] \
            for name in selected_param_name_list
    }

    return selected_state_dict

@contextmanager
def determine_enable_grad(
    enable_grad: bool
):
    if enable_grad:
        with torch.enable_grad():
            yield
    else:
        with torch.no_grad():
            yield

def soft_update_model(
    model: nn.Module, 
    target_model: nn.Module, 

    soft_update_lambda: float
):
    for (param, target_param) in zip(
        model.parameters(), 
        target_model.parameters()
    ):
        target_data = target_param.data * soft_update_lambda + param.data * (1 - soft_update_lambda)
        target_param.data.copy_(target_data)
