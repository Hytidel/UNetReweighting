from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import torch

import numpy as np

from PIL import Image


def tsfm_noise_pred_list(
    # noise_pred_list.shape = [num_inference_step, batch_size, num_channel, noise_pred_h, noise_pred_w]
    noise_pred_list: List[torch.Tensor], 

    num_inference_step: int, 
    batch_size: int
) -> List[List[torch.Tensor]]:
    # process_noise_pred_list.shape = [batch_size, num_inference_step, num_channel, noise_pred_h, noise_pred_w]
    process_noise_pred_list = [[] for _ in range(batch_size)]
    
    for step_idx in range(num_inference_step):
        for sample_idx in range(batch_size):
            process_noise_pred_list[sample_idx].append(
                noise_pred_list[step_idx][sample_idx]
            )

    return process_noise_pred_list

def tsfm_latent_list(
    # latent_list.shape = [num_inference_step + 1, batch_size, num_channel, latent_h, latent_w]
    latent_list: List[torch.Tensor], 

    num_inference_step: int, 
    batch_size: int
) -> List[List[torch.Tensor]]:
    # process_latent_list.shape = [batch_size, num_inference_step + 1, num_channel, latent_h, latent_w]
    process_latent_list = [[] for _ in range(batch_size)]
    
    for step_idx in range(num_inference_step + 1):
        for sample_idx in range(batch_size):
            process_latent_list[sample_idx].append(
                latent_list[step_idx][sample_idx]
            )

    return process_latent_list

def tsfm_pil_list(
    # pil_list.shape = [num_inference_step + 1, batch_size]
    pil_list: List[List[Image.Image]], 

    num_inference_step: int, 
    batch_size: int
) -> List[List[Image.Image]]:
    # process_pil_list = [batch_size, num_inference_step + 1]
    process_pil_list = [[] for _ in range(batch_size)]

    for step_idx in range(num_inference_step + 1):
        for sample_idx in range(batch_size):
            process_pil_list[sample_idx].append(
                pil_list[step_idx][sample_idx]
            )

    return process_pil_list
