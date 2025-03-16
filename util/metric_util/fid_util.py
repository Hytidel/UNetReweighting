# modified from official implementation of `pytorch-fid==0.3.0` library

from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
import torchvision
from torchvision import transforms

import numpy as np

from tqdm.auto import tqdm

from PIL import Image

import gc

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance


class ImageTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        img_pil_list: List[Image.Image], 
        tsfm = None
    ):
        self.img_pil_list = img_pil_list
        self.tsfm = tsfm
    
    def __len__(self):
        return len(self.img_pil_list)

    def __getitem__(
        self, 
        i: int
    ) -> Union[Image.Image, torch.FloatTensor]:
        if self.tsfm is None:
            return self.img_pil_list[i]
        
        return self.tsfm(self.img_pil_list[i])

def get_activations(
    img_pil_list: List[Image.Image], 

    model, 

    batch_size: int, 
    dims: int, 
    num_workers: Optional[int] = 1, 

    device: Optional[str] = "cpu"
) -> torch.FloatTensor:
    num_img = len(img_pil_list)

    if batch_size > num_img:
        logger(
            f"`batch_size` is larger than the data size, "
            f"set `batch_size` to data size ({num_img}). ", 
            log_type = "warning"
        )
        
        batch_size = num_img

    dataset = ImageTensorDataset(
        img_pil_list = img_pil_list, 
        tsfm = transforms.ToTensor()
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset, 

        batch_size = batch_size, 
        num_workers = num_workers, 

        shuffle = False, 
        drop_last = False
    )

    pred_arr = np.empty(shape = (num_img, dims))

    model.eval()

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : (start_idx + pred.shape[0])] = pred

        start_idx = start_idx + pred.shape[0]

    # clean up
    del dataset
    del dataloader
    torch.cuda.empty_cache()
    gc.collect()

    return pred_arr

def calculate_activation_statistics(
    img_pil_list: List[Image.Image], 

    model, 

    batch_size: int, 
    dims: int, 
    num_workers: Optional[int] = 1, 

    device: Optional[str] = "cpu"
) -> Tuple[float, float]:
    act = get_activations(
        img_pil_list = img_pil_list, 

        model = model, 

        batch_size = batch_size, 
        dims = dims, 
        num_workers = num_workers, 

        device = device
    )

    mean = np.mean(
        act, 
        axis = 0
    )
    std = np.cov(
        act, 
        rowvar = False
    )

    return mean, std

def compute_statistics(
    img_pil_list: List[torch.FloatTensor], 

    model, 

    batch_size: int, 
    dims: int, 
    num_workers: Optional[int] = 1, 

    device: Optional[str] = "cpu"
) -> Tuple[float, float]:
    mean, std = calculate_activation_statistics(
        img_pil_list = img_pil_list, 

        model = model, 

        batch_size = batch_size, 
        dims = dims, 
        num_workers = num_workers, 

        device = device
    )
    
    return mean, std

def cal_fid(
    img_pil_list_1: List[Image.Image], 
    img_pil_list_2: List[Image.Image], 

    batch_size: int, 
    feature_dim: int, 
    num_worker: Optional[int] = 1, 

    device: Optional[str] = "cpu", 

    model: Optional = None
) -> Tuple[float, nn.Module]:
    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[feature_dim]
        model = InceptionV3([block_idx]).to(device)

    mean_1, std_1 = compute_statistics(
        img_pil_list = img_pil_list_1, 

        model = model, 

        batch_size = batch_size, 
        dims = feature_dim, 
        num_workers = num_worker, 

        device = device
    )

    mean_2, std_2 = compute_statistics(
        img_pil_list = img_pil_list_2, 

        model = model, 

        batch_size = batch_size, 
        dims = feature_dim, 
        num_workers = num_worker, 

        device = device
    )

    fid_score = calculate_frechet_distance(
        mean_1, std_1, 
        mean_2, std_2
    )

    # clean up
    torch.cuda.empty_cache()
    gc.collect()

    return (
        fid_score, 
        model
    )
