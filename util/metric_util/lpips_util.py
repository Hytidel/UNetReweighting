from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol, Any

import torch
import torch.nn as nn

from PIL import Image

from util.image_util import img_pil_to_tensor

from lpips import LPIPS


def cal_lpips(
    img_pil_list_1: List[Image.Image], 
    img_pil_list_2: List[Image.Image], 
    img_size: Union[int, Tuple[int, int]], 

    lpips_net_type: Optional[str] = "alex",  # ["alex", "vgg"]

    device: Optional[str] = "cpu", 

    model: Optional = None
) -> Union[float, nn.Module]:
    if model is None:
        model = LPIPS(net = lpips_net_type) \
            .to(device)
    
    img_tensor_list_1 = [
        img_pil_to_tensor(
            img_pil = img_pil, 
            img_size = img_size
        ).to(device) \
            for img_pil in img_pil_list_1
    ]
    img_tensor_list_2 = [
        img_pil_to_tensor(
            img_pil = img_pil, 
            img_size = img_size
        ).to(device) \
            for img_pil in img_pil_list_2
    ]

    sum_lpips_score = 0.0

    for img_tensor_1, img_tensor_2 in zip(
        img_tensor_list_1, 
        img_tensor_list_2
    ):
        batch_lpips_score = model(img_tensor_1, img_tensor_2)

        sum_lpips_score += batch_lpips_score

    num_img = len(img_tensor_list_1)
    lpips_score = sum_lpips_score.item() / num_img

    return (
        lpips_score, 
        model
    )
