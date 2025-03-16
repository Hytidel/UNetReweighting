from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import torch
import torchvision
from torchvision import transforms

import numpy as np

from tqdm.auto import tqdm

from matplotlib import pyplot as plt
from PIL import Image

import requests
from io import BytesIO

from pathlib import Path


rev_transform = transforms.Compose(
    [
        # value [-1, 1] -> [0, 1]
        transforms.Lambda(
            lambda t: (t + 1) / 2
        ), 

        # reshape to [h, w, num_channel]
        transforms.Lambda(
            lambda t: t.permute(1, 2, 0)
        ), 

        # value [0, 1] -> [0, 255]
        transforms.Lambda(
            lambda t: t * 255
        ), 

        # convert to numpy
        transforms.Lambda(
            lambda t: t.numpy() \
                .astype(np.uint8)
        ), 

        # convert to PIL
        transforms.ToPILImage()
    ]
)

def load_img_path(
    img_path, 
    size = None
):
    img = Image.open(img_path) \
        .convert("RGB")

    if size is not None:
        img = img.resize(size)

    return img

def load_img_url(
    url, 
    size = None
):
    response = requests.get(
        url, 
        timeout = 1.0
    )
    img = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    if size is not None:
        img = img.resize(size)

    return img

@torch.no_grad()
def img_pil_to_tensor(
    img_pil, 
    img_size, 
    add_batch_size_dim = False
):
    transform = torchvision.transforms.Compose(
        [
            # resize
            torchvision.transforms.Resize(img_size), 
            torchvision.transforms.CenterCrop(img_size), 

            # reshape to [num_channel, h, w] , value [0, 255] -> [0, 1]
            torchvision.transforms.ToTensor(), 

            # value [0, 1] -> [-1, 1]
            torchvision.transforms.Lambda(
                lambda t: t * 2 - 1
            )
        ]
    )

    tmp_img_pil = img_pil
    img_tensor = transform(tmp_img_pil)
    
    if add_batch_size_dim:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

@torch.no_grad()
def img_tensor_to_pil(
    img_tensor, 
    remove_batch_size_dim = False
):
    tmp_img_tensor = img_tensor
    if remove_batch_size_dim:
        tmp_img_tensor = tmp_img_tensor.squeeze(dim = 0)
    
    img_pil = rev_transform(tmp_img_tensor)

    return img_pil

def save_pil_as_png(
    pil, 
    png_root_path, 
    png_filename
):
    png_root_path = Path(png_root_path)
    png_root_path.mkdir(parents = True, exist_ok = True)

    png_path = png_root_path / png_filename

    pil.save(
        png_path, 
        "PNG"
    )

def display_pil_img_list(
    img_list, 
    num_rows, num_cols, 
    figsize = (12, 10), 
    gray = False
):
    fig, axs = plt.subplots(
        num_rows, num_cols, 
        figsize = figsize
    )

    for idx, ax in enumerate(axs.flatten()):
        if idx < len(img_list):
            ax.imshow(
                img_list[idx], 
                cmap = "gray" if gray else "viridis"
            )

            ax.axis("off")
    
    plt.tight_layout()
    
    plt.show()

def save_img_tensor_list_as_png(
    img_tensor_list, 
    png_root_path, 
    png_filename, 
    num_img_per_row
):
    assert len(img_tensor_list) > 0

    # shape = [num_channel, h, w]
    if img_tensor_list[0].ndim == 3:
        img_tensor = torch.stack(
            img_tensor_list, 
            dim = 0
        )
    # shape = [batch_size, num_channel, h, w]
    else:
        img_tensor = torch.cat(
            img_tensor_list, 
            dim = 0
        )

    # value: [-1, 1] -> [0, 1] 
    img_tensor = (img_tensor + 1) / 2

    png_root_path = Path(png_root_path)
    png_root_path.mkdir(parents = True, exist_ok = True)

    png_path = png_root_path / png_filename

    torchvision.utils.save_image(
        img_tensor, 
        png_path, 
        nrow = num_img_per_row
    )
    
    logger(
        f"Successfully saved `{png_path}`. ", 
        log_type = "info"
    )

def merge_img_pil_list(
    img_pil_list: List[Image.Image], 
    num_row: Optional[int] = None, 
    num_col: Optional[int] = None, 
    background_color: Optional[Tuple[float, float, float]] = (255, 255, 255),  # white
) -> Image.Image:
    num_img = len(img_pil_list)

    # set default `num_row` and `num_col`
    if (num_row is None) or (num_col is None):
        if (num_row is None) and (num_col is None):
            num_row = 1
            num_col = num_img
        elif num_row is None:
            num_row = (num_img + num_col - 1) // num_col
        else:
            num_col = (num_img + num_row - 1) // num_row

    # check size
    num_grid = num_row * num_col
    if num_grid < num_img:
        logger(
            f"The number of grids is less than the number of images, "
            f"got {num_grid} and {num_img}, "
            f"only the first {num_grid} image(s) in `img_pil_list` will be displayed.", 
            log_type = "warning"
        )
    elif num_grid > num_img:
        logger(
            f"The number of grids is larger than the number of images, "
            f"got {num_grid} and {num_img}, "
            f"the grid(s) for the unprovided image(s) will remain blank. ", 
            log_type = "warning"
        )

    width_list, height_list = zip(
        *(img.size for img in img_pil_list)
    )

    max_width = max(width_list)
    max_height = max(height_list)

    res_img_pil = Image.new(
        "RGB", 
        (max_width * num_col, max_height * num_row), 
        background_color
    )
    
    for i in range(num_row):
        break_loop = False

        for j in range(num_col):
            img_pil_idx = i * num_col + j
            if img_pil_idx >= num_img:
                break_loop = True
                break
            
            res_img_pil.paste(
                img_pil_list[img_pil_idx], 
                (j * max_width, i * max_height)
            )

        if break_loop:
            break
    
    return res_img_pil

def load_img_folder_as_pil_list(
    img_root_path: str, 
    size: Optional[Union[float, Tuple[float]]] = None, 
    sort_lambda: Optional = None
) -> List[Image.Image]:
    img_root_path = Path(img_root_path)

    if not img_root_path.is_dir():
        raise ValueError(
            f"Path `{img_root_path}` not exists. " 
        )

    img_filename_list = img_root_path.iterdir()

    if sort_lambda is None:
        img_filename_list.sort(key = sort_lambda)

    img_path_list = [
        (img_root_path / img_filename) \
            for img_filename in img_filename_list
    ]

    img_pil_list = [
        load_img_path(
            img_path = img_path, 
            size = size
        ) for img_path in img_path_list
    ]
    return img_pil_list

def load_img_folder_as_tensor(
    img_root_path: str, 
    size: Optional[Union[float, Tuple[float]]] = (512, 512), 
) -> Tuple[int, torch.Tensor]:
    img_pil_list = load_img_folder_as_pil_list(
        img_root_path = img_root_path, 
        size = size
    )

    num_img = len(img_pil_list)

    tsfm = transforms.Compose(
        [
            transforms.Resize(size), 

            # value: [0, 255] -> [0, 1]
            transforms.ToTensor()
        ]
    )

    img_tensor_list = [
        tsfm(img_pil) \
            for img_pil in img_pil_list
    ]

    batch_img_tensor = torch.stack(img_tensor_list)

    return num_img, batch_img_tensor

def split_img_pil(
    img_pil: Image.Image, 
    num_row: int, 
    num_col: int
) -> List[Image.Image]:
    tot_width, tot_height = img_pil.size

    single_width, single_height = tot_width / num_col, tot_height / num_row

    crop_img_pil_list = []
    for i in range(num_row):
        for j in range(num_col):
            top_left_x, top_left_y = j * single_width, i * single_height
            bottom_right_x, bottom_right_y = (j + 1) * single_width, (i + 1) * single_height

            crop_img_pil = img_pil.crop(
                (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            )
            crop_img_pil_list.append(crop_img_pil)

    return crop_img_pil_list

def save_pil_list_as_png(
    pil_list: List[Image.Image], 
    dst_img_root_path: str, 
    png_filename: Union[str, List[str]]
):
    num_img = len(pil_list)
    
    if isinstance(png_filename, list):
        if len(png_filename) != num_img:
            raise ValueError(
                f"The number of `png_filename` does not match the number of images, "
                f"got {len(png_filename)} and {num_img}. "
            )
        else:
            png_name_list = png_filename
    else:
        dst_img_root_path = Path(dst_img_root_path)
        true_png_name = dst_img_root_path.stem
        png_name_list = [
            f"{true_png_name}_{i}.png" \
                for i in range(num_img)
        ]

    for (pil, png_filename) in zip(pil_list, png_name_list):
        save_pil_as_png(
            pil = pil, 
            png_root_path = dst_img_root_path, 
            png_filename = png_filename
        )

def cal_img_pair_mse(
    img_pil_1: Image.Image, 
    img_pil_2: Image.Image, 
) -> float:
    img_np_1 = np.array(
        img_pil_1, 
        dtype = np.float32
    )
    img_np_2 = np.array(
        img_pil_2, 
        dtype = np.float32
    )

    if img_np_1.shape != img_np_2.shape:
        raise ValueError(
            f"Image must have the same shape, "
            f"got {img_np_1.shape} and {img_np_2.shape}. "
        )

    if np.max(img_np_1) > 1.0:
        img_np_1 /= 255.0
    if np.max(img_np_2) > 1.0:
        img_np_2 /= 255.0
    
    return np.mean(
        (img_np_1 - img_np_2) ** 2
    )
    