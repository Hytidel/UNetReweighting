from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import pickle

import numpy as np

from functools import lru_cache

import torch

from pathlib import Path


_PKL_CACHE_MAXSIZE = 100


def load_pkl(
    pkl_path: str
) -> Union[List, np.ndarray, torch.Tensor]:
    pkl_path = Path(pkl_path)

    if not pkl_path.is_file():
        raise ValueError(
            f"File `{pkl_path}` not exists. "
        )

    res = None
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    
    return res

@lru_cache(maxsize = _PKL_CACHE_MAXSIZE)
def load_pkl_cached(
    pkl_path: str
) -> Union[List, np.ndarray, torch.Tensor]:
    return load_pkl(pkl_path)

def save_pkl(
    var: Union[List, np.ndarray, torch.Tensor], 
    pkl_root_path: str, 
    pkl_filename: str
): 
    pkl_root_path = Path(pkl_root_path)
    pkl_root_path.mkdir(parents = True, exist_ok = True)
    
    pkl_path = pkl_root_path / pkl_filename

    with open(pkl_path, "wb") as f:
        pickle.dump(var, f)
    