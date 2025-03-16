from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import json

import numpy as np

from pathlib import Path


def load_json(
    json_path: str
):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data

def save_json(
    data: Union[Dict, List], 

    json_root_path: str, 
    json_filename: str, 

    encoding: Optional[str] = "utf-8"
): 
    json_root_path = Path(json_root_path)
    json_root_path.mkdir(parents = True, exist_ok = True)
    
    json_path = json_root_path / json_filename

    with open(
        json_path, 
        "w", 
        encoding = encoding
    ) as f:
        json.dump(
            data, 
            f
        )
