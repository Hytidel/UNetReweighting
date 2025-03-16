from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import numpy as np


def quadratic_sum(
    num_weight: int, 
    weight_list: Union[List[float], np.ndarray]
):
    if num_weight > len(weight_list):
        raise ValueError(
            f"`num_weight` is larger than the length of `weight_list`, "
            f"got {num_weight} and {len(weight_list)}. "
        )

    res = sum(
        (weight_list[i] ** 2) \
            for i in range(num_weight)
    )
    
    return res

def get_energy_func(
    energy_func_name: str, 
) -> Callable:
    if energy_func_name == "quadratic_sum":
        return quadratic_sum
    else:
        raise NotImplementedError(
            f"Unsupported energy function: `{energy_func_name}`. "
        )
    