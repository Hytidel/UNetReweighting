from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from pathlib import Path


def is_data_exist(
    save_weight_threshold_matrix_list_path: str, 
    num_round: int
) -> bool:
    save_weight_threshold_matrix_list_path = Path(save_weight_threshold_matrix_list_path)
    
    if save_weight_threshold_matrix_list_path.is_dir():
        data_path \
            = save_weight_threshold_matrix_list_path / f"round-{num_round - 1}" / "history_weight_threshold_matrix_list.yaml"
            
        if not data_path.is_file():
            return False
    else:
        return False

    return True
