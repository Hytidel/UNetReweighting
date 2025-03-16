from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from pathlib import Path


def is_data_exist(
    save_png_root_path: str, 
    num_sample: int, 

    sample_start_idx: Optional[int] = 0
) -> bool:
    save_png_root_path = Path(save_png_root_path)
    
    if save_png_root_path.is_dir():
        data_path = save_png_root_path / f"{sample_start_idx + num_sample - 1}.png"
        
        if not data_path.is_file():
            return False
    else:
        return False

    return True

def is_data_both_exist(
    sample_folder_root_path: Union[str, Path], 
    importance_folder_root_path: Union[str, Path], 

    weight_matrix_name: str
) -> bool:
    if isinstance(sample_folder_root_path, str):
        sample_folder_root_path = Path(sample_folder_root_path)
    
    save_png_root_path = sample_folder_root_path / "png" / "default"
    if not save_png_root_path.is_dir():
        return False

    if isinstance(importance_folder_root_path, str):
        importance_folder_root_path = Path(importance_folder_root_path)
    
    weight_matrix_path = importance_folder_root_path / "importance_weight_matrix" / f"{weight_matrix_name}.yaml"
    if not weight_matrix_path.is_file():
        return False
    
    return True
