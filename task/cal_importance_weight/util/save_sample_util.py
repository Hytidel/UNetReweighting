from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from pathlib import Path


def is_data_exist(
    folder_root_path: Union[str, Path], 

    weight_range_str: str
) -> bool:
    if isinstance(folder_root_path, str):
        folder_root_path = Path(folder_root_path)
    
    tmp_path = folder_root_path / "importance_weight_matrix" / f"{weight_range_str}.yaml"
    
    if not tmp_path.is_file():
        return False

    return True
