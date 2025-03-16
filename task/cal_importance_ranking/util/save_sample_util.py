from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from pathlib import Path


def is_data_exist(
    folder_root_path: Union[str, Path], 

    save_importance_score: bool, 
    save_prefix_importance_score: bool, 
    
    save_importance_ranking: bool, 
    save_prefix_importance_ranking: bool, 
    
    save_strategy_skip_1: bool, 
    save_strategy_skip_2: bool, 
) -> bool:
    if isinstance(folder_root_path, str):
        folder_root_path = Path(folder_root_path)
    
    # importance_score_matrix
    tmp_root_path = folder_root_path / "importance_score_matrix"
    if save_importance_score:
        tmp_path = tmp_root_path / "importance_score_matrix.yaml"
        if not tmp_path.is_file():
            return False
    if save_prefix_importance_score:
        tmp_path = tmp_root_path / "prefix_importance_score_matrix_list.yaml"
        if not tmp_path.is_file():
            return False

    # importance_ranking_matrix
    tmp_root_path = folder_root_path / "importance_ranking_matrix"
    if save_importance_ranking:
        tmp_path = tmp_root_path / "importance_ranking_matrix.yaml"
        if not tmp_path.is_file():
            return False
    if save_prefix_importance_ranking:
        tmp_path = tmp_root_path / "prefix_importance_ranking_matrix_list.yaml"
        if not tmp_path.is_file():
            return False

    # skipping_strategy
    save_strategy_root_path = folder_root_path / "skipping_strategy"
    tmp_root_path = save_strategy_root_path / "skip-1"
    if not tmp_root_path.is_dir():
        return False
    tmp_root_path = save_strategy_root_path / "skip-2"
    if not tmp_root_path.is_dir():
        return False

    return True
