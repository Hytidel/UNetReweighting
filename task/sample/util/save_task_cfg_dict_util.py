from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import copy

from pathlib import Path

from util.yaml_util import load_yaml


def get_true_task_cfg_dict(
    task_cfg_dict: Dict, 
    prompt: str, 
    seed_list: List[int], 
    save_task_cfg_dict_root_path: str
) -> Dict:
    save_task_cfg_dict_root_path = Path(save_task_cfg_dict_root_path)

    tmp_task_cfg_dict = None
    if save_task_cfg_dict_root_path.is_file():
        task_cfg_dict_path = save_task_cfg_dict_root_path / "cfg.yaml"
        tmp_task_cfg_dict = load_yaml(task_cfg_dict_path)

    if tmp_task_cfg_dict is None:
        tmp_task_cfg_dict = copy.deepcopy(task_cfg_dict)
        tmp_task_cfg_dict["seed_list"] = copy.deepcopy(seed_list)
    else:
        for seed in task_cfg_dict["seed_list"]:
            tmp_task_cfg_dict["seed_list"].append(seed)
    tmp_task_cfg_dict["prompt"] = prompt
    
    return tmp_task_cfg_dict
    