from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

from functools import partial

import numpy as np

from tqdm.auto import tqdm

import gc

from pathlib import Path

import concurrent.futures as cf

from util.basic_util import (
    pause, 
    get_global_variable, 
    is_none, 
    get_true_value, 
    get_timestamp
)
from util.yaml_util import (
    load_yaml, save_yaml, 
    convert_numpy_type_to_native_type
)
from util.numpy_util import tsfm_to_2d_matrix, cal_matrix_ranking

from .util.save_sample_util import is_data_exist


# @torch.no_grad()
def run_cal_importance_weight_implement(
    cfg: DictConfig
):
    # ---------= [Basic Global Variables] =---------
    exp_name = get_global_variable("exp_name")
    start_time = get_global_variable("start_time")
    device = get_global_variable("device")
    seed = get_global_variable("seed")
    exp_time_str = f"{exp_name}_{start_time}"

    concurrent_max_worker = get_global_variable("concurrent_max_worker")

    # ---------= [Category Root Path] =---------
    logger(f"[Category Root Path] Loading started. ")

    category_root_path = get_true_value(cfg["task"]["category_root_path"])
    category_root_path = Path(category_root_path)

    logger(f"    category_root_path: {category_root_path}")

    logger(
        f"[Category Root Path] Loading finished. "
        "\n"
    )

    # ---------= [Vote Num Round] =---------
    logger(f"[Vote Num Round] Loading started. ")

    vote_num_round = get_true_value(cfg["task"]["vote_num_round"])

    logger(f"    vote_num_round: {vote_num_round}")

    logger(
        f"[Vote Num Round] Loading finished. "
        "\n"
    )

    # ---------= [Reverse Ranking] =---------
    logger(f"[Reverse Ranking] Loading started. ")

    reverse_ranking = get_true_value(cfg["task"]["reverse_ranking"])

    logger(f"    reverse_ranking: {reverse_ranking}")

    logger(
        f"[Reverse Ranking] Loading finished. "
        "\n"
    )

    # ---------= [Importance Weight Matrix] =---------
    logger(f"[Importance Weight Matrix] Loading started. ")

    save_importance_weight_matrix = get_true_value(cfg["task"]["importance_weight_matrix"]["save_importance_weight_matrix"])

    logger(f"    save_importance_weight_matrix: {save_importance_weight_matrix}")

    logger(
        f"[Importance Weight Matrix] Loading finished. "
        "\n"
    )

    # ---------= [Importance Weight] =---------
    logger(f"[Importance Weight] Loading started. ")

    low_importance_weight = get_true_value(cfg["task"]["importance_weight"]["low_importance_weight"])
    high_importance_weight = get_true_value(cfg["task"]["importance_weight"]["high_importance_weight"])

    logger(f"    low_importance_weight: {low_importance_weight}")
    logger(f"    high_importance_weight: {high_importance_weight}")

    weight_range_str = f"{low_importance_weight}_{high_importance_weight}"
    if reverse_ranking:
        weight_range_str = f"{weight_range_str}_rev"

    logger(
        f"[Importance Weight] Loading finished. "
        "\n"
    )

    # ---------= [Importance Weight Matrix] =---------
    logger(f"[Importance Weight Matrix] Loading started. ")

    save_importance_weight_matrix = get_true_value(cfg["task"]["importance_weight_matrix"]["save_importance_weight_matrix"])

    logger(f"    save_importance_weight_matrix: {save_importance_weight_matrix}")

    logger(
        f"[Importance Weight Matrix] Loading finished. "
        "\n"
    )

    # ---------= [All Components Loaded] =---------
    logger(
        f"All components loaded. "
        "\n"
    )

    # ---------= [Load History Weight Threshold Matrix List] =---------
    folder_root_path_list = [_ for _ in category_root_path.iterdir()]

    folder_root_path_tuple_list = []  # (folder_idx, folder_root_path)

    for folder_idx, folder_root_path in enumerate(
        tqdm(
            folder_root_path_list, 
            
            desc = f"[Check Data Existence]"
        )
    ):
        if not is_data_exist(
            folder_root_path = folder_root_path, 

            weight_range_str = weight_range_str
        ):
            folder_root_path_tuple_list.append((folder_idx, folder_root_path))

    num_folder = len(folder_root_path_tuple_list)
    tot_num_folder = len(folder_root_path_list)

    logger(f"    num_folder / tot_num_folder: {num_folder} / {tot_num_folder}")

    for (folder_idx, folder_root_path) in tqdm(
        folder_root_path_tuple_list, 

        desc = f"[Save Everything]"
    ):
        # ---------= [Cal Importance Weight Matrix] =---------
        importance_score_matrix_path = folder_root_path / "importance_score_matrix" / "importance_score_matrix.yaml"
        importance_score_matrix = load_yaml(importance_score_matrix_path)["importance_score_matrix"]

        # importance_score_matrix.shape = (num_inference_step, num_attn_block)
        importance_score_matrix = np.asarray(importance_score_matrix)
        num_inference_step, num_attn_block = importance_score_matrix.shape

        max_score = vote_num_round * num_attn_block

        # importance_weight_matrix.shape = (num_inference_step, num_attn_block)
        importance_weight_matrix = np.zeros(
            shape = (num_inference_step, num_attn_block), 
            dtype = np.float32
        )

        for step_idx, importance_score_list in enumerate(importance_score_matrix):
            # min_val = np.min(importance_score_list)
            # max_val = np.max(importance_score_list)

            # if np.isclose(min_val, max_val):
            #     normed_importance_score_list = np.ones(
            #         shape = (num_attn_block, ), 
            #         dtype = np.float32
            #     )
            # else:
            #     normed_importance_score_list = (importance_score_list - min_val) / (max_val - min_val)

            if reverse_ranking:
                normed_importance_score_list = (max_score - importance_score_list) / max_score
            else:
                normed_importance_score_list = importance_score_list / max_score
            
            importance_weight_matrix[step_idx] \
                = normed_importance_score_list * (high_importance_weight - low_importance_weight) + low_importance_weight

        # logger(f"importance_weight_matrix: {importance_weight_matrix}")

        save_importance_weight_matrix_root_path = folder_root_path / "importance_weight_matrix"
        if save_importance_weight_matrix:
            tmp_dict = {"importance_weight_matrix": importance_weight_matrix}
            tmp_dict = convert_numpy_type_to_native_type(tmp_dict)
            
            save_yaml(
                tmp_dict, 
                yaml_root_path = save_importance_weight_matrix_root_path, 
                yaml_filename = f"{weight_range_str}.yaml"
            )

    # clean up
    gc.collect()

    # `run_cal_importance_weight_implement()` done
    pass

def run_cal_importance_weight(
    cfg: DictConfig
):
    run_cal_importance_weight_implement(cfg)

    pass
