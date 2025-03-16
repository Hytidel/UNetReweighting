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
from .util.skipping_strategy_util import (
    gen_strategy_skip_1, 
    gen_strategy_skip_2
)


# @torch.no_grad()
def run_cal_importance_ranking_implement(
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

    # ---------= [Eps] =---------
    logger(f"[Eps] Loading started. ")

    eps = get_true_value(cfg["task"]["eps"])

    logger(f"    eps: {eps}")

    logger(
        f"[Eps] Loading finished. "
        "\n"
    )

    # ---------= [Importance Score] =---------
    logger(f"[Importance Score] Loading started. ")

    save_importance_score = get_true_value(cfg["task"]["importance_score"]["save_importance_score"])
    save_prefix_importance_score = get_true_value(cfg["task"]["importance_score"]["save_prefix_importance_score"])

    logger(f"    save_importance_score: {save_importance_score}")
    logger(f"    save_prefix_importance_score: {save_prefix_importance_score}")

    logger(
        f"[Importance Score] Loading finished. "
        "\n"
    )

    # ---------= [Importance Ranking] =---------
    logger(f"[Importance Ranking] Loading started. ")

    save_importance_ranking = get_true_value(cfg["task"]["importance_ranking"]["save_importance_ranking"])
    save_prefix_importance_ranking = get_true_value(cfg["task"]["importance_ranking"]["save_prefix_importance_ranking"])

    logger(f"    save_importance_ranking: {save_importance_ranking}")
    logger(f"    save_prefix_importance_ranking: {save_prefix_importance_ranking}")

    logger(
        f"[Importance Ranking] Loading finished. "
        "\n"
    )

    # ---------= [Skipping Strategy] =---------
    logger(f"[Skipping Strategy] Loading started. ")

    save_strategy_skip_1 = get_true_value(cfg["task"]["skipping_stategy"]["save_strategy_skip_1"])
    save_strategy_skip_2 = get_true_value(cfg["task"]["skipping_stategy"]["save_strategy_skip_2"])

    logger(f"    save_strategy_skip_1: {save_strategy_skip_1}")
    logger(f"    save_strategy_skip_2: {save_strategy_skip_2}")

    logger(
        f"[Skipping Strategy] Loading finished. "
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

            save_importance_score = save_importance_score, 
            save_prefix_importance_score = save_prefix_importance_score, 

            save_importance_ranking = save_importance_ranking, 
            save_prefix_importance_ranking = save_prefix_importance_ranking, 

            save_strategy_skip_1 = save_strategy_skip_1, 
            save_strategy_skip_2 = save_strategy_skip_2
        ):
            folder_root_path_tuple_list.append((folder_idx, folder_root_path))

    num_folder = len(folder_root_path_tuple_list)
    tot_num_folder = len(folder_root_path_list)

    logger(f"    num_folder / tot_num_folder: {num_folder} / {tot_num_folder}")

    for (folder_idx, folder_root_path) in tqdm(
        folder_root_path_tuple_list, 

        desc = f"[Save Everything]"
    ):
        # ---------= [Load History Weight Threshold Matrix List] =---------
        weight_threshold_matrix_list_root_path = folder_root_path / "weight_threshold_matrix_list"
        round_path_list = [
            Path(round_path) \
                for round_path in weight_threshold_matrix_list_root_path.iterdir() \
                    if round_path.stem.startswith("round")
        ]

        history_weight_threshold_matrix_list_path_list = [
            round_path / "history_weight_threshold_matrix_list.yaml" \
                for round_path in round_path_list
        ]

        last_weight_threshold_matrix_list = [
            load_yaml(history_weight_threshold_matrix_list_path)["history_weight_threshold_matrix_list"][-1] \
                for history_weight_threshold_matrix_list_path in history_weight_threshold_matrix_list_path_list
        ]

        # last_weight_threshold_matrix_list.shape = (num_round, num_inference_step, num_attn_block)
        last_weight_threshold_matrix_list = np.asarray(last_weight_threshold_matrix_list)
        
        num_round, num_inference_step, num_attn_block = last_weight_threshold_matrix_list.shape

        # logger(f"    num_round: {num_round}")
        # logger(f"    num_inference_step: {num_inference_step}")
        # logger(f"    num_attn_block: {num_attn_block}")

        # ---------= [Voting] =---------
        # importance_score_matrix.shape = (num_inference_step, num_attn_block)
        importance_score_matrix = np.zeros(
            shape = (num_inference_step, num_attn_block), 
            dtype = np.int32
        )

        # prefix_importance_score_matrix_list.shape = (num_round, num_inference_step, num_attn_block)
        prefix_importance_score_matrix_list = []

        # prefix_importance_ranking_matrix_list.shape = (num_round, num_inference_step, num_attn_block)
        prefix_importance_ranking_matrix_list = []

        for round_idx, last_weight_threshold_matrix in enumerate(last_weight_threshold_matrix_list):
            # voting
            for step_idx in range(num_inference_step):
                # weight_threshold_list.shape = (num_attn_block, )
                weight_threshold_list = last_weight_threshold_matrix[step_idx]

                sorted_blk_idx_list = np.argsort(
                    -weight_threshold_list + np.random.uniform(
                        -eps, eps, 
                        size = weight_threshold_list.shape
                    )
                )

                for rk, blk_idx in enumerate(sorted_blk_idx_list):
                    importance_score_matrix[step_idx][blk_idx] += rk + 1

            prefix_importance_score_matrix_list.append(importance_score_matrix)

            # importance_ranking_matrix.shape = (num_inference_step, num_attn_block)
            importance_ranking_matrix = np.argsort(
                importance_score_matrix + np.random.uniform(
                    -eps, eps, 
                    size = importance_score_matrix.shape
                ), 
                axis = 1
            )

            prefix_importance_ranking_matrix_list.append(importance_ranking_matrix)
    
        with cf.ThreadPoolExecutor(
            max_workers = concurrent_max_worker
        ) as executor:
            # ---------= [Save Importance Score Matrix] =---------
            save_importance_score_matrix_root_path = folder_root_path / "importance_score_matrix"

            last_importance_score_matrix = prefix_importance_score_matrix_list[-1]
            yaml_dict = {"importance_score_matrix": last_importance_score_matrix}
            yaml_dict = convert_numpy_type_to_native_type(yaml_dict)

            executor.submit(
                save_yaml, 

                yaml_dict, 
                save_importance_score_matrix_root_path, 
                "importance_score_matrix.yaml"
            )

            if save_prefix_importance_score:
                yaml_dict = {"prefix_importance_score_matrix_list": prefix_importance_score_matrix_list}
                yaml_dict = convert_numpy_type_to_native_type(yaml_dict)

                executor.submit(
                    save_yaml, 

                    yaml_dict, 
                    save_importance_score_matrix_root_path, 
                    "prefix_importance_score_matrix_list.yaml"
                )

            # ---------= [Save Importance Ranking Matrix] =---------
            save_importance_ranking_matrix_root_path = folder_root_path / "importance_ranking_matrix"

            last_importance_ranking_matrix = prefix_importance_ranking_matrix_list[-1]
            yaml_dict = {"importance_ranking_matrix": last_importance_ranking_matrix}
            yaml_dict = convert_numpy_type_to_native_type(yaml_dict)

            executor.submit(
                save_yaml, 

                yaml_dict, 
                save_importance_ranking_matrix_root_path, 
                "importance_ranking_matrix.yaml"
            )

            if save_prefix_importance_ranking:
                yaml_dict = {"prefix_importance_ranking_matrix_list": prefix_importance_ranking_matrix_list}
                yaml_dict = convert_numpy_type_to_native_type(yaml_dict)

                executor.submit(
                    save_yaml, 

                    yaml_dict, 
                    save_importance_ranking_matrix_root_path, 
                    "prefix_importance_ranking_matrix_list.yaml"
                )

            # ---------= [Get Skipping Strategy] =---------
            save_strategy_root_path = folder_root_path / "skipping_strategy"
            if save_strategy_skip_1:
                save_strategy_skip_1_root_path = save_strategy_root_path / "skip-1"

                strategy_list, duplicate_list = gen_strategy_skip_1(
                    importance_ranking_matrix = last_importance_ranking_matrix, 

                    num_inference_step = num_inference_step, 
                    num_attn_block = num_attn_block
                )

                for strategy_idx, (strategy, duplicate) in enumerate(
                    zip(strategy_list, duplicate_list)
                ):
                    strategy *= 1.5

                    yaml_dict = {
                        "history_weight_threshold_matrix_list": strategy, 
                        "duplicate": duplicate
                    }

                    yaml_dict = convert_numpy_type_to_native_type(yaml_dict)

                    executor.submit(
                        save_yaml, 

                        yaml_dict, 
                        save_strategy_skip_1_root_path, 
                        f"skip-1_{strategy_idx}.yaml"
                    )

            if save_strategy_skip_2:
                save_strategy_skip_2_root_path = save_strategy_root_path / "skip-2"

                strategy_list, duplicate_list = gen_strategy_skip_2(
                    importance_ranking_matrix = last_importance_ranking_matrix, 

                    num_inference_step = num_inference_step, 
                    num_attn_block = num_attn_block
                )

                for strategy_idx, (strategy, duplicate) in enumerate(
                    zip(strategy_list, duplicate_list)
                ):
                    strategy *= 1.5
                    
                    yaml_dict = {
                        "history_weight_threshold_matrix_list": strategy, 
                        "duplicate": duplicate
                    }

                    yaml_dict = convert_numpy_type_to_native_type(yaml_dict)

                    executor.submit(
                        save_yaml, 

                        yaml_dict, 
                        save_strategy_skip_2_root_path, 
                        f"skip-2_{strategy_idx}.yaml"
                    )

    # clean up
    gc.collect()

    # `run_cal_importance_ranking_implement()` done
    pass

def run_cal_importance_ranking(
    cfg: DictConfig
):
    run_cal_importance_ranking_implement(cfg)

    pass
