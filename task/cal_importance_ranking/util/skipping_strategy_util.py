from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import numpy as np

import copy


def check_duplicate(
    history_strategy_skipping_matrix_list: List[np.ndarray], 
    strategy_skipping_matrix: np.ndarray
) -> bool:
    for history_strategy_skipping_matrix in history_strategy_skipping_matrix_list:
        if np.equal(strategy_skipping_matrix, history_strategy_skipping_matrix).all():
            return True
    
    return False

def gen_strategy_skip_1(
    # importance_ranking_matrix.shape = (num_inference_step, num_attn_block)
    importance_ranking_matrix: np.ndarray, 

    num_inference_step: int, 
    num_attn_block: int
) -> Tuple[List[np.ndarray], List[int]]:
    history_strategy_skipping_matrix_list = []

    # baseline
    for blk_idx in range(num_attn_block):
        strategy_skipping_matrix = np.zeros(
            shape = (num_inference_step, num_attn_block), 
            dtype = np.int8
        )

        strategy_skipping_matrix[:, blk_idx] = 1

        history_strategy_skipping_matrix_list.append(strategy_skipping_matrix)

    # strategy_list.shape = (num_strategy, 1, num_inference_step, num_attn_block)
    strategy_list = []
    duplicate_list = []

    # strategy_skipping_matrix.shape = (num_inference_step, num_attn_block)
    strategy_skipping_matrix = np.zeros(
        shape = (1, num_inference_step, num_attn_block), 
        dtype = np.int8
    )

    def dfs(
        step_idx: int
    ):
        if step_idx >= num_inference_step:
            tmp_strategy = history_weight_threshold_matrix_list = np.ones(
                shape = (1, num_inference_step, num_attn_block), 
                dtype = np.float16
            ) * strategy_skipping_matrix
            strategy_list.append(tmp_strategy)

            if check_duplicate(
                history_strategy_skipping_matrix_list = history_strategy_skipping_matrix_list, 
                strategy_skipping_matrix = strategy_skipping_matrix
            ):
                duplicate_list.append(1)
            else:
                duplicate_list.append(0)
                history_strategy_skipping_matrix_list.append(
                    copy.deepcopy(strategy_skipping_matrix)
                )

            return
        
        for state in range(2):
            # `state = 0` to skip the 0-th blk
            if state == 0:
                skip_blk_idx = importance_ranking_matrix[step_idx][0]
            # `state = 1` to skip the 1-st blk
            elif state == 1:
                skip_blk_idx = importance_ranking_matrix[step_idx][1]

            strategy_skipping_matrix[:, step_idx, skip_blk_idx] = 1

            dfs(step_idx + 1)

            strategy_skipping_matrix[:, step_idx, skip_blk_idx] = 0

    dfs(step_idx = 0)

    return strategy_list, duplicate_list

def gen_strategy_skip_2(
    # importance_ranking_matrix.shape = (num_inference_step, num_attn_block)
    importance_ranking_matrix: np.ndarray, 

    num_inference_step: int, 
    num_attn_block: int
) -> Tuple[List[np.ndarray], List[int]]:
    history_strategy_skipping_matrix_list = []

    # baseline
    for floor_blk in range(num_attn_block // 2):
        strategy_skipping_matrix = np.zeros(
            shape = (num_inference_step, num_attn_block), 
            dtype = np.int8
        )

        strategy_skipping_matrix[:, floor_blk] = 1
        strategy_skipping_matrix[:, (num_attn_block - floor_blk - 1)] = 1

        history_strategy_skipping_matrix_list.append(strategy_skipping_matrix)

    # strategy_list.shape = (num_strategy, 1, num_inference_step, num_attn_block)
    strategy_list = []
    duplicate_list = []

    # strategy_skipping_matrix.shape = (num_inference_step, num_attn_block)
    strategy_skipping_matrix = np.zeros(
        shape = (1, num_inference_step, num_attn_block), 
        dtype = np.int8
    )

    def dfs(
        step_idx: int
    ):
        if step_idx >= num_inference_step:
            tmp_strategy = history_weight_threshold_matrix_list = np.ones(
                shape = (1, num_inference_step, num_attn_block), 
                dtype = np.float16
            ) * strategy_skipping_matrix
            strategy_list.append(tmp_strategy)

            if check_duplicate(
                history_strategy_skipping_matrix_list = history_strategy_skipping_matrix_list, 
                strategy_skipping_matrix = strategy_skipping_matrix
            ):
                duplicate_list.append(1)
            else:
                duplicate_list.append(0)
                history_strategy_skipping_matrix_list.append(
                    copy.deepcopy(strategy_skipping_matrix)
                )

            return
        
        for state in range(3):
            skip_blk_idx_list = []

            # `state = 0` to skip the 0-th, 1-st blk
            if state == 0:
                skip_blk_idx_list.append(importance_ranking_matrix[step_idx][0])
                skip_blk_idx_list.append(importance_ranking_matrix[step_idx][1])
            # `state = 1` to skip the 0-th, 2-nd blk
            elif state == 1:
                skip_blk_idx_list.append(importance_ranking_matrix[step_idx][0])
                skip_blk_idx_list.append(importance_ranking_matrix[step_idx][2])
            # `state = 2` to skip the 1-st, 2-nd blk
            elif state == 2:
                skip_blk_idx_list.append(importance_ranking_matrix[step_idx][1])
                skip_blk_idx_list.append(importance_ranking_matrix[step_idx][2])

            for skip_blk_idx in skip_blk_idx_list:
                strategy_skipping_matrix[:, step_idx, skip_blk_idx] = 1

            dfs(step_idx + 1)

            for skip_blk_idx in skip_blk_idx_list:
                strategy_skipping_matrix[:, step_idx, skip_blk_idx] = 0

    dfs(step_idx = 0)

    return strategy_list, duplicate_list
