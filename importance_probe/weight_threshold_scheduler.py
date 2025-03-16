from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import numpy as np

import copy

from util.yaml_util import (
    save_yaml, 
    convert_numpy_type_to_native_type
)
from util.plot_util import (
    get_line_chart, 
    merge_line_chart_list
)
from util.numpy_util import (
    tsfm_to_1d_array, 
    tsfm_to_2d_matrix
)


class WeightThresholdScheduler():
    def __init__(
        self, 
        num_inference_step: int, 
        num_weight_threshold: int, 
        init_weight_threshold: Optional[Union[float, List[float], List[List[float]]]] = 0.0, 

        min_weight_threshold: Optional[Union[float, List[float]]] = 0.0, 
        max_weight_threshold: Optional[Union[float, List[float]]] = 1.0, 
    ):
        self.num_inference_step = num_inference_step
        self.num_weight_threshold = num_weight_threshold

        min_weight_threshold_list, max_weight_threshold_list = self._preprocess_input(
            num_inference_step = num_inference_step, 
            num_weight_threshold = num_weight_threshold, 
            min_weight_threshold = min_weight_threshold, 
            max_weight_threshold = max_weight_threshold, 
        )

        self.min_weight_list = min_weight_threshold_list
        self.max_weight_list = max_weight_threshold_list

        self.history_weight_threshold_matrix_list = []

    def _preprocess_input(
        self, 
        num_inference_step: int, 
        num_weight_threshold: int, 
        min_weight_threshold: Union[float, List[float]], 
        max_weight_threshold: Union[float, List[float]], 
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        min_weight_threshold_list = tsfm_to_1d_array(
            array = min_weight_threshold, 
            target_length = num_weight_threshold
        )

        max_weight_threshold_list = tsfm_to_1d_array(
            array = max_weight_threshold, 
            target_length = num_weight_threshold
        )

        return min_weight_threshold_list, max_weight_threshold_list
    
    def init(
        self, 

        init_weight_threshold: Optional[Union[float, List[float], List[List[float]]]] = 0.0
    ):
        # init_weight_threshold_matrix.shape = (num_inference_step, num_weight_threshold)
        init_weight_threshold_matrix = tsfm_to_2d_matrix(
            matrix = init_weight_threshold, 
            target_shape = (self.num_inference_step, self.num_weight_threshold)
        )
        self.weight_threshold_matrix = init_weight_threshold_matrix

        del self.history_weight_threshold_matrix_list

        # history_weight_threshold_matrix_list.shape = (num_epoch + 1, num_inference_step, num_weight_threshold)
        self.history_weight_threshold_matrix_list = []

        # initialize `weight_threshold_scheduler`
        self._save_history_weight_threshold_matrix()

    def _save_history_weight_threshold_matrix(
        self, 
        weight_threshold_matrix: Optional[np.ndarray] = None
    ):
        if weight_threshold_matrix is None:
            weight_threshold_matrix = self.weight_threshold_matrix
        
        self.history_weight_threshold_matrix_list.append(
            copy.deepcopy(weight_threshold_matrix)
        )

    def get_weight_threshold_matrix(
        self, 
        history_idx: Optional[int] = None
    ) -> np.ndarray:
        if history_idx is None:
            history_idx = -1
        
        # weight_threshold_matrix.shape = (num_inference_step, num_weight_threshold)
        weight_threshold_matrix = copy.deepcopy(self.history_weight_threshold_matrix_list[history_idx])

        return weight_threshold_matrix

    def update_weight_threshold_matrix(
        self, 
        weight_threshold_matrix: np.ndarray, 
        update_strategy: Optional[str] = "hard",  # ["hard", "soft", "probability"]
        weight_eps: Optional[float] = 1e-2, 
        inference_step_sample_accepted_mask: Union[List[bool], np.ndarray] = None, 
        
        # used when `update_strategy = "soft"` 
        epoch_idx: Optional[int] = None, 
        num_epoch: Optional[int] = None, 

        # used_when `update_strategy = probability`
        sample_accepted_prob_dict: Optional[Dict] = None, 
        sample_rejected_prob_dict: Optional[Dict] = None, 
    ):
        inference_step_sample_accepted_mask = tsfm_to_1d_array(
            array = inference_step_sample_accepted_mask, 
            target_length = weight_threshold_matrix.shape[0]
        )

        if not (update_strategy in ["hard", "soft", "probability"]):
            raise NotImplementedError(
                f"Unsupported update_strategy: `{update_strategy}`. "
            )

        if update_strategy == "soft":
            if (epoch_idx is None) or (num_epoch is None):
                raise ValueError(
                    f"Must provide `epoch_idx` and `num_epoch` when  `update_strategy = \"soft\"`. "
                )
        if update_strategy == "probability":
            if (sample_accepted_prob_dict is None) or (sample_rejected_prob_dict is None):
                raise ValueError(
                    f"Must provide `sample_accepted_prob_dict` and `sample_rejected_prob_dict` when  `update_strategy = \"probability\"`. "
                )

        original_weight_threshold_matrix = copy.deepcopy(self.weight_threshold_matrix)
        target_weight_threshold_matrix = weight_threshold_matrix - weight_eps

        new_weight_threshold_matrix = copy.deepcopy(weight_threshold_matrix)
        
        if update_strategy == "hard":
            # ---------= [Sample Accepted] =---------
            new_weight_threshold_matrix[inference_step_sample_accepted_mask] \
                = target_weight_threshold_matrix[inference_step_sample_accepted_mask]
            
            # ---------= [Sample Rejected] =---------
            new_weight_threshold_matrix[~inference_step_sample_accepted_mask] \
                = 1.0 / 2 * original_weight_threshold_matrix[~inference_step_sample_accepted_mask]
        elif update_strategy == "soft":
            # ---------= [Sample Accepted] =---------
            diff_weight_threshold_matrix \
                = target_weight_threshold_matrix[inference_step_sample_accepted_mask] \
                    - original_weight_threshold_matrix[inference_step_sample_accepted_mask]
            new_weight_threshold_matrix[inference_step_sample_accepted_mask] \
                = original_weight_threshold_matrix[inference_step_sample_accepted_mask] \
                    + (1 - (epoch_idx + 1) / num_epoch) * diff_weight_threshold_matrix
            
            # ---------= [Sample Rejected] =---------
            new_weight_threshold_matrix[~inference_step_sample_accepted_mask] \
                = (epoch_idx + 1) / num_epoch * original_weight_threshold_matrix[~inference_step_sample_accepted_mask]
        elif update_strategy == "probability":
            # ---------= [Sample Accepted] =---------
            p_threshold_high = sample_accepted_prob_dict["threshold_high"]
            p_threshold_moderate_st = sample_accepted_prob_dict["threshold_moderate_st"]
            p_threshold_moderate_ed = sample_accepted_prob_dict["threshold_moderate_ed"]
            p_threshold_moderate \
                = p_threshold_moderate_st \
                    + (epoch_idx + 1) / num_epoch * (p_threshold_moderate_ed - p_threshold_moderate_st)
            p_threshold_low_st = sample_accepted_prob_dict["threshold_low_st"]
            p_threshold_low_ed = sample_accepted_prob_dict["threshold_low_ed"]
            p_threshold_low \
                = p_threshold_low_st - (epoch_idx + 1) / num_epoch * (p_threshold_low_st - p_threshold_low_ed)

            # check probability
            p_sum = p_threshold_high + p_threshold_moderate + p_threshold_low
            if not np.isclose(p_sum, 1.0):
                raise ValueError(
                    f"`p_threshold_high + p_threshold_moderate + p_threshold_low` doesn't equal to 1.0, "
                    f"got {p_sum}. "
                )

            new_weight_threshold_matrix[inference_step_sample_accepted_mask] \
                = p_threshold_high * (1.0 / 2 * original_weight_threshold_matrix[inference_step_sample_accepted_mask]) \
                    + p_threshold_moderate * original_weight_threshold_matrix[inference_step_sample_accepted_mask] \
                    + p_threshold_low * target_weight_threshold_matrix[inference_step_sample_accepted_mask]

            # ---------= [Sample Rejected] =---------
            p_threshold_high_st = sample_rejected_prob_dict["threshold_high_st"]
            p_threshold_high_ed = sample_rejected_prob_dict["threshold_high_ed"]
            p_threshold_high \
                = p_threshold_high_st - (epoch_idx + 1) / num_epoch * (p_threshold_high_st - p_threshold_high_ed)
            p_threshold_moderate_st = sample_rejected_prob_dict["threshold_moderate_st"]
            p_threshold_moderate_ed = sample_rejected_prob_dict["threshold_moderate_ed"]
            p_threshold_moderate \
                = p_threshold_moderate_st \
                    + (epoch_idx + 1) / num_epoch * (p_threshold_moderate_ed - p_threshold_moderate_st)
            p_threshold_low = sample_rejected_prob_dict["threshold_low"]

            # check probability
            p_sum = p_threshold_high + p_threshold_moderate + p_threshold_low
            if not np.isclose(p_sum, 1.0):
                raise ValueError(
                    f"`p_threshold_high + p_threshold_moderate + p_threshold_low` doesn't equal to 1.0, "
                    f"got {p_sum}. "
                )

            new_weight_threshold_matrix[~inference_step_sample_accepted_mask] \
                = p_threshold_high * (1.0 / 2 * original_weight_threshold_matrix[~inference_step_sample_accepted_mask]) \
                    + p_threshold_moderate * original_weight_threshold_matrix[~inference_step_sample_accepted_mask] \
                    + p_threshold_low * target_weight_threshold_matrix[~inference_step_sample_accepted_mask]
        
        self.weight_threshold_matrix = np.clip(
            new_weight_threshold_matrix.copy(), 
            self.min_weight_list, self.max_weight_list
        )

    def get_passable_weight_matrix(
        self, 
        weight_eps: Optional[float] = 1e-2
    ) -> np.ndarray:
        passable_weight_matrix = np.clip(
            self.weight_threshold_matrix - weight_eps, 
            self.min_weight_list, self.max_weight_list
        )

        return passable_weight_matrix

    def _save_last_weight_threshold_matrix_as_yaml(
        self, 
        yaml_root_path, 
        yaml_filename
    ):
        last_weight_threshold_matrix = self.history_weight_threshold_matrix_list[-1]

        yaml_dict = {"weight_threshold_matrix": last_weight_threshold_matrix}
        
        yaml_dict = convert_numpy_type_to_native_type(yaml_dict)
        save_yaml(
            cfg = yaml_dict, 
            yaml_root_path = yaml_root_path, 
            yaml_filename = yaml_filename
        )
    
    def _save_history_weight_threshold_matrix_list_as_yaml(
        self, 
        yaml_root_path, 
        yaml_filename
    ):
        history_weight_threshold_matrix_list = copy.deepcopy(self.history_weight_threshold_matrix_list)

        yaml_dict = {"history_weight_threshold_matrix_list": history_weight_threshold_matrix_list}

        yaml_dict = convert_numpy_type_to_native_type(yaml_dict)
        save_yaml(
            cfg = yaml_dict, 
            yaml_root_path = yaml_root_path, 
            yaml_filename = yaml_filename
        )

    @classmethod
    def tsfm_history_weight_threshold_matrix_list(
        cls, 
        history_weight_threshold_matrix_list: Union[List[List[List[float]]], List[np.ndarray], np.ndarray]
    ) -> List[np.ndarray]:
        # history_weight_threshold_matrix_list.shape = (num_ver, num_inference_step, num_weight_threshold)
        history_weight_threshold_matrix_list = copy.deepcopy(history_weight_threshold_matrix_list)
        history_weight_threshold_matrix_list = np.asarray(history_weight_threshold_matrix_list)

        if len(history_weight_threshold_matrix_list.shape) != 3:
            raise ValueError(
                f"The shape of `history_weight_threshold_matrix_list` doesn't match "
                f"(num_ver, num_inference_step, num_weight_threshold), got {history_weight_threshold_matrix_list.shape}. "
            )

        num_ver = history_weight_threshold_matrix_list.shape[0]
        num_inference_step = history_weight_threshold_matrix_list.shape[1]
        num_weight_threshold = history_weight_threshold_matrix_list.shape[2]

        # inference_step_weight_threshold_matrix_list.shape = (num_inference_step, num_weight_threshold, num_ver)
        inference_step_weight_threshold_matrix_list = []
        for inference_step_idx in range(num_inference_step):
            # inference_step_importance_matrix.shape = (num_weight_threshold, num_ver)
            inference_step_importance_matrix = np.zeros(shape = (num_weight_threshold, num_ver))
            for weight_threshold_idx in range(num_weight_threshold):
                for ver_idx in range(num_ver):
                    inference_step_importance_matrix[weight_threshold_idx][ver_idx] \
                        = history_weight_threshold_matrix_list[ver_idx][inference_step_idx][weight_threshold_idx]
                    
            inference_step_weight_threshold_matrix_list.append(inference_step_importance_matrix)

        return inference_step_weight_threshold_matrix_list
    
    @classmethod
    def get_inference_step_chart_list(
        cls, 
        inference_step_weight_threshold_matrix_list: List[np.ndarray], 
        figsize: Optional[Union[float, Tuple[float, float]]] = (10, 8), 
        marker_list: Union[str, List[str]] = None, 
    ) -> List[Tuple]:
        # inference_step_weight_threshold_matrix_list.shape = (num_inference_step, num_weight_threshold, num_ver)
        num_inference_step = len(inference_step_weight_threshold_matrix_list)
        if len(inference_step_weight_threshold_matrix_list[0].shape) != 2:
            raise ValueError(
                f"The shape of `inference_step_weight_threshold_matrix_list` doesn't match "
                f"(num_inference_step, num_weight_threshold, num_ver), "
                f"got ({num_inference_step}, {inference_step_weight_threshold_matrix_list.shape[0]}, {inference_step_weight_threshold_matrix_list.shape[1]}). "
            )
        else:
            num_weight_threshold = inference_step_weight_threshold_matrix_list[0].shape[0]  # num_y_list_list
            num_ver = inference_step_weight_threshold_matrix_list[0].shape[1]  # num_x_list
        
        x_list = [i for i in range(num_ver)]

        inference_step_chart_list = []
        for inference_step_idx in range(num_inference_step):
            y_list_list = [
                inference_step_weight_threshold_matrix_list[inference_step_idx][weight_threshold_idx] \
                    for weight_threshold_idx in range(num_weight_threshold)
            ]

            fig, ax = get_line_chart(
                figsize = figsize, 

                x_list = x_list, y_list_list = y_list_list, 
                y_label_list = [f"blk-{i}" for i in range(num_weight_threshold)], 

                marker_list = marker_list, 

                plot_title = f"step-{inference_step_idx}", 
                plot_x_label = "epoch", plot_y_label = "weight_threshold"
            )

            inference_step_chart_list.append((fig, ax))

        return inference_step_chart_list
