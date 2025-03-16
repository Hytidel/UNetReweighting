from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import numpy as np

from util.numpy_util import tsfm_to_1d_array

from importance_probe.energy_function import get_energy_func


class WeightScalingScheduler():
    def __init__(
        self, 
        num_weight: int, 
        init_weight: Optional[Union[float, List[float], np.ndarray]] = 1.0, 

        min_weight: Optional[Union[float, List[float]]] = 0.0, 
        max_weight: Optional[Union[float, List[float]]] = 1.0, 

        energy_func: Optional[str] = "quadratic_sum", 
    ):
        self.num_weight = num_weight

        init_weight_list, \
            min_weight_list, max_weight_list = self._preprocess_input(
                num_weight = num_weight, 
                init_weight = init_weight, 
                min_weight = min_weight, 
                max_weight = max_weight
            )

        self.weight_list = init_weight_list
        self.min_weight_list = min_weight_list
        self.max_weight_list = max_weight_list

        self.energy_func = get_energy_func(energy_func_name = energy_func)

    def _preprocess_input(
        self, 
        num_weight: int, 
        init_weight: Union[float, List[float], np.ndarray], 
        min_weight: Union[float, List[float]], 
        max_weight: Union[float, List[float]], 
    ) -> Tuple[List[float], List[float], List[float]]:
        init_weight_list = tsfm_to_1d_array(
            array = init_weight, 
            target_length = num_weight
        )

        min_weight_list = tsfm_to_1d_array(
            array = min_weight, 
            target_length = num_weight
        )

        max_weight_list = tsfm_to_1d_array(
            array = max_weight, 
            target_length = num_weight
        )

        return init_weight_list, \
            min_weight_list, max_weight_list

    def step_bias(
        self, 
        bias: Union[float, List[float], np.ndarray], 
    ) -> np.ndarray:
        bias_list = tsfm_to_1d_array(
            array = bias, 
            target_length = self.num_weight
        )

        new_weight_list = np.clip(
            self.weight_list + bias_list, 
            self.min_weight_list, self.max_weight_list
        )

        return new_weight_list

    def set_weight(
        self, 
        weight: Union[float, List[float], np.ndarray]
    ):
        weight_list = tsfm_to_1d_array(
            array = weight, 
            target_length = self.num_weight
        )
        
        self.weight_list = np.clip(
            weight_list, 
            self.min_weight_list, self.max_weight_list
        )

    @property
    def energy(
        self,
    ):
        return self.energy_func(
            num_weight = self.num_weight, 
            weight_list = self.weight_list
        )

    def get_random_uniform_bias(
        self, 
        max_bias: Union[float, List[float]], 
        min_bias: Optional[Union[float, List[float]]] = None, 
        energy_drop: bool = True
    ) -> np.ndarray:
        max_bias_list = tsfm_to_1d_array(
            array = max_bias, 
            target_length = self.num_weight
        )

        if min_bias is None:
            min_bias = -max_bias

        min_bias_list = tsfm_to_1d_array(
            array = min_bias, 
            target_length = self.num_weight
        )

        # adjust `max_bias_list` and `min_bias_list` to ensure energy drop
        if energy_drop:
            init_energy = self.energy
            abs_bias_lim = np.sqrt(init_energy / self.num_weight)

            min_bias_list = np.maximum(
                min_bias_list, 
                -abs_bias_lim
            )
            max_bias_list = np.minimum(
                max_bias_list, 
                abs_bias_lim
            )

            for i in range(self.num_weight):
                if self.weight_list[i] >= 0:
                    max_bias_list[i] = np.min(max_bias_list[i], 0)
                elif self.weight_list[i] < 0:  # unreachable
                    min_bias_list[i] = np.max(min_bias_list[i], 0)

        bias_list = np.random.uniform(
            low = min_bias_list, high = max_bias_list, 
            size = (self.num_weight, )
        )

        return bias_list

    def step(
        self, 
        max_bias: Union[float, List[float]], 
        min_bias: Optional[Union[float, List[float]]] = None, 
        energy_drop: bool = True
    ) -> np.ndarray:
        bias_list = self.get_random_uniform_bias(
            max_bias = max_bias, min_bias = min_bias, 
            energy_drop = energy_drop
        )
        
        return self.step_bias(bias = bias_list)

    def get_random_uniform_weight_list(
        self, 
        min_weight: Optional[float] = 0.0, 
        max_weight: Optional[float] = 1.0
    ) -> np.ndarray:
        weight_list = np.random.uniform(
            low = min_weight, high = max_weight, 
            size = (self.num_weight, )
        )

        weight_list = np.clip(
            weight_list, 
            self.min_weight_list, self.max_weight_list
        )

        return weight_list
