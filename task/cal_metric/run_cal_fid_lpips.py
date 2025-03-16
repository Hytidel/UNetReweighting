from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

from pathlib import Path

import torch

import gc

from util.basic_util import pause, get_global_variable, is_none, get_true_value
from util.image_util import (
    load_img_path, 
    img_pil_to_tensor
)
from util.yaml_util import load_yaml, save_yaml, convert_numpy_type_to_native_type

from util.metric_util.fid_util import cal_fid
from util.metric_util.lpips_util import cal_lpips


# @torch.no_grad()
def run_cal_fid_lpips_implement(
    cfg: DictConfig
):
    # ---------= [Basic Global Variables] =---------
    exp_name = get_global_variable("exp_name")
    start_time = get_global_variable("start_time")
    device = get_global_variable("device")
    seed = get_global_variable("seed")
    exp_time_str = f"{exp_name}_{start_time}"

    concurrent_max_worker = get_global_variable("concurrent_max_worker")

    # ---------= [Misc] =---------
    logger(f"[Misc] Loading started. ")

    sd_type = get_true_value(cfg["task"]["sd_type"])

    logger(f"    sd_type: {sd_type}")

    split = get_true_value(cfg["task"]["split"])

    logger(f"    split: {split}")

    logger(
        f"[Misc] Loading finished. "
        "\n"
    )

    # ---------= [Image] =---------
    logger(f"[Image] Loading started. ")

    folder_root_path = get_true_value(cfg["task"]["img"]["folder_root_path"])
    folder_root_path = Path(folder_root_path)

    logger(f"    folder_root_path: {folder_root_path}")

    img_size = get_true_value(cfg["task"]["img"]["img_size"])

    logger(f"    img_size: {img_size}")

    logger(
        f"[Image] Loading finished. "
        "\n"
    )

    # ---------= [FID] =---------
    logger(f"[FID] Loading started. ")

    fid_feature_dim = get_true_value(cfg["task"]["fid"]["feature_dim"])
    fid_batch_size = get_true_value(cfg["task"]["fid"]["batch_size"])
    fid_num_worker = get_true_value(cfg["task"]["fid"]["num_worker"])

    logger(f"    fid_feature_dim: {fid_feature_dim}")
    logger(f"    fid_batch_size: {fid_batch_size}")
    logger(f"    fid_num_worker: {fid_num_worker}")

    logger(
        f"[FID] Loading finished. "
        "\n"
    )

    # ---------= [LPIPS] =---------
    logger(f"[LPIPS] Loading started. ")

    lpips_model_net = get_true_value(cfg["task"]["lpips"]["model_net"])

    logger(f"    lpips_model_net: {lpips_model_net}")

    logger(
        f"[LPIPS] Loading finished. "
        "\n"
    )

    # ---------= [All Components Loaded] =---------
    logger(
        f"All components loaded. "
        "\n"
    )

    # ---------= [Preprocess] =---------
    ref_img_root_path = folder_root_path / "default_default"
    ref_img_path_list = list(
        ref_img_path for ref_img_path in ref_img_root_path.iterdir() \
            if ref_img_path.suffix in [".png"]
    )
    ref_img_pil_list = [
        load_img_path(ref_img_path) \
            for ref_img_path in ref_img_path_list
    ]

    num_img = len(ref_img_pil_list)
    num_batch = (num_img + fid_batch_size - 1) // fid_batch_size

    folder_path_list = list(
        folder_path for folder_path in folder_root_path.iterdir() \
            if folder_path.stem != "default_default"
    )

    strategy_name_dict = {
        "static-0,6": "c.0", 
        "static-1,5": "c.1", 
        "static-2,4": "c.2"
    }

    def get_strategy_name(
        folder_name: str
    ) -> str:
        weight_threshold_matrix_name = folder_name[8: ]

        if weight_threshold_matrix_name.startswith("static"):
            if ',' in weight_threshold_matrix_name:
                strategy_name = strategy_name_dict[weight_threshold_matrix_name]
            else:
                strategy_name = f"a.{weight_threshold_matrix_name[-1]}"
        elif weight_threshold_matrix_name.startswith("skip-1"):
            strategy_name = f"b.{weight_threshold_matrix_name[-1]}"
        else:
            strategy_name = f"d.{weight_threshold_matrix_name[-1]}"
        
        return strategy_name

    # ---------= [Cal Metric] =---------
    res_dict = {
        "pruning_res_dict": {
            split: {}
        }
    }

    fid_model = None
    lpips_model = None
    
    for gen_img_root_path in folder_path_list:
        gen_img_path_list = list(
            gen_img_path for gen_img_path in gen_img_root_path.iterdir() \
                if gen_img_path.suffix in [".png"]
        )
        gen_img_pil_list = [
            load_img_path(gen_img_path) \
                for gen_img_path in gen_img_path_list
        ]

        # ---------= [Cal FID] =---------
        sum_fid_score = 0.0
    
        for batch_idx in range(num_batch):
            start_idx = batch_idx * fid_batch_size

            if (batch_idx < num_batch - 1) or (num_img % fid_batch_size == 0):
                true_batch_size = fid_batch_size
            else:
                true_batch_size = num_img % fid_batch_size
            
            batch_ref_img_pil_list = ref_img_pil_list[start_idx: (start_idx + true_batch_size)]
            batch_gen_img_pil_list = gen_img_pil_list[start_idx: (start_idx + true_batch_size)]

            (
                batch_fid_score, 
                batch_model
            ) = cal_fid(
                img_pil_list_1 = batch_ref_img_pil_list, 
                img_pil_list_2 = batch_gen_img_pil_list, 

                batch_size = fid_batch_size, 
                feature_dim = fid_feature_dim, 
                num_worker = fid_num_worker, 

                device = device
            )

            sum_fid_score += batch_fid_score * true_batch_size

            if fid_model is None:
                fid_model = batch_model

            # clean up
            del batch_ref_img_pil_list, batch_gen_img_pil_list
            torch.cuda.empty_cache()
            gc.collect()
    
        avg_fid_score = sum_fid_score / num_img

        # ---------= [Cal LPIPS] =---------
        (
            batch_lpips_score, 
            batch_lpips_model
        ) = cal_lpips(
            img_pil_list_1 = ref_img_pil_list, 
            img_pil_list_2 = gen_img_pil_list, 
            img_size = img_size, 

            lpips_net_type = lpips_model_net, 

            device = device, 

            model = lpips_model
        )

        avg_lpips_score = batch_lpips_score

        if lpips_model is None:
            lpips_model = batch_lpips_model

        # ---------= [Record Result] =---------
        logger(f"img:")
        logger(f"    ref_img_root_path: {ref_img_root_path}")
        logger(f"    gen_img_root_path: {gen_img_root_path}")

        logger(f"    [Result] avg_fid_score: {avg_fid_score}")
        logger(f"    [Result] avg_lpips_score: {avg_lpips_score}")

        folder_name = gen_img_root_path.stem
        strategy_name = get_strategy_name(folder_name)

        res_dict["pruning_res_dict"][split][strategy_name] = {
            "FID": f"{avg_fid_score:.4f}", 
            "LPIPS": f"{avg_lpips_score:.4f}"
        }

        # clean up
        del gen_img_pil_list
        torch.cuda.empty_cache()
        gc.collect()

        # goto `for gen_img_root_path`
        pass

    # ---------= [Save Yaml] =---------
    logger(
        f"res_dict: {res_dict}"
    )

    res_dict = convert_numpy_type_to_native_type(res_dict)

    save_yaml(
        res_dict, 

        yaml_root_path = folder_root_path, 
        yaml_filename = f"{sd_type}.yaml"
    )

    # clean up
    del fid_model
    del lpips_model
    torch.cuda.empty_cache()
    gc.collect()

    # `run_cal_fid_lpips_implement()` done
    pass

def run_cal_fid_lpips(
    cfg: DictConfig
):
    run_cal_fid_lpips_implement(cfg)

    pass
