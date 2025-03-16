from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

import torch

from pathlib import Path

from tqdm.auto import tqdm

import gc

import concurrent.futures as cf

from util.basic_util import pause, get_global_variable, is_none, get_true_value
from util.image_util import load_img_path
from util.yaml_util import load_yaml, save_yaml, convert_numpy_type_to_native_type

from third_party import hpsv2


# @torch.no_grad()
def run_cal_hpsv2_implement(
    cfg: DictConfig
):
    # ---------= [Basic Global Variables] =---------
    exp_name = get_global_variable("exp_name")
    start_time = get_global_variable("start_time")
    device = get_global_variable("device")
    seed = get_global_variable("seed")
    exp_time_str = f"{exp_name}_{start_time}"

    concurrent_max_worker = get_global_variable("concurrent_max_worker")

    # ---------= [Image Root Path] =---------
    logger(f"[Image Root Path] Loading started. ")

    hps_model_ckpt_path = get_true_value(cfg["task"]["hps_v2"]["hps_model_ckpt_path"])
    ViT_model_ckpt_path = get_true_value(cfg["task"]["hps_v2"]["ViT_model_ckpt_path"])
    prompt_batch_size = get_true_value(cfg["task"]["hps_v2"]["prompt_batch_size"])

    logger(f"    hps_model_ckpt_path: {hps_model_ckpt_path}")
    logger(f"    ViT_model_ckpt_path: {ViT_model_ckpt_path}")
    logger(f"    prompt_batch_size: {prompt_batch_size}")

    logger(
        f"[Image Root Path] Loading finished. "
        "\n"
    )

    # ---------= [Image Root Path] =---------
    logger(f"[Image Root Path] Loading started. ")

    category_root_path = get_true_value(cfg["task"]["img_root_path"]["category_root_path"])
    category_root_path = Path(category_root_path)
    weight_matrix_name = get_true_value(cfg["task"]["img_root_path"]["weight_matrix_name"])

    logger(f"    category_root_path: {category_root_path}")
    logger(f"    weight_matrix_name: {weight_matrix_name}")

    logger(
        f"[Image Root Path] Loading finished. "
        "\n"
    )

    # ---------= [Sample] =---------
    logger(f"[Sample] Loading started. ")

    num_sample = get_true_value(cfg["task"]["sample"]["num_sample"])

    logger(f"    num_sample: {num_sample}")

    logger(
        f"[Sample] Loading finished. "
        "\n"
    )

    # ---------= [All Components Loaded] =---------
    logger(
        f"All components loaded. "
        "\n"
    )

    # ---------= [Cal HPSv2 Score] =---------
    folder_path_list = [_ for _ in category_root_path.iterdir()]
    cfg_dict_path_list = [
        folder_path / "cfg.yaml" \
            for folder_path in folder_path_list
    ]

    prompt_list = [
        load_yaml(cfg_dict_path)["prompt"] \
            for cfg_dict_path in cfg_dict_path_list
    ]

    prompt_tuple_list = []  # (prompt_idx, prompt)

    for prompt_idx, prompt in enumerate(
        tqdm(
            prompt_list, 

            desc = f"[Check Data Existence]"
        )
    ):
        img_root_path = folder_path_list[prompt_idx] / "png" / weight_matrix_name
        hpsv2_dict_path = img_root_path / "metric" / "hpsv2_score.yaml"
        
        if (img_root_path.is_dir()) and (not hpsv2_dict_path.is_file()):
            prompt_tuple_list.append((prompt_idx, prompt))

    num_prompt = len(prompt_tuple_list)
    tot_num_prompt = len(prompt_list)
    
    logger(f"    num_prompt / tot_num_prompt: {num_prompt} / {tot_num_prompt}")

    num_batch = (num_prompt + prompt_batch_size - 1) // prompt_batch_size

    logger(f"    num_batch: {num_batch}")

    def implement_batch(
        batch_idx: int
    ):
        batch_prompt_tuple_list = prompt_tuple_list[
            batch_idx * prompt_batch_size: 
            min((batch_idx + 1) * prompt_batch_size, num_prompt)
        ]

        logger(f"[Batch {batch_idx}]")
        logger(f"    batch_prompt_tuple_list: {batch_prompt_tuple_list}")

        for (prompt_idx, prompt) in batch_prompt_tuple_list:
            # logger(f"    (prompt_idx, prompt): ({prompt_idx}, {prompt})")

            img_path_list = [
                folder_path_list[prompt_idx] / "png" / weight_matrix_name / f"{sample_idx}.png" \
                    for sample_idx in range(num_sample)
            ]
            img_pil_list = [
                load_img_path(img_path) \
                    for img_path in img_path_list
            ]
        
            hps_v2_1_score_list = hpsv2.score(
                img_pil_list, 
                prompt = prompt, 

                hps_model_ckpt_path = hps_model_ckpt_path, 
                ViT_model_ckpt_path = ViT_model_ckpt_path, 
                hps_version = "v2.1"
            )

            yaml_dict = {
                "hpsv2_score": [None for _ in range(num_sample)], 
                "avg_hpsv2_score": None
            }

            hps_v2_1_score_list = [float(_) for _ in hps_v2_1_score_list]
            for sample_idx in range(num_sample):
                yaml_dict["hpsv2_score"] = hps_v2_1_score_list
                
                avg_hpsv2_score = sum(yaml_dict["hpsv2_score"]) / num_sample
                yaml_dict["avg_hpsv2_score"] = avg_hpsv2_score

            logger(f"yaml_dict: {yaml_dict}")

            yaml_dict = convert_numpy_type_to_native_type(yaml_dict)
            save_yaml(
                yaml_dict, 

                yaml_root_path = folder_path_list[prompt_idx] / "png" / weight_matrix_name / "metric", 
                yaml_filename = "hpsv2_score.yaml"
            )

        # clean up
        del img_pil_list
        del hps_v2_1_score_list
        torch.cuda.empty_cache()
        gc.collect()

        # `implement_batch()` done
        pass
    
    def implement_epoch(
        epoch_idx: int
    ):
        for batch_idx in range(num_batch):
            implement_batch(batch_idx = batch_idx)

    num_epoch_ = (num_prompt > 0)
    for epoch_idx in tqdm(
        range(num_epoch_)
    ):
        implement_epoch(epoch_idx = epoch_idx)

    # clean up
    torch.cuda.empty_cache()
    gc.collect()

    # `run_cal_hpsv2_implement()` done
    pass

def run_cal_hpsv2(
    cfg: DictConfig
):
    run_cal_hpsv2_implement(cfg)

    pass
