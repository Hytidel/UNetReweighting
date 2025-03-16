from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

import torch

from einops import rearrange

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import random

import subprocess

from pathlib import Path
import os
import shutil

from util.basic_util import pause, get_global_variable, is_none, get_true_value
from util.torch_util import get_generator
from util.image_util import save_pil_as_png
from util.pipeline_util import load_pipeline, load_scheduler, get_inference_step_minus_one, img_latent_to_pil
from util.yaml_util import load_yaml, save_yaml, convert_numpy_type_to_native_type
from util.pkl_util import load_pkl, save_pkl
from util.json_util import load_json
from util.plot_util import get_line_chart



def test_implement(
    cfg: DictConfig
):
    # ---------= [Global Variables] =---------
    logger(f"[Global Variables] Loading started. ")

    exp_name = get_global_variable("exp_name")
    device = get_global_variable("device")
    seed = get_global_variable("seed")

    logger(f"[Global Variables] Loading finished. ")

    # ---------= [Task] =---------
    

    # `test_implement()` done
    pass

def test(
    cfg: DictConfig
):
    test_implement(cfg)

    pass