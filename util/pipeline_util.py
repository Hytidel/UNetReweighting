from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import torch
import torchvision

from PIL import Image

import numpy as np

from pathlib import Path

from util.basic_util import is_none, get_true_value


INFERENCE_STEP_MINUS_ONE_SCHEDULER_LIST = [
    "PNDMScheduler"
]

def load_pipeline(
    pipeline_type: str, 
    pipeline_path: str, 
    torch_dtype: str = None, 
    variant: str = None, 
):
    import importlib

    def get_pipeline_class(
        pipeline_name: str
    ):
        try:
            module = importlib.import_module("diffusers.pipelines")
            Pipeline = getattr(module, pipeline_name)

            return Pipeline
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"{e}\n"
                f"Unsupported `pipeline_name`, got {pipeline_name}. "
            )
            
            return None

    pipeline = get_pipeline_class(pipeline_type) \
        .from_pretrained(
            pipeline_path, 
            torch_dtype = getattr(torch, torch_dtype), 
            variant = variant
        )

    return pipeline

def load_scheduler(
    scheduler_type: str, 
    pipeline
):
    import importlib

    def get_scheduler_class(
        scheduler_type: str
    ):
        try:
            module = importlib.import_module("diffusers.schedulers")
            Scheduler = getattr(module, scheduler_type)

            return Scheduler
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"{e}\n"
                f"Unsupported `scheduler_type`, got {scheduler_type}. "
            )
            
            return None

    pipeline.scheduler = get_scheduler_class(scheduler_type) \
        .from_config(pipeline.scheduler.config)

def get_scheduler(
    scheduler_type: str, 
    pipeline
):
    import importlib

    def get_scheduler_class(
        scheduler_type: str
    ):
        try:
            module = importlib.import_module("diffusers.schedulers")
            Scheduler = getattr(module, scheduler_type)

            return Scheduler
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"{e}\n"
                f"Unsupported `scheduler_type`, got {scheduler_type}. "
            )
            
            return None

    scheduler = get_scheduler_class(scheduler_type) \
        .from_config(pipeline.scheduler.config)
    
    return scheduler

def get_inference_step_minus_one(
    scheduler_type: str
) -> bool:
    return scheduler_type in INFERENCE_STEP_MINUS_ONE_SCHEDULER_LIST

def load_unet(
    unet_type: str,  # ["UNet2DModel", "UNet2DConditionModel"]
    pipeline
):
    import importlib

    def get_unet_class(
        unet_type: str
    ):
        try:
            module = importlib.import_module("diffusers.models")
            UNet = getattr(module, unet_type)

            return UNet
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"{e}\n"
                f"Unsupported `unet_type`, got {unet_type}. "
            )
            
            return None

    unet = get_unet_class(unet_type) \
        .from_config(pipeline.unet.config)
        
    return unet

@torch.no_grad()
def img_pil_to_latent(
    img_pil, 
    pipeline
) -> torch.Tensor:
    # value [0, 255] -> [0, 1] -> [-1, 1]
    img_tensor = torchvision.transforms.functional.to_tensor(img_pil) \
        .unsqueeze(0) * 2 - 1
    
    img_tensor = img_tensor.to(
        dtype = pipeline.vae.dtype, 
        device = pipeline.device
    )
    
    latent = pipeline.vae.encode(img_tensor)

    latent = 0.18215 * latent.latent_dist.sample()

    return latent

@torch.no_grad()
def img_latent_to_pil(
    img_latent, 
    pipeline
) -> Image.Image:
    if hasattr(pipeline, "decode_latents"):
        img_numpy = pipeline.decode_latents(img_latent)

        return pipeline.numpy_to_pil(img_numpy)
    else:
        # make sure the VAE is in `float32` mode, as it overflows in `float16`
        if (pipeline.vae.dtype == torch.float16) and pipeline.vae.config.force_upcast:
            pipeline.upcast_vae()
            img_latent = img_latent.to(
                next(
                    iter(
                        pipeline.vae.post_quant_conv.parameters()
                    )
                ).dtype
            )

        img_tensor = pipeline.vae.decode(
            img_latent / pipeline.vae.config.scaling_factor, 
            return_dict = False
        )[0]
        
        img_pil = pipeline.image_processor.postprocess(
            img_tensor, 
            output_type = "pil"
        )

        return img_pil

def process_prompt_list(
    prompt: Union[str, List[str]], 
    batch_size: Optional[int] = None, 
    negative_prompt: Union[str, List[str]] = None, 
):
    if isinstance(prompt, list):
        if is_none(batch_size):
            batch_size = len(prompt)
        elif batch_size != len(prompt):
            raise ValueError(
                f"The length of the `prompt` list doesn't match `batch_size`, "
                f"got {len(prompt)} and {batch_size}. "
            )
    elif is_none(batch_size):
        batch_size = 1
        prompt = [prompt]
    else:
        prompt = [prompt] * batch_size

    if negative_prompt is not None:
        if isinstance(negative_prompt, list):
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"The length of the `negative_prompt` list doesn't match `batch_size`, "
                    f"got {len(negative_prompt)} and {batch_size}. "
                    )
        else:
            negative_prompt = [negative_prompt] * batch_size

    return prompt, batch_size, negative_prompt
