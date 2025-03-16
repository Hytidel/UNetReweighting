# modified from HuggingFace diffusers (0.32.1) `models/unets/unet_2d_blocks`

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import apply_freeu

from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D, 
    CrossAttnUpBlock2D, 
    UNetMidBlock2DCrossAttn
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def cross_attn_down_block_2d_forward(
    self,
    hidden_states: torch.Tensor,
    temb: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    additional_residuals: Optional[torch.Tensor] = None, 

    # attention weighting
    weight: Union[float, torch.Tensor] = None, 
    weight_threshold: Union[float, torch.Tensor] = None, 
    weight_eps: float = 1e-6
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    # default `weight` and `weight_threshold`
    # if weight is None:
    #     weight = 1.0
    # if weight_threshold is None:
    #     weight_threshold = 0.0
    
    # if not isinstance(weight, torch.Tensor):
    #     weight = torch.tensor(
    #         weight, 

    #         device = hidden_states.device, 
    #         dtype = hidden_states.dtype
    #     )
    # if not isinstance(weight_threshold, torch.Tensor):
    #     weight_threshold = torch.tensor(
    #         weight_threshold, 

    #         device = hidden_states.device, 
    #         dtype = hidden_states.dtype
    #     )
        
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    output_states = ()

    blocks = list(zip(self.resnets, self.attentions))
    
    for i, (resnet, attn) in enumerate(blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )

            # attention weighting
            if weight + weight_eps >= weight_threshold:
                attn_res = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # weight_tensor = torch.tensor(
                #     weight, 
                #     device = attn_res.device, 
                #     dtype = attn_res.dtype
                # )

                # hidden_states = attn_res * weight_tensor

                hidden_states = attn_res * weight
        else:
            hidden_states = resnet(hidden_states, temb)
            
            # attention weighting
            if weight + weight_eps >= weight_threshold:
                attn_res = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # weight_tensor = torch.tensor(
                #     weight, 
                #     device = attn_res.device, 
                #     dtype = attn_res.dtype
                # )

                # hidden_states = attn_res * weight_tensor

                hidden_states = attn_res * weight

        # apply additional residuals to the output of the last pair of resnet and attention blocks
        if i == len(blocks) - 1 and additional_residuals is not None:
            hidden_states = hidden_states + additional_residuals

        output_states = output_states + (hidden_states,)

    if self.downsamplers is not None:
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)

        output_states = output_states + (hidden_states,)

    return hidden_states, output_states


def cross_attn_up_block_2d_forward(
    self,
    hidden_states: torch.Tensor,
    res_hidden_states_tuple: Tuple[torch.Tensor, ...],
    temb: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    upsample_size: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None, 

    # attention weighting
    weight: Union[float, torch.Tensor] = 1.0, 
    weight_threshold: Union[float, torch.Tensor] = 0.0, 
    weight_eps: float = 1e-6
) -> torch.Tensor:
    # default `weight` and `weight_threshold`
    # if weight is None:
    #     weight = 1.0
    # if weight_threshold is None:
    #     weight_threshold = 0.0
    
    # if not isinstance(weight, torch.Tensor):
    #     weight = torch.tensor(
    #         weight, 

    #         device = hidden_states.device, 
    #         dtype = hidden_states.dtype
    #     )
    # if not isinstance(weight_threshold, torch.Tensor):
    #     weight_threshold = torch.tensor(
    #         weight_threshold, 

    #         device = hidden_states.device, 
    #         dtype = hidden_states.dtype
    #     )

    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    is_freeu_enabled = (
        getattr(self, "s1", None)
        and getattr(self, "s2", None)
        and getattr(self, "b1", None)
        and getattr(self, "b2", None)
    )

    for resnet, attn in zip(self.resnets, self.attentions):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        # FreeU: Only operate on the first two stages
        if is_freeu_enabled:
            hidden_states, res_hidden_states = apply_freeu(
                self.resolution_idx,
                hidden_states,
                res_hidden_states,
                s1=self.s1,
                s2=self.s2,
                b1=self.b1,
                b2=self.b2,
            )

        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )

            # attention weighting
            if weight + weight_eps >= weight_threshold:
                attn_res = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # weight_tensor = torch.tensor(
                #     weight, 
                #     device = attn_res.device, 
                #     dtype = attn_res.dtype
                # )

                # hidden_states = attn_res * weight_tensor

                hidden_states = attn_res * weight
        else:
            hidden_states = resnet(hidden_states, temb)

            # attention weighting
            if weight + weight_eps >= weight_threshold:
                attn_res = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # weight_tensor = torch.tensor(
                #     weight, 
                #     device = attn_res.device, 
                #     dtype = attn_res.dtype
                # )

                # hidden_states = attn_res * weight_tensor

                hidden_states = attn_res * weight

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states


def unet_mid_block_2d_cross_attn_forward(
    self,
    hidden_states: torch.Tensor,
    temb: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None, 

    # attention weighting
    weight: Union[float, torch.Tensor] = 1.0, 
    weight_threshold: Union[float, torch.Tensor] = 0.0, 
    weight_eps: float = 1e-6
) -> torch.Tensor:
    # default `weight` and `weight_threshold`
    # if weight is None:
    #     weight = 1.0
    # if weight_threshold is None:
    #     weight_threshold = 0.0
    
    # if not isinstance(weight, torch.Tensor):
    #     weight = torch.tensor(
    #         weight, 

    #         device = hidden_states.device, 
    #         dtype = hidden_states.dtype
    #     )
    # if not isinstance(weight_threshold, torch.Tensor):
    #     weight_threshold = torch.tensor(
    #         weight_threshold, 

    #         device = hidden_states.device, 
    #         dtype = hidden_states.dtype
    #     )

    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    hidden_states = self.resnets[0](hidden_states, temb)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

            # attention weighting
            if weight + weight_eps >= weight_threshold:
                attn_res = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # weight_tensor = torch.tensor(
                #     weight, 
                #     device = attn_res.device, 
                #     dtype = attn_res.dtype
                # )

                # hidden_states = attn_res * weight_tensor

                hidden_states = attn_res * weight

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
        else:
            # attention weighting
            if weight + weight_eps >= weight_threshold:
                attn_res = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                # weight_tensor = torch.tensor(
                #     weight, 
                #     device = attn_res.device, 
                #     dtype = attn_res.dtype
                # )

                # hidden_states = attn_res * weight_tensor

                hidden_states = attn_res * weight

            hidden_states = resnet(hidden_states, temb)

    return hidden_states
