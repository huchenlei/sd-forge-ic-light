""" A1111 IC Light extension backend."""

import os
from typing import Callable
import torch
import numpy as np

from modules import devices
from modules.paths import models_path
from modules.safe import unsafe_torch_load
from modules.processing import StableDiffusionProcessing

try:
    from lib_modelpatcher.model_patcher import ModulePatch
    from lib_modelpatcher.sd_model_patcher import StableDiffusionModelPatchers
except ImportError as e:
    print("Please install sd-webui-model-patcher")
    raise e

from .args import ICLightArgs
from .ic_light_nodes import ICLight


def vae_encode(sd_model, img: torch.Tensor) -> torch.Tensor:
    """img: [B, C, H, W]"""
    return sd_model.get_first_stage_encoding(sd_model.encode_first_stage(img))


def apply_ic_light(
    p: StableDiffusionProcessing,
    args: ICLightArgs,
):
    device = devices.get_device_for("ic_light")
    dtype = devices.dtype_unet

    # Load model
    unet_path = os.path.join(models_path, "unet", args.model_type.model_name)
    ic_model_state_dict = unsafe_torch_load(unet_path, device=device)

    # Get input
    input_rgb: np.ndarray = args.get_input_rgb(device=device)

    # Apply IC Light
    model_patchers: StableDiffusionModelPatchers = p.model_patchers

    # [B, 4, H, W]
    c_concat = vae_encode(p.sd_model, args.get_concat_cond(input_rgb, p))
    # [1, 4 * B, H, W]
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    def apply_c_concat(unet, old_forward: Callable) -> Callable:
        def new_forward(x, timesteps=None, context=None, **kwargs):
            # Expand according to batch number.
            c_concat = torch.cat(
                ([concat_conds.to(x.device)] * (x.shape[0] // concat_conds.shape[0])),
                dim=0,
            )
            new_x = torch.cat([x] + c_concat, dim=1)
            return old_forward(new_x, timesteps, context, **kwargs)

        return new_forward

    # Patch unet forward.
    model_patchers.unet_patcher.add_module_patch(
        ".", ModulePatch(create_new_forward_func=apply_c_concat)
    )

    # Note: no need to prefix with "diffusion_model." as unet is the root module.
    model_patchers.unet_patcher.add_patches(
        patches={
            key: (value.to(dtype=dtype, device=device),)
            for key, value in ic_model_state_dict.items()
        }
    )
