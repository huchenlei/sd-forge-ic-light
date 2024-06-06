from modules.processing import StableDiffusionProcessing
from modules import devices

from typing import Callable
import safetensors.torch
import numpy as np
import torch

try:
    from lib_modelpatcher.model_patcher import ModulePatch
except ImportError as e:
    print("Please install [sd-webui-model-patcher] first!")
    print("https://github.com/huchenlei/sd-webui-model-patcher")
    raise e

from .args import ICLightArgs
from .utils import numpy2pytorch


def vae_encode(sd_model, image: torch.Tensor) -> torch.Tensor:
    """
    image: [B, C, H, W] format tensor. Value from -1.0 to 1.0
    Return: tensor in [B, C, H, W] format

    Note: Input image format differs from forge/comfy's vae input format
    """
    return sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image))


def apply_ic_light(
    p: StableDiffusionProcessing,
    args: ICLightArgs,
):
    device = devices.get_device_for("ic_light")
    dtype = devices.dtype_unet

    # Load model
    ic_model_state_dict = safetensors.torch.load_file(args.model_type.path)

    # Get input
    input_fg_rgb: np.ndarray = args.input_fg_rgb

    # [B, 4, H, W]
    concat_conds = vae_encode(
        p.sd_model,
        numpy2pytorch(args.get_concat_cond(input_fg_rgb, p)).to(
            dtype=devices.dtype_vae, device=device
        ),
    ).to(dtype=devices.dtype_unet)

    # [1, 4 * B, H, W]
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    def apply_c_concat(unet, old_forward: Callable) -> Callable:
        def new_forward(x, timesteps=None, context=None, **kwargs):
            # Expand according to batch number.
            c_concat = torch.cat(
                ([concat_conds.to(x.device)] * (x.shape[0] // concat_conds.shape[0])),
                dim=0,
            )
            new_x = torch.cat([x, c_concat], dim=1)
            return old_forward(new_x, timesteps, context, **kwargs)

        return new_forward

    # Patch unet forward.
    p.model_patcher.add_module_patch(
        "diffusion_model", ModulePatch(create_new_forward_func=apply_c_concat)
    )

    # Patch weights.
    p.model_patcher.add_patches(
        patches={
            "diffusion_model." + key: (value.to(dtype=dtype, device=device),)
            for key, value in ic_model_state_dict.items()
        }
    )

    # Add input image to extra result images
    if not getattr(p, "is_hr_pass", False):
        if not getattr(p, "extra_result_images", None):
            p.extra_result_images = [input_fg_rgb]
        else:
            assert isinstance(p.extra_result_images, list)
            p.extra_result_images.append(input_fg_rgb)
