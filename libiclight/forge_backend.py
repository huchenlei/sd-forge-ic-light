""" SD Forge IC Light extension backend."""

import os
import numpy as np

from modules.paths import models_path
from modules.processing import StableDiffusionProcessing

from ldm_patched.modules.utils import load_torch_file
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.sd import VAE
from ldm_patched.modules.model_management import get_torch_device

from .args import ICLightArgs
from .ic_light_nodes import ICLight


def apply_ic_light(
    p: StableDiffusionProcessing,
    args: ICLightArgs,
):
    device = get_torch_device()

    # Load model
    unet_path = os.path.join(models_path, "unet", args.model_type.model_name)
    ic_model_state_dict = load_torch_file(unet_path, device=device)

    # Get input
    input_rgb: np.ndarray = args.get_input_rgb(device=device)

    # Apply IC Light
    work_model: ModelPatcher = p.sd_model.forge_objects.unet.clone()
    vae: VAE = p.sd_model.forge_objects.vae.clone()
    node = ICLight()
    patched_unet: ModelPatcher = node.apply(
        model=work_model,
        ic_model_state_dict=ic_model_state_dict,
        c_concat=args.get_c_concat(input_rgb, vae, p, device=device),
    )[0]
    p.sd_model.forge_objects.unet = patched_unet

    # Add input image to extra result images
    is_hr_pass = getattr(p, "is_hr_pass", False)
    if not is_hr_pass:
        p.extra_result_images.append(input_rgb)
