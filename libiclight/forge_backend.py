from modules.processing import StableDiffusionProcessing

import numpy as np
import torch

from ldm_patched.modules.model_management import get_torch_device
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.utils import load_torch_file
from ldm_patched.modules.sd import VAE

from .args import ICLightArgs
from .ic_light_nodes import ICLight
from .utils import forge_numpy2pytorch


def apply_ic_light(
    p: StableDiffusionProcessing,
    args: ICLightArgs,
):
    device = get_torch_device()

    # Load model
    ic_model_state_dict = load_torch_file(args.model_type.path, device=device)

    # Get input
    input_fg_rgb: np.ndarray = args.input_fg_rgb

    # Apply IC Light
    work_model: ModelPatcher = p.sd_model.forge_objects.unet.clone()
    vae: VAE = p.sd_model.forge_objects.vae.clone()
    node = ICLight()

    # [B, C, H, W]
    pixel_concat = forge_numpy2pytorch(args.get_concat_cond(input_fg_rgb, p)).to(
        device=vae.device, dtype=torch.float16
    )
    # [B, H, W, C]
    # Forge/ComfyUI's VAE accepts [B, H, W, C] format.
    pixel_concat = pixel_concat.movedim(1, 3)

    patched_unet: ModelPatcher = node.apply(
        model=work_model,
        ic_model_state_dict=ic_model_state_dict,
        c_concat={"samples": vae.encode(pixel_concat)},
    )[0]
    p.sd_model.forge_objects.unet = patched_unet

    # Add input image to extra result images
    if not getattr(p, "is_hr_pass", False):
        p.extra_result_images.append(input_fg_rgb)
