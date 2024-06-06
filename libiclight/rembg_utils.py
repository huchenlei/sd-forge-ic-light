# ============================================================= #
# Reference:                                                    #
# https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg #
# ============================================================= #

from modules.paths import models_path
from PIL import Image
import numpy as np
import rembg
import os

AVAILABLE_MODELS = (
    "u2net_human_seg",
    "isnet-anime",
    # "u2net",
    # "u2netp",
    # "u2net_cloth_seg",
    # "silueta",
    # "isnet-general-use",
)

GREY = (127, 127, 127, 255)


def run_rmbg(
    np_image: np.array,
    model: str = "u2net_human_seg",
    foreground_threshold: int = 225,
    background_threshold: int = 16,
    erode_size: int = 16,
    bg: tuple = GREY,
) -> np.array:

    if "U2NET_HOME" not in os.environ:
        os.environ["U2NET_HOME"] = os.path.join(models_path, "u2net")

    image = Image.fromarray(np_image.astype(np.uint8)).convert("RGB")

    processed_image = rembg.remove(
        image,
        session=rembg.new_session(model),
        alpha_matting=True,
        alpha_matting_foreground_threshold=foreground_threshold,
        alpha_matting_background_threshold=background_threshold,
        alpha_matting_erode_size=erode_size,
        post_process_mask=True,
        only_mask=False,
        bgcolor=bg,
    )

    return np.array(processed_image.convert("RGB")).astype(np.uint8)
