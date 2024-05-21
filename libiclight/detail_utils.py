# =========================================== #
# Reference:                                  #
# https://youtu.be/5EuYKEvugLU?feature=shared #
# =========================================== #

from PIL import Image
import numpy as np
import cv2


def restore_detail(
    ic_light_image: np.array,
    original_image: np.array,
    blur_radius: int = 5,
) -> Image:

    h, w, c = ic_light_image.shape
    original_image = cv2.resize(original_image, (w, h))

    if len(ic_light_image.shape) == 2:
        ic_light_image = cv2.cvtColor(ic_light_image, cv2.COLOR_GRAY2RGB)
    elif ic_light_image.shape[2] == 4:
        ic_light_image = cv2.cvtColor(ic_light_image, cv2.COLOR_RGBA2RGB)

    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)

    assert ic_light_image.shape[2] == 3
    assert original_image.shape[2] == 3

    ic_light_image = ic_light_image.astype(np.float32) / 255.0
    original_image = original_image.astype(np.float32) / 255.0

    blurred_ic_light = cv2.GaussianBlur(ic_light_image, (blur_radius, blur_radius), 0)
    blurred_original = cv2.GaussianBlur(original_image, (blur_radius, blur_radius), 0)

    DoG = original_image - blurred_original + blurred_ic_light
    DoG = np.clip(DoG * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(DoG)
