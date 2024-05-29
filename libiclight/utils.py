import torch
import numpy as np
from PIL import Image
from .briarmbg import BriaRMBG


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    """Note: A1111's VAE accepts -1 ~ 1 tensors."""
    h = (
        torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    )  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


@torch.inference_mode()
def forge_numpy2pytorch(img: np.ndarray) -> torch.Tensor:
    """Note: Forge/ComfyUI's VAE accepts 0 ~ 1 tensors."""
    h = torch.from_numpy(img.astype(np.float32) / 255.0)
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


def align_dim_latent(x: int) -> int:
    """Align the pixel dimension (w/h) to latent dimension.
    Stable diffusion 1:8 ratio for latent/pixel, i.e.,
    1 latent unit == 8 pixel unit."""
    return (x // 8) * 8


def make_masked_area_grey(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Make the masked area grey."""
    return (
        (image.astype(np.float32) * alpha + (1 - alpha) * 127)
        .astype(np.uint8)
        .clip(0, 255)
    )


class BriarmbgService:
    """Singleton service class for BriaRMBG."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.rmbg_model = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

    @torch.inference_mode()
    def run_rmbg(self, img, device=None) -> np.ndarray:
        if device is None:
            device = torch.device("cuda")
        H, W, C = img.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
        alpha = self.rmbg_model.to(device)(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        return alpha
