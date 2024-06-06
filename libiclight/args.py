from modules import scripts
from modules.api import api
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
)

from scripts.model_loader import ModelType
from libiclight.rembg_utils import run_rmbg
from libiclight.utils import (
    align_dim_latent,
    make_masked_area_grey,
    resize_and_center_crop,
)

from pydantic import BaseModel, validator
from typing import Optional
from enum import Enum
import numpy as np
import torch


class BGSourceFC(Enum):
    """BGSource for FC model."""

    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    CUSTOM = "Custom LightMap"

    def get_bg(
        self,
        image_width: int,
        image_height: int,
        **kwargs,
    ) -> np.ndarray:

        match self:

            case BGSourceFC.LEFT:
                gradient = np.linspace(255, 0, image_width)
                image = np.tile(gradient, (image_height, 1))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFC.RIGHT:
                gradient = np.linspace(0, 255, image_width)
                image = np.tile(gradient, (image_height, 1))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFC.TOP:
                gradient = np.linspace(255, 0, image_height)[:, None]
                image = np.tile(gradient, (1, image_width))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFC.BOTTOM:
                gradient = np.linspace(0, 255, image_height)[:, None]
                image = np.tile(gradient, (1, image_width))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case _:
                raise NotImplementedError("Wrong initial latent!")

        return input_bg


class BGSourceFBC(Enum):
    """BGSource for FBC model."""

    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"

    def get_bg(
        self,
        image_width: int,
        image_height: int,
        uploaded_bg: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        match self:

            case BGSourceFBC.UPLOAD:
                assert uploaded_bg is not None
                input_bg = uploaded_bg

            case BGSourceFBC.UPLOAD_FLIP:
                assert uploaded_bg is not None
                input_bg = np.fliplr(uploaded_bg)

            case BGSourceFBC.GREY:
                input_bg = (
                    np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
                )

            case BGSourceFBC.LEFT:
                gradient = np.linspace(224, 32, image_width)
                image = np.tile(gradient, (image_height, 1))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFBC.RIGHT:
                gradient = np.linspace(32, 224, image_width)
                image = np.tile(gradient, (image_height, 1))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFBC.TOP:
                gradient = np.linspace(224, 32, image_height)[:, None]
                image = np.tile(gradient, (1, image_width))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFBC.BOTTOM:
                gradient = np.linspace(32, 224, image_height)[:, None]
                image = np.tile(gradient, (1, image_width))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case _:
                raise NotImplementedError("Wrong background source!")

        return input_bg


class ICLightArgs(BaseModel):
    enabled: bool = False
    model_type: ModelType = None
    restore_detail: bool = False
    use_rmbg_for_restore: bool = False
    blur_radius: int = 5
    input_fg: Optional[np.ndarray] = None
    uploaded_bg: Optional[np.ndarray] = None
    bg_source_fc: BGSourceFC = BGSourceFC.NONE
    bg_source_fbc: BGSourceFBC = BGSourceFBC.UPLOAD
    remove_bg: bool = True

    @classmethod
    def cls_decode_base64(cls, base64string: str) -> np.ndarray:
        return np.array(api.decode_base64_to_image(base64string)).astype("uint8")

    @validator("input_fg", "uploaded_bg", pre=True, allow_reuse=True)
    def parse_image(cls, value) -> np.ndarray:
        if isinstance(value, str):
            return cls.cls_decode_base64(value)
        assert isinstance(value, np.ndarray) or value is None
        return value

    @validator("model_type", pre=True, allow_reuse=True)
    def parse_model_type(cls, value) -> ModelType:
        if isinstance(value, str):
            return ModelType.get(value)
        assert isinstance(value, ModelType) or value is None
        return value

    @classmethod
    def fetch_from(cls, p: StableDiffusionProcessing):
        script_runner: scripts.ScriptRunner = p.scripts
        ic_light_script: scripts.Script = [
            script
            for script in script_runner.alwayson_scripts
            if script.title() == "IC Light"
        ][0]

        args = p.script_args[ic_light_script.args_from : ic_light_script.args_to]

        assert len(args) == 1
        return ICLightArgs(**args[0])

    class Config:
        arbitrary_types_allowed = True

    def get_concat_cond(
        self,
        processed_fg: np.ndarray,  # fg with bg removed.
        p: StableDiffusionProcessing,
    ) -> np.ndarray:
        """Returns concat condition in [B, H, W, C] format."""

        if getattr(p, "is_hr_pass", False):
            assert isinstance(p, StableDiffusionProcessingTxt2Img)
            # TODO: Move the calculation to Forge main repo.
            if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                hr_x = int(p.width * p.hr_scale)
                hr_y = int(p.height * p.hr_scale)
            else:
                hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
            image_width = align_dim_latent(hr_x)
            image_height = align_dim_latent(hr_y)

        else:
            image_width = p.width
            image_height = p.height

        fg = resize_and_center_crop(processed_fg, image_width, image_height)

        match self.model_type:
            case ModelType.FC:
                np_concat = [fg]
            case ModelType.FBC:
                bg = resize_and_center_crop(
                    self.bg_source_fbc.get_bg(
                        image_width, image_height, self.uploaded_bg
                    ),
                    image_width,
                    image_height,
                )
                np_concat = [fg, bg]
            case _:
                raise SystemError

        return np.stack(np_concat, axis=0)

    def get_input_rgb(self, device: torch.device = None) -> np.ndarray:
        """Returns rgb image in format [H, W, C=3]"""
        if self.remove_bg:
            input_rgb: np.ndarray = run_rmbg(self.input_fg)
        else:
            if len(self.input_fg.shape) < 3:
                raise NotImplementedError("Does not support L Images...")
            if self.input_fg.shape[-1] == 4:
                input_rgb = make_masked_area_grey(
                    self.input_fg[..., :3],
                    self.input_fg[..., 3:].astype(np.float32) / 255.0,
                )
            else:
                input_rgb = self.input_fg

        assert input_rgb.shape[2] == 3, "Input Image should be in RGB channels."

        return input_rgb
