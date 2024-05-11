from dataclasses import dataclass
import os
import torch
import gradio as gr
import numpy as np
from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel

from modules import scripts, script_callbacks
from modules.ui_components import InputAccordion
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img
from modules.paths import models_path
from ldm_patched.modules.utils import load_torch_file
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.sd import VAE

from libiclight.ic_light_nodes import ICLight
from libiclight.briarmbg import BriaRMBG
from libiclight.utils import (
    align_dim_latent,
    run_rmbg,
    resize_and_center_crop,
    forge_numpy2pytorch,
)


@dataclass
class A1111Context:
    """Contains all components from A1111."""

    txt2img_submit_button: Optional[gr.components.Component] = None
    img2img_submit_button: Optional[gr.components.Component] = None

    # Slider controls from A1111 WebUI.
    img2img_w_slider: Optional[gr.components.Component] = None
    img2img_h_slider: Optional[gr.components.Component] = None

    img2img_image: Optional[gr.Image] = None

    def set_component(self, component: gr.components.Component):
        id_mapping = {
            "txt2img_generate": "txt2img_submit_button",
            "img2img_generate": "img2img_submit_button",
            "img2img_width": "img2img_w_slider",
            "img2img_height": "img2img_h_slider",
            "img2img_image": "img2img_image",
        }
        elem_id = getattr(component, "elem_id", None)
        if elem_id in id_mapping and getattr(self, id_mapping[elem_id]) is None:
            setattr(self, id_mapping[elem_id], component)


class BGSourceFC(Enum):
    """BGSource for FC model."""

    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

    def get_bg(
        self,
        image_width: int,
        image_height: int,
        **kwargs,
    ) -> np.ndarray:
        bg_source = self
        if bg_source == BGSourceFC.NONE:
            pass
        elif bg_source == BGSourceFC.LEFT:
            gradient = np.linspace(255, 0, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSourceFC.RIGHT:
            gradient = np.linspace(0, 255, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSourceFC.TOP:
            gradient = np.linspace(255, 0, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSourceFC.BOTTOM:
            gradient = np.linspace(0, 255, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            raise "Wrong initial latent!"

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
        bg_source = self
        if bg_source == BGSourceFBC.UPLOAD:
            assert uploaded_bg is not None
            input_bg = uploaded_bg
        elif bg_source == BGSourceFBC.UPLOAD_FLIP:
            input_bg = np.fliplr(uploaded_bg)
        elif bg_source == BGSourceFBC.GREY:
            input_bg = (
                np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
            )
        elif bg_source == BGSourceFBC.LEFT:
            gradient = np.linspace(224, 32, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSourceFBC.RIGHT:
            gradient = np.linspace(32, 224, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSourceFBC.TOP:
            gradient = np.linspace(224, 32, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSourceFBC.BOTTOM:
            gradient = np.linspace(32, 224, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            raise "Wrong background source!"
        return input_bg


class ModelType(Enum):
    FC = "FC"
    FBC = "FBC"

    @property
    def model_name(self) -> str:
        if self == ModelType.FC:
            return "iclight_sd15_fc_unet_ldm.safetensors"
        else:
            assert self == ModelType.FBC
            return "iclight_sd15_fbc_unet_ldm.safetensors"


class ICLightArgs(BaseModel):
    enabled: bool = False
    model_type: ModelType = ModelType.FC
    input_fg: np.ndarray
    uploaded_bg: Optional[np.ndarray] = None
    bg_source_fc: BGSourceFC = BGSourceFC.NONE
    bg_source_fbc: BGSourceFBC = BGSourceFBC.UPLOAD

    class Config:
        arbitrary_types_allowed = True

    def get_c_concat(
        self,
        processed_fg: np.ndarray,  # fg with bg removed.
        vae: VAE,
        p: StableDiffusionProcessing,
        device: torch.device,
    ) -> dict:
        is_hr_pass = getattr(p, 'is_hr_pass', False)
        if is_hr_pass:
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
        if self.model_type == ModelType.FC:
            np_concat = [fg]
        else:
            assert self.model_type == ModelType.FBC
            bg = resize_and_center_crop(
                self.bg_source_fbc.get_bg(image_width, image_height, self.uploaded_bg),
                image_width,
                image_height,
            )
            np_concat = [fg, bg]

        concat_conds = forge_numpy2pytorch(np.stack(np_concat, axis=0)).to(
            device=device, dtype=torch.float16
        )

        # Optional: Use mode instead of sample from VAE output.
        # vae.first_stage_model.regularization.sample = False
        latent_concat_conds = vae.encode(concat_conds)
        return {"samples": latent_concat_conds}


class ICLightForge(scripts.Script):
    DEFAULT_ARGS = ICLightArgs(
        input_fg=np.zeros(shape=[1, 1, 1], dtype=np.uint8),
    )
    a1111_context = A1111Context()

    def title(self):
        return "IC Light"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> Tuple[gr.components.Component, ...]:
        if is_img2img:
            model_type_choices = [ModelType.FC.value]
            bg_source_fc_choices = [e.value for e in BGSourceFC if e != BGSourceFC.NONE]
        else:
            model_type_choices = [ModelType.FC.value, ModelType.FBC.value]
            bg_source_fc_choices = [BGSourceFC.NONE.value]

        with InputAccordion(value=False, label=self.title()) as enabled:
            with gr.Row():
                input_fg = gr.Image(
                    source="upload", type="numpy", label="Foreground", height=480
                )
                uploaded_bg = gr.Image(
                    source="upload",
                    type="numpy",
                    label="Background",
                    height=480,
                    interactive=True,
                    visible=not is_img2img,
                )

            model_type = gr.Dropdown(
                label="Model",
                choices=model_type_choices,
                value=ModelType.FC.value,
                interactive=True,
            )

            bg_source_fc = gr.Radio(
                label="Background Source",
                choices=bg_source_fc_choices,
                value=BGSourceFC.NONE.value,
                type="value",
                visible=True,
                interactive=True,
            )

            bg_source_fbc = gr.Radio(
                label="Background Source",
                choices=[e.value for e in BGSourceFBC],
                value=BGSourceFBC.UPLOAD.value,
                type="value",
                visible=False,
                interactive=True,
            )

        # TODO return a dict here so that API calls are cleaner.
        states = (
            enabled,
            model_type,
            input_fg,
            uploaded_bg,
            bg_source_fc,
            bg_source_fbc,
        )

        if is_img2img:

            def update_img2img_input(bg_source_fc: str, height: int, width: int):
                return gr.update(
                    value=BGSourceFC(bg_source_fc).get_bg(
                        image_width=width, image_height=height
                    )
                )

            # FC need to change img2img input.
            for component in (
                bg_source_fc,
                ICLightForge.a1111_context.img2img_h_slider,
                ICLightForge.a1111_context.img2img_w_slider,
            ):
                component.change(
                    fn=update_img2img_input,
                    inputs=[
                        bg_source_fc,
                        ICLightForge.a1111_context.img2img_h_slider,
                        ICLightForge.a1111_context.img2img_w_slider,
                    ],
                    outputs=ICLightForge.a1111_context.img2img_image,
                )

        def on_model_type_change(model_type: str):
            model_type = ModelType(model_type)
            if model_type == ModelType.FC:
                return gr.update(visible=True), gr.update(visible=False)
            else:
                assert model_type == ModelType.FBC
                return gr.update(visible=False), gr.update(visible=True)

        model_type.change(
            fn=on_model_type_change,
            inputs=[model_type],
            outputs=[bg_source_fc, bg_source_fbc],
            show_progress=False,
        )

        return states

    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *script_args, **kwargs
    ):
        args = ICLightArgs(
            **{
                k: v
                for k, v in zip(
                    vars(self.DEFAULT_ARGS).keys(),
                    script_args,
                )
            }
        )
        if not args.enabled:
            return

        device = torch.device("cuda")
        rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4").to(device=device)
        alpha = run_rmbg(rmbg, img=args.input_fg, device=device)
        input_rgb: np.ndarray = (
            (args.input_fg.astype(np.float32) * alpha).astype(np.uint8).clip(0, 255)
        )

        work_model: ModelPatcher = p.sd_model.forge_objects.unet.clone()
        vae: ModelPatcher = p.sd_model.forge_objects.vae.clone()
        unet_path = os.path.join(models_path, "unet", args.model_type.model_name)
        ic_model_state_dict = load_torch_file(unet_path, device=device)
        node = ICLight()

        patched_unet: ModelPatcher = node.apply(
            model=work_model,
            ic_model_state_dict=ic_model_state_dict,
            c_concat=args.get_c_concat(input_rgb, vae, p, device=device),
        )[0]

        p.sd_model.forge_objects.unet = patched_unet

    @staticmethod
    def on_after_component(component, **_kwargs):
        """Register the A1111 component."""
        ICLightForge.a1111_context.set_component(component)


script_callbacks.on_after_component(ICLightForge.on_after_component)
