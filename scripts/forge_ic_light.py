from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.utils import load_torch_file

from modules import scripts, script_callbacks
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.ui_components import InputAccordion
from modules.api import api

from scripts.model_loader import ModelType

from libiclight.ic_light_nodes import ICLight
from libiclight.rembg_utils import run_rmbg
from libiclight.utils import (
    align_dim_latent,
    resize_and_center_crop,
    forge_numpy2pytorch,
)

from pydantic import BaseModel, validator
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import gradio as gr
import numpy as np
import torch


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
    input_fg: Optional[np.ndarray] = None
    uploaded_bg: Optional[np.ndarray] = None
    bg_source_fc: BGSourceFC = BGSourceFC.NONE
    bg_source_fbc: BGSourceFBC = BGSourceFBC.UPLOAD

    @classmethod
    def cls_decode_base64(cls, base64string: str) -> np.ndarray:
        return np.array(api.decode_base64_to_image(base64string)).astype("uint8")

    @validator("input_fg", pre=True)
    def parse_input_fg(cls, value) -> np.ndarray:
        if isinstance(value, str):
            return cls.cls_decode_base64(value)
        assert isinstance(value, np.ndarray) or value is None
        return value

    @validator("uploaded_bg", pre=True)
    def parse_input_bg(cls, value) -> np.ndarray:
        if isinstance(value, str):
            return cls.cls_decode_base64(value)
        assert isinstance(value, np.ndarray) or value is None
        return value

    @classmethod
    def fetch_from(cls, p):
        script_runner: scripts.ScriptRunner = p.scripts
        ic_light_script: scripts.Script = [
            script
            for script in script_runner.alwayson_scripts
            if script.title() == "IC Light"
        ][0]

        args = p.script_args[ic_light_script.args_from : ic_light_script.args_to]
        args[0]["model_type"] = ModelType.get(args[0]["model_type"])

        assert len(args) == 1
        return ICLightArgs(**args[0])

    class Config:
        arbitrary_types_allowed = True

    def get_c_concat(
        self,
        processed_fg: np.ndarray,  # fg with bg removed.
        vae,
        p,
        device: torch.device,
    ) -> dict:

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
    DEFAULT_ARGS = ICLightArgs(input_fg=np.zeros((1, 1, 1), dtype=np.uint8))
    a1111_context = A1111Context()

    def title(self):
        return "IC Light"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> Tuple[gr.components.Component, ...]:

        if is_img2img:
            model_type_choices = [ModelType.FC.name]
            bg_source_fc_choices = [e.value for e in BGSourceFC if e != BGSourceFC.NONE]
        else:
            model_type_choices = [ModelType.FC.name, ModelType.FBC.name]
            bg_source_fc_choices = [BGSourceFC.NONE.value]

        with InputAccordion(value=False, label=self.title()) as enabled:

            with gr.Row():
                input_fg = gr.Image(
                    source="upload",
                    type="numpy",
                    label="Foreground",
                    height=480,
                    interactive=True,
                    visible=True,
                )
                uploaded_bg = gr.Image(
                    source="upload",
                    type="numpy",
                    label="Background",
                    height=480,
                    interactive=True,
                    visible=False,
                )

            model_type = gr.Dropdown(
                label="Model",
                choices=model_type_choices,
                value=ModelType.FC.name,
                interactive=True,
            )

            bg_source_fc = gr.Radio(
                label="Background Source",
                choices=bg_source_fc_choices,
                value=(
                    BGSourceFC.CUSTOM.value if is_img2img else BGSourceFC.NONE.value
                ),
                type="value",
                visible=is_img2img,
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

        state = gr.State({})
        (
            ICLightForge.a1111_context.img2img_submit_button
            if is_img2img
            else ICLightForge.a1111_context.txt2img_submit_button
        ).click(
            fn=lambda *args: {
                k: v
                for k, v in zip(
                    vars(self.DEFAULT_ARGS).keys(),
                    args,
                )
            },
            inputs=[
                enabled,
                model_type,
                input_fg,
                uploaded_bg,
                bg_source_fc,
                bg_source_fbc,
            ],
            outputs=state,
            queue=False,
        )

        if is_img2img:

            def update_img2img_input(bg_source_fc: str, height: int, width: int):
                bg_source_fc = BGSourceFC(bg_source_fc)
                if bg_source_fc == BGSourceFC.CUSTOM:
                    return gr.skip()

                return gr.update(
                    value=bg_source_fc.get_bg(image_width=width, image_height=height)
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

        def on_model_change(model_type: str):
            match ModelType.get(model_type):
                case ModelType.FC:
                    return (
                        gr.update(visible=is_img2img),
                        gr.update(visible=False),
                        gr.update(visible=False),
                    )
                case ModelType.FBC:
                    return (
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True),
                    )
                case _:
                    raise SystemError

        model_type.change(
            fn=on_model_change,
            inputs=[model_type],
            outputs=[bg_source_fc, bg_source_fbc, uploaded_bg],
            show_progress=False,
        )

        return (state,)

    def process_before_every_sampling(self, p, *args, **kwargs):
        args = ICLightArgs.fetch_from(p)
        if not args.enabled:
            return

        device = torch.device("cuda")
        input_rgb: np.ndarray = run_rmbg(args.input_fg)

        work_model: ModelPatcher = p.sd_model.forge_objects.unet.clone()
        vae = p.sd_model.forge_objects.vae.clone()
        ic_model_state_dict = load_torch_file(args.model_type.path, device=device)
        ic_light = ICLight()

        patched_unet: ModelPatcher = ic_light.apply(
            model=work_model,
            ic_model_state_dict=ic_model_state_dict,
            c_concat=args.get_c_concat(input_rgb, vae, p, device=device),
        )[0]

        p.sd_model.forge_objects.unet = patched_unet
        p.extra_result_images.append(input_rgb)

    @staticmethod
    def on_after_component(component, **_kwargs):
        """Register the A1111 component."""
        ICLightForge.a1111_context.set_component(component)


script_callbacks.on_after_component(ICLightForge.on_after_component)
