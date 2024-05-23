from ldm_patched.modules.model_management import get_torch_device
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.utils import load_torch_file

from modules import scripts, script_callbacks
from modules.ui_components import InputAccordion
from modules.api import api

from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)

from scripts.model_loader import ModelType
from scripts.ic_modes import t2i_fc, t2i_fbc, i2i_fc

from libiclight.detail_utils import restore_detail
from libiclight.ic_light_nodes import ICLight
from libiclight.rembg_utils import run_rmbg
from libiclight.utils import (
    align_dim_latent,
    make_masked_area_grey,
    resize_and_center_crop,
    forge_numpy2pytorch,
)

from pydantic import BaseModel, validator
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from PIL import Image
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

    def get_c_concat(
        self,
        processed_fg: np.ndarray,  # fg with bg removed.
        vae,
        p: StableDiffusionProcessing,
        device: torch.device,
    ) -> dict:

        if getattr(p, "is_hr_pass", False):
            assert isinstance(p, StableDiffusionProcessingTxt2Img)

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

    def __init__(self):
        self.args: ICLightArgs = None

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
                model_type = gr.Dropdown(
                    label="Mode",
                    choices=model_type_choices,
                    value=ModelType.FC.name,
                    interactive=(not is_img2img),
                )

                desc = gr.Markdown(i2i_fc if is_img2img else t2i_fc)

            with gr.Row():
                input_fg = gr.Image(
                    source="upload",
                    type="numpy",
                    label=("Lighting Conditioning" if is_img2img else "Foreground"),
                    height=480,
                    interactive=True,
                    visible=True,
                    image_mode="RGBA",
                )
                uploaded_bg = gr.Image(
                    source="upload",
                    type="numpy",
                    label="Background",
                    height=480,
                    interactive=True,
                    visible=False,
                )

            remove_bg = gr.Checkbox(
                label="Background Removal",
                info="Disable if you already have a subject with the background removed",
                value=True,
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
                choices=[BGSourceFBC.UPLOAD.value, BGSourceFBC.UPLOAD_FLIP.value],
                value=BGSourceFBC.UPLOAD.value,
                type="value",
                visible=False,
                interactive=True,
            )

            with InputAccordion(value=False, label="Restore Details") as restore_detail:

                use_rmbg = gr.Checkbox(
                    label="Use the [Image with Background Removed] instead of the [Original Input]"
                )

                blur_radius = gr.Slider(
                    label="Blur Radius",
                    info="for Difference of Gaussian",
                    minimum=1,
                    maximum=9,
                    step=2,
                    value=5,
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
                restore_detail,
                use_rmbg,
                blur_radius,
                input_fg,
                uploaded_bg,
                bg_source_fc,
                bg_source_fbc,
                remove_bg,
            ],
            outputs=state,
            queue=False,
        )

        if is_img2img:

            def update_img2img_input(bg_source_fc: str):
                bg_source_fc = BGSourceFC(bg_source_fc)
                if bg_source_fc == BGSourceFC.CUSTOM:
                    return gr.skip()

                return gr.update(value=bg_source_fc.get_bg(512, 512))

            bg_source_fc.change(
                fn=update_img2img_input,
                inputs=[bg_source_fc],
                outputs=[input_fg],
            )

        else:

            def on_model_change(model_type: str):
                match ModelType.get(model_type):
                    case ModelType.FC:
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(value=t2i_fc),
                        )
                    case ModelType.FBC:
                        return (
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(value=t2i_fbc),
                        )
                    case _:
                        raise SystemError

            model_type.change(
                fn=on_model_change,
                inputs=[model_type],
                outputs=[bg_source_fbc, uploaded_bg, desc],
                show_progress=False,
            )

        return (state,)

    def before_process(self, p, *args, **kwargs):
        args = ICLightArgs.fetch_from(p)
        if not args.enabled:
            self.args = None
            return

        if isinstance(p, StableDiffusionProcessingImg2Img):
            input_image = np.asarray(p.init_images[0]).astype(np.uint8)
            p.init_images[0] = Image.fromarray(args.input_fg)
            args.input_fg = input_image

        self.args = args

    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *args, **kwargs
    ):
        if (self.args is None) or (not self.args.enabled):
            return

        device = get_torch_device()

        if self.args.remove_bg:
            input_rgb: np.ndarray = run_rmbg(self.args.input_fg)
        else:
            if len(self.args.input_fg.shape) < 3:
                raise NotImplementedError
            if self.args.input_fg.shape[2] == 4:
                input_rgb = make_masked_area_grey(
                    self.args.input_fg[..., :3],
                    self.args.input_fg[..., 3:].astype(np.float32) / 255.0,
                )
            else:
                input_rgb = self.args.input_fg

        self.input_rgb = (
            input_rgb
            if (self.args.restore_detail and self.args.use_rmbg_for_restore)
            else None
        )

        assert input_rgb.shape[2] == 3

        work_model: ModelPatcher = p.sd_model.forge_objects.unet.clone()
        vae = p.sd_model.forge_objects.vae.clone()
        ic_model_state_dict = load_torch_file(self.args.model_type.path, device=device)
        ic_light = ICLight()

        patched_unet: ModelPatcher = ic_light.apply(
            model=work_model,
            ic_model_state_dict=ic_model_state_dict,
            c_concat=self.args.get_c_concat(input_rgb, vae, p, device=device),
        )[0]

        p.sd_model.forge_objects.unet = patched_unet

        if not getattr(p, "is_hr_pass", False):
            p.extra_result_images.append(input_rgb)

    def postprocess_image(self, p, pp, *args, **kwargs):
        if (
            (self.args is None)
            or (not self.args.enabled)
            or (not self.args.restore_detail)
        ):
            return

        p.extra_result_images.append(
            restore_detail(
                np.asarray(pp.image).astype(np.uint8),
                (
                    self.input_rgb
                    if self.args.use_rmbg_for_restore
                    else self.args.input_fg
                ),
                int(self.args.blur_radius),
            )
        )

    @staticmethod
    def on_after_component(component, **_kwargs):
        """Register the A1111 component."""
        ICLightForge.a1111_context.set_component(component)


script_callbacks.on_after_component(ICLightForge.on_after_component)
