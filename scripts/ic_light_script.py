from dataclasses import dataclass
import gradio as gr
import numpy as np
from typing import Optional, Tuple

from modules import scripts, script_callbacks
from modules.ui_components import InputAccordion
from modules.processing import StableDiffusionProcessing

from libiclight.args import ICLightArgs, BGSourceFC, BGSourceFBC, ModelType


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


class ICLightScript(scripts.Script):
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
                    source="upload",
                    type="numpy",
                    label="Foreground",
                    height=480,
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
                value=True,
                interactive=True,
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

        state = gr.State({})
        (
            ICLightScript.a1111_context.img2img_submit_button
            if is_img2img
            else ICLightScript.a1111_context.txt2img_submit_button
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
                remove_bg,
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
                ICLightScript.a1111_context.img2img_h_slider,
                ICLightScript.a1111_context.img2img_w_slider,
            ):
                component.change(
                    fn=update_img2img_input,
                    inputs=[
                        bg_source_fc,
                        ICLightScript.a1111_context.img2img_h_slider,
                        ICLightScript.a1111_context.img2img_w_slider,
                    ],
                    outputs=ICLightScript.a1111_context.img2img_image,
                )

        def shift_enum_radios(model_type: str):
            model_type = ModelType(model_type)
            if model_type == ModelType.FC:
                return gr.update(visible=True), gr.update(visible=False)
            else:
                assert model_type == ModelType.FBC
                return gr.update(visible=False), gr.update(visible=True)

        model_type.change(
            fn=shift_enum_radios,
            inputs=[model_type],
            outputs=[bg_source_fc, bg_source_fbc],
            show_progress=False,
        )

        model_type.change(
            fn=lambda model_type: gr.update(
                visible=ModelType(model_type) == ModelType.FBC
            ),
            inputs=[model_type],
            outputs=[uploaded_bg],
            show_progress=False,
        )

        return (state,)

    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *args, **kwargs
    ):
        """Forge impl."""
        args = ICLightArgs.fetch_from(p)
        if not args.enabled:
            return

        from libiclight.forge_backend import apply_ic_light

        apply_ic_light(p, args)

    @staticmethod
    def on_after_component(component, **_kwargs):
        """Register the A1111 component."""
        ICLightScript.a1111_context.set_component(component)


script_callbacks.on_after_component(ICLightScript.on_after_component)
