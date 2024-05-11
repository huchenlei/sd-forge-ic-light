# sd-forge-ic-light
SD Forge extension for [IC-Light](https://github.com/lllyasviel/IC-Light). This extension is based on https://github.com/huchenlei/ComfyUI-IC-Light-Native.

## Install
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/608fbe20-1430-4efa-93bc-166f629eaaa5)

### Download models
IC-Light main repo is based on diffusers. In order to load it with UnetLoader in Forge, state_dict keys need to convert to ldm format. You can download models with ldm keys here: https://huggingface.co/huchenlei/IC-Light-ldm/tree/main

There are 2 models:
- iclight_sd15_fc_unet_ldm: Use this in FG workflows
- iclight_sd15_fbc_unet_ldm: Use this in BG workflows

After you download these models, please put them under `stable-diffusion-webui-forge/models/unet`. You might want to manually create the `unet` folder.

## How to use
For best result, it is recommended to use low CFG and strong denosing strength.

### Given FG, Generate BG and relight [Txt2Img][HR availble]
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/00fbae46-b5cf-4415-89ac-5b23b1a8f463)

### Given FG and light map, Genereate BG and relight [Img2Img]
After you select value from the radio, the img2img input will automatically be set by the extension. 
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/1b9e9c87-e8ef-4505-ab04-ade37336a8a3)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/618ba4d4-5df7-4084-bdf1-44927f77a581)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/899bf635-1aac-40e5-bf4f-ca801e7922d5)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/4b768af2-c9ac-4fc2-9762-a2df45ec3371)

### Given FG and BG, Put FG on BG and relight [Txt2Img][HR available]
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/ff6dcbb7-9f8a-4918-b3dd-aaaa85ea04ce)

## TODOs
- Add API support
- Add infotext support
- Add image with bg removed to output
- Add option to skip remove bg operation
