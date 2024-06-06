# SD Forge IC-Light
This is an Extension for the [Forge Webui](https://github.com/lllyasviel/stable-diffusion-webui-forge), which implements [IC-Light](https://github.com/lllyasviel/IC-Light), allowing you to manipulate the illumination of images.

> This only works with SD 1.5 checkpoints

> Now supports [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)[*](#getting-started)

## Getting Started
0. For **Automatic1111**, install [sd-webui-model-patcher](https://github.com/huchenlei/sd-webui-model-patcher) first
1. Download the <ins>two</ins> models from [Releases](https://github.com/Haoming02/sd-forge-ic-light/releases)
2. Create a new folder, `ic-light`, inside your webui `models` folder
3. Place the 2 models inside the said folder
4. **(Optional)** You can rename the models, as long as the filenames contain either **`fc`** or **`fbc`**

## How to use
To get the best result, it is recommended to use low CFG (2.0) and strong denoising strength.

### Given FG, Generate BG and relight [Txt2Img][HR available]
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/00fbae46-b5cf-4415-89ac-5b23b1a8f463)

### Given FG and light map, Genereate BG and relight [Img2Img]
After you select the value from the radio, the img2img input will automatically be set by the extension.
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/1b9e9c87-e8ef-4505-ab04-ade37336a8a3)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/618ba4d4-5df7-4084-bdf1-44927f77a581)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/899bf635-1aac-40e5-bf4f-ca801e7922d5)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/4b768af2-c9ac-4fc2-9762-a2df45ec3371)

### Given FG and BG, Put FG on BG and relight [Txt2Img][HR available]
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/c5e57d36-8191-412c-8eb3-3ba1bc109571)

### Skip remove bg
If the default remove bg cannot achieve your desired effect, you can use other tools to create an RGBA image
and uncheck the remove bg checkbox.
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/fc6c583e-9de5-4555-ac36-48ca3f47fce7)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/dbf24894-2cfe-4d61-9529-9d2620380f0d)

## [2024-06-06] Major Update Release Note

- Load models from `ic-light` folder with arbitrary filenames. You need to rename your `unet` folder to `ic-light`.Use the pre-built `rembg` package instead of diffusers version.
- **New Feature:** Implement *Difference of Gaussians* to reintroduce some details, e.g. text, after the processing.
- **New Feature:** Implement reinforce-fg option which allows better preservation of fg base color.

## Known Issue

- For **Automatic1111** implementation, **Hires. Fix** is not supported yet. This is caused by A1111 code structure, which is hard to modify. We might want to make our own fork of A1111 later to make it work, as PR is not likely to get merged in A1111 repo.
- `Restore Details` does not work properly when the input and output aspect ratios are different
