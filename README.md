# SD Forge IC-Light
This is an Extension for the [Forge Webui](https://github.com/lllyasviel/stable-diffusion-webui-forge), which implements [IC-Light](https://github.com/lllyasviel/IC-Light), allowing you to manipulate the illumination of images.

> This only works with SD 1.5 checkpoints

> Now supports **[Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)**! You will need to install [sd-webui-model-patcher](https://github.com/huchenlei/sd-webui-model-patcher) first for it to work.

## Getting Started
1. Download the <ins>two</ins> models from [Releases](https://github.com/Haoming02/sd-forge-ic-light/releases)
2. Create a new folder, `ic-light`, inside your webui `models` folder
3. Place the 2 models inside said folder
4. **(Optional)** You can rename the models, as long as the filenames contain either **`fc`** or **`fbc`**

## How to use
W.I.P

## Known Issue
- In **Automatic1111** implementation, **Hires. Fix** is not supported yet
- `Restore Details` does not work when input and output resolution are different

<hr>

## Based on:
- https://github.com/lllyasviel/IC-Light
- https://github.com/huchenlei/sd-forge-ic-light
