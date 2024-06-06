# SD Forge IC-Light
This is an Extension for the [Forge Webui](https://github.com/lllyasviel/stable-diffusion-webui-forge), which implements [IC-Light](https://github.com/lllyasviel/IC-Light), allowing you to manipulate the illumination of images.

> This only works with SD 1.5 checkpoints

> Now supports [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)[*](#getting-started)

## Getting Started
0. For **Automatic1111**, install [sd-webui-model-patcher](https://github.com/huchenlei/sd-webui-model-patcher) first
1. Download the <ins>two</ins> models from [Releases](https://github.com/Haoming02/sd-forge-ic-light/releases)
2. Create a new folder, `ic-light`, inside your webui `models` folder
3. Place the 2 models inside said folder
4. **(Optional)** You can rename the models, as long as the filenames contain either **`fc`** or **`fbc`**

## How to use
W.I.P

## Main Differences of the Fork
- Reorganize UIs
- Add explanations on how each mode works
- Load models from `ic-light` folder with arbitrary filenames
- Use pre-built `rembg` package instead
- **Important:** Swap the way how `img2img` works: Now the input image is the subject while the extension takes lighting conditions.
    > This just makes more sense to me...
- **New:** Implement *Difference of Gaussians* to reintroduce some details after the processing

## Known Issue
- For **Automatic1111** implementation, **Hires. Fix** is not supported yet
- `Restore Details` does not work properly when the input and output aspect ratio is different

<hr>

## Based on:
- https://github.com/lllyasviel/IC-Light
- https://github.com/huchenlei/sd-forge-ic-light
