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

## [2024-06-06] Major Update Release Note [PR https://github.com/huchenlei/sd-forge-ic-light/pull/24]

- Load models from `ic-light` folder with arbitrary filenames. You need to rename your `unet` folder to `ic-light`.
- Use the pre-built `rembg` package instead of diffusers version.
- **New Feature:** Implement *Difference of Gaussians* to reintroduce some details, e.g. text, after the processing.
- **New Feature:** Implement reinforce-fg option which allows better preservation of fg base color.

## How to use
For best result, it is recommended to use low CFG and strong denosing strength.

### Given FG, Generate BG and relight [Txt2Img][HR available]
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/53f4041a-c8d3-4950-9579-596df5121d8e)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/d61e2f08-a29f-46ca-b1d3-cd447e489698)

Infotext:
```
sunshine from window, a beautiful girl, absurdres, highres, (masterpiece:1.2), (best quality, highest quality),
Negative prompt: (lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green
Steps: 25, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 2, Seed: 3615474644, Size: 512x768, Model hash: e6415c4892, Model: realisticVisionV20_v20, Clip skip: 2, Version: v1.9.3-13-g8e355f08
```

### Given FG and light map, Genereate BG and relight [Img2Img]
Img2Img input image is lightmap. After you select value from the radio, the img2img input will automatically be set by the extension. You can also upload your own lightmap by selecting `Custom`.
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/72321ae9-e5c5-448c-85c5-dc0326d9559d)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/1f92f432-b56f-477f-8a55-6976f4818a43)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/81d93520-5118-4b93-927c-6199e8696f80)

Infotext:
```
sunshine from window, a beautiful girl, absurdres, highres, (masterpiece:1.2), (best quality, highest quality),
Negative prompt: (lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green
Steps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 2, Seed: 2984374687, Size: 512x768, Model hash: e6415c4892, Model: realisticVisionV20_v20, Denoising strength: 0.96, Clip skip: 2, Version: v1.9.3-13-g8e355f08
```

### Given FG and BG, Put FG on BG and relight [Txt2Img][HR available]
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/a0776fe9-f8f6-49d2-8d6a-a86354ba44f7)
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/6bd71534-5e8b-4eb8-887b-17642864341c)

Infotext:
```
a beautiful girl, absurdres, highres, (masterpiece:1.2), (best quality, highest quality),
Negative prompt: (lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green
Steps: 25, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 2, Seed: 2230971215, Size: 512x768, Model hash: e6415c4892, Model: realisticVisionV20_v20, Clip skip: 2, Version: v1.9.3-13-g8e355f08
```

### [2024-06-06] Restore Detail
Detail transfer was originally implemented in https://github.com/kijai/ComfyUI-IC-Light. It captures high frequency details, e.g. text, in the input fg image and reapplys them to the output image.
![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/8b63a088-8324-4292-8487-ad555b2dc73f)

Original output:

![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/825154aa-eb51-4b51-bf7d-ea07d1945b21)

After detail restore:

![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/0836f4b5-52f6-4af0-9a68-d65298ba80e2)

Infotext:
```
A bottle of oyster sauce, kitchen counter, absurdres, highres, (masterpiece:1.2), (best quality, highest quality),
Negative prompt: (lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green
Steps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 2, Seed: 2984374687, Size: 768x768, Model hash: e6415c4892, Model: realisticVisionV20_v20, Denoising strength: 0.96, Clip skip: 2, Version: v1.9.3-13-g8e355f08
```

### [2024-06-06] Reinforce FG
A big problem of IC-Light is that it often alters the FG object's base color too much. By adding the fg image on top of the lightmap, this issue can be alleviated. This essentially implements this comfyui [workflow](https://github.com/huchenlei/ComfyUI-IC-Light-Native/blob/main/examples/ic_light_preserve_color.json).
Here is a comparison:

Without reinforce-fg: You can observe that the fg object almost looks transparent.

![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/2d108f57-af2b-460e-82e6-91f9e3286374)

With reinforce-fg (Same seed): The fg object no longer look transparent.

![image](https://github.com/huchenlei/sd-forge-ic-light/assets/20929282/4d503341-6354-4edf-9830-35fe18e7fcad)

### Skip remove bg
If the default remove bg cannot achieve your desired effect, you can use other tools to create an RGBA image and uncheck the remove bg checkbox. Image with grey background can also be used the same way as RGBA image.

## Known Issue

- For **Automatic1111** implementation, **Hires. Fix** is not supported yet. This is caused by A1111 code structure, which is hard to modify. We might want to make our own fork of A1111 later to make it work, as PR is not likely to get merged in A1111 repo.
- `Restore Details` does not work properly when the input and output aspect ratios are different
