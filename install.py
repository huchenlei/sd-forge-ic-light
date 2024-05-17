import launch

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg==2.0.50 --no-deps", "rembg")

for dep in ("onnxruntime", "pymatting", "pooch"):
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for rembg")

if not launch.is_installed("pydantic"):
    launch.run_pip("install pydantic~=1.10.11", "pydantic for ic-light")
