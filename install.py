import launch

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg==2.0.50 --no-deps", "rembg")

if not launch.is_installed("onnxruntime"):
    launch.run_pip("install onnxruntime", "rembg")
