import io
import cv2
import base64
import requests
from PIL import Image
from pathlib import Path


def send_request(url, payload):
    response = requests.post(url=url, json=payload)
    image_string = response.json()["images"][0]
    return Image.open(io.BytesIO(base64.b64decode(image_string)))


def read_image(img_path) -> str:
    img = cv2.imread(img_path)
    retval, bytes = cv2.imencode(".png", img)
    encoded_image = base64.b64encode(bytes).decode("utf-8")
    return encoded_image


url = "http://localhost:7860/sdapi/v1/txt2img"
body = {
    "prompt": "a beautiful woman, sunshine from window",
    "negative_prompt": "lowres, bad anatomy, bad hands, cropped, worst quality",
    "batch_size": 1,
    "steps": 20,
    "cfg_scale": 2,
    "width": 512,
    "height": 768,
    "alwayson_scripts": {
        "IC Light": {
            "args": [
                {
                    "enabled": True,
                    "model_type": "FC",
                    "input_fg": read_image(
                        str(Path(__file__).parent / "images" / "i3.png")
                    ),
                    "bg_source_fc": "None",
                }
            ],
        },
    },
}


if __name__ == "__main__":
    image = send_request(url=url, payload=body)
    image.show()
