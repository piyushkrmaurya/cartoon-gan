import json


import numpy as np
import requests
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from io import StringIO, BytesIO

from flask import Flask, render_template, request, send_file

from predict import predict_images
from models.generator import Generator
from utils.transforms import get_no_aug_transform

app = Flask(__name__, template_folder="./")

user_stated_device = "cpu"
device = torch.device(user_stated_device)
pretrained_dir = "./checkpoints/trained_netG.pth"
netG = Generator().to(device)
netG.eval()
if user_stated_device == "cuda":
    netG.load_state_dict(torch.load(pretrained_dir))
else:
    netG.load_state_dict(torch.load(pretrained_dir, map_location=torch.device('cpu')))


def inv_normalize(img):
    mean = torch.Tensor([0.485+0.1, 0.456+0.1, 0.406+0.1]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    img = img.clamp(0, 1)
    return img

def predict_images(image_list):
    trf = get_no_aug_transform()
    image_list = torch.from_numpy(np.array([trf(img).numpy() for img in image_list])).to(device)

    with torch.no_grad():
        generated_images = netG(image_list)
    generated_images = inv_normalize(generated_images)

    pil_images = []
    for i in range(generated_images.size()[0]):
        generated_image = generated_images[i].cpu()
        pil_images.append(TF.to_pil_image(generated_image))
    return pil_images

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sample")
def sample():
    return render_template("sample.html")


@app.route("/api/", methods=["POST"])
def api():
    if request.get_json() and "url" in request.get_json():
        url = request.get_json().get("url")

        image_bytes = BytesIO()
        response = requests.get(url, stream=True)

        if response.ok:
            for block in response.iter_content(1024):
                if not block:
                    break
                image_bytes.write(block)
        converted_bytes = image_bytes.getvalue()
        img = Image.open(BytesIO(converted_bytes))

    elif request.files.get("file", None):
        img = Image.open(request.files['file'].stream)

    else:
        return "Error"


    img = img.convert("RGB")
    output = predict_images([img])[0]
    output.save("output.png")

    img_io = BytesIO()
    output.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


if __name__ == "__main__":
    app.run()
