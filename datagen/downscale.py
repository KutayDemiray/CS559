import torch
import torch.nn.functional as F
import torchvision.transforms as TF

import os
import shutil

from PIL import Image

input_path = "../nerf_pretrain/data/drawer-open-v2"
output_path = "../nerf_pretrain/data/drawer-open-v2-128"

to_tensor = TF.ToTensor()
to_pil = TF.ToPILImage()

if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(os.path.join(output_path, "rgb")):
    os.mkdir(os.path.join(output_path, "rgb"))

if not os.path.exists(os.path.join(output_path, "seg")):
    os.mkdir(os.path.join(output_path, "seg"))

if not os.path.exists(os.path.join(output_path, "depth")):
    os.mkdir(os.path.join(output_path, "depth"))

for subdir, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith(".png"):
            file_path = os.path.join(subdir, file)
            img = Image.open(file_path)
            img = to_tensor(img)
            img = F.interpolate(img[None, ...], (128, 128), mode="nearest").view(
                -1, 128, 128
            )
            img = to_pil(img)
            out_path = os.path.join(output_path, subdir.split("/")[-1], file)
            img.save(out_path)

        elif file.endswith(".json"):
            file_path = os.path.join(subdir, file)
            out_path = os.path.join(output_path, file)
            shutil.copy(file_path, out_path)
