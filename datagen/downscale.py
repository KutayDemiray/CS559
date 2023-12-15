import torch
import torch.nn.functional as F
import torchvision.transforms as TF

import os
import shutil
import numpy as np

from PIL import Image

input_path = "../nerf_pretrain/data/drawer-open-v2"
output_path = "../nerf_pretrain/data/drawer-open-v2-128"

to_tensor = TF.ToTensor()
to_pil = TF.ToPILImage()


# kutay
def one_hot_affordance(obs: np.ndarray, color: np.ndarray = np.array([156, 104, 125])):
    # obs = obs.permute(1, 2, 0).numpy()
    binary_seg = np.zeros((128, 128))

    interest = (obs == color).all(axis=2)
    binary_seg[interest] = 1
    binary_seg[binary_seg >= 0.5] = 1
    binary_seg[binary_seg < 0.5] = 0

    binary_seg = torch.tensor(binary_seg)

    return binary_seg


if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(os.path.join(output_path, "rgb")):
    os.mkdir(os.path.join(output_path, "rgb"))

if not os.path.exists(os.path.join(output_path, "seg")):
    os.mkdir(os.path.join(output_path, "seg"))

if not os.path.exists(os.path.join(output_path, "depth")):
    os.mkdir(os.path.join(output_path, "depth"))

if not os.path.exists(os.path.join(output_path, "mask")):
    os.mkdir(os.path.join(output_path, "mask"))

for subdir, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith(".png"):
            file_path = os.path.join(subdir, file)
            img = Image.open(file_path)
            img = img.resize((128, 128), resample=Image.NEAREST)

            if subdir.split("/")[-1] == "seg":
                mask_path = os.path.join(output_path, "mask", file)
                mask = one_hot_affordance(np.array(img))
                mask = to_pil(mask)
                mask.save(mask_path)

            # img = to_tensor(img)
            # img = F.interpolate(img[None, ...], (128, 128), mode="nearest").view(
            #    -1, 128, 128
            # )

            # img = to_pil(img)
            out_path = os.path.join(output_path, subdir.split("/")[-1], file)

            img.save(out_path)

        elif file.endswith(".json"):
            file_path = os.path.join(subdir, file)
            out_path = os.path.join(output_path, file)
            shutil.copy(file_path, out_path)
