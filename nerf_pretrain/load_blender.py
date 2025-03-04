import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from PIL import Image

trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
    )
    return c2w


def load_blender_data(
    basedir, half_res=False, testskip=1, eps: int = 30, timesteps: int = 30
):
    splits = ["train", "val", "test"]
    splits = ["e%03d_t%03d" % (i, j) for i in range(eps) for j in range(timesteps)]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, "{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_semantics = []

    counts = [0]
    i = 1
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        semantics = []

        if "train" in s or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"]:
            # print("frame", i)
            fname = os.path.join(basedir, frame["file_path"])
            suffix = fname.split("/")[-1][4:]
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
            semantics.append(
                np.array(
                    Image.open(os.path.join(basedir, "seg", "seg_" + suffix))
                ).astype(np.float32)
                # / 50
            )
            i += 1

        imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        semantics = np.array(semantics).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(np.expand_dims(imgs, axis=1))
        all_poses.append(np.expand_dims(poses, axis=1))
        all_semantics.append(np.expand_dims(semantics, axis=1))

    print("merge splits done")

    imgs = np.concatenate(all_imgs, 1)
    poses = np.concatenate(all_poses, 1)
    semantics = np.concatenate(all_semantics, 1)

    # print(imgs.shape)
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    # imgs = np.concatenate(all_imgs, 1)
    # poses = np.concatenate(all_poses, 1)
    # print(imgs.shape)
    H, W = imgs[0, 0].shape[:2]
    camera_angle_y = float(meta["camera_angle_y"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_y)

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0

        imgs_half_res = np.zeros((imgs.shape[0], imgs.shape[1], H, W, 4))
        for i in range(len(imgs)):
            for j in range(len(imgs[0])):
                imgs_half_res[i, j] = cv2.resize(
                    imgs[i, j], (W, H), interpolation=cv2.INTER_AREA
                )
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    print("load blender done")

    return imgs, poses, render_poses, [H, W, focal], i_split, semantics
