# import mujoco_py

from PIL import Image
from matplotlib import cm
import numpy as np

import metaworld
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
from metaworld.policies.sawyer_soccer_v2_policy import SawyerSoccerV2Policy
from metaworld.policies.sawyer_hammer_v2_policy import SawyerHammerV2Policy
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
import random

import time

from config_yaz import *

import sys
import argparse
import cv2
import math

from save_config import *

from scipy.spatial.transform import Rotation as R

from datetime import datetime

parser = argparse.ArgumentParser()
# environment
parser.add_argument("--env_name", default="window-open-v2")
parser.add_argument(
    "--save_mode", choices=["rgb_array", "rgbd", "rgbd_segmentation_array", "human"]
)  # rgb_array returns rgb + d + seg (cursed)
parser.add_argument("--data_folder", default="../nerf_pretrain/data")
parser.add_argument("--n_episodes", default=30)
parser.add_argument("--n_timesteps", default=120)
parser.add_argument("--epsilon", default=0.5)
parser.add_argument("--n_cams", default=3)
parser.add_argument("--timestep_freq", default=1)
parser.add_argument("--img_size", default=480, type=int)

args = parser.parse_args()

valid_tasks = [
    "window-open-v2",
    "drawer-open-v2",
    "hammer-v2",
    "soccer-v2",
]  # tasks used in snerl

args.save_mode = args.save_mode
task_name = args.env_name
image_save = args.save_mode is not "human"
data_folder = args.data_folder
render_mode = args.save_mode
n_episodes = int(args.n_episodes)
n_timesteps = int(args.n_timesteps)
n_cams = int(args.n_cams)
epsilon = float(args.epsilon)
timestep_freq = int(args.timestep_freq)

print(args)

assert task_name in valid_tasks, "task name not in valid tasks"

ml1 = metaworld.ML1(task_name)  # Construct the benchmark, sampling tasks

env = ml1.train_classes[task_name](
    render_mode=render_mode
)  # Create an environment with task `window_open`

task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

# print("class env", type(env))
if task_name == "window-open-v2":
    policy = SawyerWindowOpenV2Policy()
elif task_name == "soccer-v2":
    policy = SawyerSoccerV2Policy()
elif task_name == "drawer-open-v2":
    policy = SawyerDrawerOpenV2Policy()
elif task_name == "hammer-v2":
    policy = SawyerHammerV2Policy()


def get_poses(cam_positions, cam_rotations):
    n_cams = cam_positions.shape[0]
    ret = np.zeros((n_cams, 4, 4))
    for i in range(n_cams):
        rot = cam_rotations[i].reshape((3, 3))
        pos = cam_positions[i]
        ret[i, 0:3, 0:3] = rot
        ret[i, 0:3, 3] = pos.T
        ret[i, 3, 3] = 1

    return ret


fovy = 45
height = args.img_size
width = args.img_size
focal = 0.5 * height / np.tan(fovy * np.pi / 180)

cam_positions = env.mujoco_renderer.data.cam_xpos[4:-2]
cam_rotations = env.mujoco_renderer.data.cam_xmat[4:-2]
poses = get_poses(cam_positions, cam_rotations)
print(poses.shape)

# print(poses[0])


a = R.from_matrix(poses[0][:3, :3])
print(a.as_euler("xyz", degrees=True))
print(a.as_euler("xyz", degrees=False))
# print(env.camera_name)

if not os.path.exists(f"{data_folder}/{task_name}"):
    os.mkdir(f"{data_folder}/{task_name}")

if not os.path.exists(f"{data_folder}/{task_name}/rgb"):
    os.mkdir(f"{data_folder}/{task_name}/rgb")

if not os.path.exists(f"{data_folder}/{task_name}/depth"):
    os.mkdir(f"{data_folder}/{task_name}/depth")

if not os.path.exists(f"{data_folder}/{task_name}/seg"):
    os.mkdir(f"{data_folder}/{task_name}/seg")

# images = np.zeros((n_episodes * n_timesteps, args.img_size, args.img_size, 3))


for ep in range(n_episodes):
    ts = datetime.timestamp(datetime.now())
    print(f"[{datetime.fromtimestamp(ts)}] Episode {ep + 1} of {n_episodes}")
    obs, info = env.reset()  # Reset environment

    terminated = False
    truncated = False
    total_reward = 0

    for t in range(n_timesteps * timestep_freq):
        # epsilon greedy
        p = random.random()
        if p < epsilon:
            a = env.action_space.sample()
        else:
            a = policy.get_action(obs=obs)

        # RL step
        obs, reward, terminated, truncated, info = env.step(
            a
        )  # Step the environment with the sampled random action

        total_reward += reward

        # Save observation
        if image_save and t % timestep_freq == 0:
            for camId in range(n_cams):
                env.camera_name = f"CUSTOM{camId}"

                rgb, seg, depth = env.render()

                rgb = Image.fromarray(rgb)
                seg = Image.fromarray(seg)
                depth = Image.fromarray(depth)

                # resize images to desired size
                if args.img_size != 480:
                    rgb = rgb.resize((args.img_size, args.img_size))
                    seg = seg.resize((args.img_size, args.img_size))
                    depth = depth.resize((args.img_size, args.img_size))

                rgb.save(
                    f"{data_folder}/{task_name}/rgb/rgb_e{ep:03d}_t{t:03d}_cam{camId:02d}.png"
                )

                if (
                    render_mode == "rgbd"
                    or render_mode == "rgbd_segmentation_array"
                    or render_mode == "rgb_array"
                ):
                    """
                    depth = rgb_img[:, :, 3]
                    depth -= np.min(depth)
                    depth /= 2 * depth[depth <= 1].mean()
                    depth = 255 * np.clip(depth, 0, 1)
                    depth = Image.fromarray(depth.astype(np.uint8))
                    """

                    depth.save(
                        f"{data_folder}/{task_name}/depth/d_e{ep:03d}_t{t:03d}_cam{camId:02d}.png"
                    )

                if (
                    render_mode == "rgbd_segmentation_array"
                    or render_mode == "rgb_array"
                ):
                    seg.save(
                        f"{data_folder}/{task_name}/seg/seg_e{ep:03d}_t{t:03d}_cam{camId:02d}.png"
                    )

            save_single_config(
                path=f"{data_folder}/{task_name}",
                ep=ep,
                timestep=t,
                n_cams=n_cams,
                cam_poses=poses[:n_cams],
            )
        else:
            env.render()

        # episode end
        if truncated or terminated:
            print(total_reward)
            break
