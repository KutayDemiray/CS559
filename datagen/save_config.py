import numpy as np
import json
import os


def save_single_config(
    path: str, ep: int, timestep: int, n_cams: int, cam_poses: np.ndarray
):
    transform = {}
    transform["camera_angle_y"] = np.pi / 4
    transform["frames"] = [
        {
            "file_path": f"./rgb/rgb_e{ep:03d}_t{timestep:03d}_cam{cam:02d}.png",
            "rotation": 0.012566370614359171,
            "transform_matrix": cam_poses[cam].tolist(),
        }
        for cam in range(n_cams)
    ]

    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path, f"e{ep:03d}_t{timestep:03d}.json"), "w") as file:
        json_obj = json.dumps(transform, indent=4)
        file.write(json_obj)


def save_full_config(
    path: str,
    n_eps: int,
    n_timesteps: int,
    n_cams: int,
    cam_poses: np.ndarray,
    val_eps: np.ndarray = None,
    test_eps: np.ndarray = None,
):
    if not os.path.exists(f"{path}/rgb"):
        os.mkdir(f"{path}/rgb")

    train_path = f"{path}/{'transforms' if val_eps is None and test_eps is None else 'transforms_train'}.json"

    transforms = {}
    transforms["camera_angle_y"] = np.pi / 4
    transforms["frames"] = [
        {
            "file_path": f"./rgb/rgb_e{ep:03d}_t{t:03d}_cam{cam:02d}.png",
            "rotation": 0.012566370614359171,
            "transform_matrix": cam_poses[cam].tolist(),
        }
        for cam in range(n_cams)
        for ep in range(n_eps)
        for t in range(n_timesteps)
    ]

    with open(train_path, "w") as file:
        json_obj = json.dumps(transforms, indent=4)
        file.write(json_obj)
