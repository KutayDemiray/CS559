import numpy as np
import json
import os

import numpy as np


def save_single_config(
    path: str, ep: int, timestep: int, n_cams: int, cam_poses: np.ndarray
):
    os.mkdir(os.path.join(path, "intrinsics"))
    os.mkdir(os.path.join(path, "pose"))

    f = 0.5 * 480 / np.tan(np.pi / 8)
    camera_matrix_flat = [f, 0, 240, 0, f, 240, 0, 0, 1]
    camera_matrix_str = " ".join(camera_matrix_flat)

    for i in range(0, n_cams):
        with open(os.path.join(path, "intrinsics", f"{i:06d}.txt")) as file:
            file.write(camera_matrix_str)

        with open(os.path.join(path, "pose", f"{i:06d}.txt")) as file:
            pose = cam_poses[i]
            pose = pose.flatten()
            pose = list(pose)
            pose_str = " ".join(pose)
            file.write(pose_str)

    """
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
    """
