import random
import numpy as np
from scipy.spatial.transform import Rotation as R

goal_pos = np.array(
    [0, 0.75, 0.75]
)  # euler = 0 0 0 durumunda aşağı bakıyor. ilerisi y, yukarısı z, sağ taraf x
poses = np.zeros((100, 4, 4))

with open("./test2.txt", "w") as the_file:
    for i in range(10):
        posx = random.randrange(-100, 100) / 100
        posy = random.randrange(-100, 100) / 100 + 0.5
        posz = random.randrange(0, 20) / 10 + 0.5
        # pos = np.array([posx, posy, posz])

        # r = - pos + goal_pos - np.array([0, 0, -1])

        # x'teki değişim kadar z değişecek. z = atan(r(y) / r(x))
        # pitch = np.arcsin(r[2])
        """
        yaw = np.arctan2(r[1], r[0]) + np.pi/2
        pitch = -np.arctan2(np.sqrt(r[0]**2 + r[1]**2), r[2])
        roll = np.arctan2(np.sqrt(r[2]**2 + r[1]**2), r[0]) / 104

        r = R.from_euler('xyz', np.array([pitch, roll, yaw]), degrees=False)
        cam_rotation = r.as_matrix()
        cam_position = np.array([posx, posy, posz])
        poses[i, :3, :3] = cam_rotation
        poses[i, :3, 3] = cam_position
        poses[i, 3, 3] = 1
"""

        # the_file.write(f'<camera name="CUSTOM{i}" fovy="45" euler="{pitch} 0 {yaw}" pos="{posx} {posy} {posz}"/>\n')
        the_file.write(
            f'<camera name="CUSTOM{i}" fovy="45" mode="targetbody" target="buttonbox" pos="{posx} {posy} {posz}"/>\n'
        )
# np.save("/home/yigit/Metaworld/NERF_Dataset/poses", poses)
