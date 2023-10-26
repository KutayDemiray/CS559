import mujoco_py


import metaworld
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
from metaworld.policies.sawyer_soccer_v2_policy import SawyerSoccerV2Policy
from metaworld.policies.sawyer_hammer_v2_policy import SawyerHammerV2Policy
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
import random

import time

import sys

# from mujoco_py import GlfwContext

# GlfwContext(offscreen=True)  # Create a window to init GLFW.

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

valid_tasks = [
    "window-open-v2",
    "drawer-open-v2",
    "hammer-v2",
    "soccer-v2",
]  # tasks used in snerl

task_name = valid_tasks[3]

assert task_name in valid_tasks, "task name not in valid tasks"

ml1 = metaworld.ML1(task_name)  # Construct the benchmark, sampling tasks

env = ml1.train_classes[task_name](
    # render_mode="rgb_array"
)  # Create an environment with task `window_open`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

print("class env", type(env))
if task_name == "window-open-v2":
    policy = SawyerWindowOpenV2Policy()
elif task_name == "soccer-v2":
    policy = SawyerSoccerV2Policy()
elif task_name == "drawer-open-v2":
    policy = SawyerDrawerOpenV2Policy()
elif task_name == "hammer-v2":
    policy = SawyerHammerV2Policy()

epsilon = 0.0

while True:
    obs = env.reset()  # Reset environment
    done = False
    while not done:
        try:
            p = random.random()
            # print("p:", p)
            if p < epsilon:
                a = env.action_space.sample()  # Sample an action
            else:
                a = policy.get_action(obs=obs)
            obs, reward, done, info = env.step(
                a
            )  # Step the environment with the sampled random action
            # time.sleep(0.5)
            img = env.render()
            # print(type(img))
            # print(i)
        except ValueError:
            break

    print("episode done")

    # get rgb image (doesn't work yet)
    # img = env.render(offscreen=True)
    # img = env.sim.render(mode="offscreen", width=640, height=480)
    print(type(img))

print(obs, reward, done, info)
print("all good so far")
# print(print([m for m in dir(type(env)) if not m.startswith("__")]))
