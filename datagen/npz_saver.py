import numpy as np

images = np.load("/home/yigit/Metaworld/NERF_Dataset/images.npy")
focal = np.load("/home/yigit/Metaworld/NERF_Dataset/focal.npy")
poses = np.load("/home/yigit/Metaworld/NERF_Dataset/poses.npy")



np.savez("/home/yigit/Metaworld/NERF_Dataset/tiny_nerf_data", images=images, focal=focal, poses=poses)
