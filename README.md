# CS559 Project

This is the implementation of our CS559 Project "Learning Deep State Representations for Reinforcement Learning". It is based on and built upon the codebase of [SNeRL](https://github.com/jayLEE0301/snerl_official/).

## Setup Instructions
0. Create a conda environment:
```
conda create -n envname python=3.9
conda activate envname
```

1. Install [MuJoCo](https://github.com/deepmind/mujoco) and task environments:
```
cd metaworld
pip install -e .
cd ..
```

2. install [pytorch](https://pytorch.org/get-started/locally/) (use tested on pytorch 2.1.1 with CUDA 11.8)



3. install additional dependencies:
```
pip install scikit-image
pip install tensorboard
pip install termcolor
pip install imageio
pip install imageio-ffmpeg
pip install opencv-python
pip install matplotlib
pip isntall tqdm
pip install timm
pip install configargparse
```





## Usage
First, obtain the data using the code in "datagen" folder.

### Pretrain Encoder
```
cd nerf_pretrain
python run_nerf.py --config configs/{env_name}.txt
```

### Train Donstream RL
0. Locate pretained model in './encoder_pretrained/{env_name}/snerl.tar'


1. Use the following commands to train RL agents:


drawer-open-v2
```
CUDA_VISIBLE_DEVICES=0 python snerl/train.py --env_name drawer-open-v2 --encoder_type nerf --save_tb --frame_stack 2 --eval_freq 10000 --batch_size 128 --save_video --save_model --image_size 128 --camera_name cam_1_1 cam_7_4 cam_14_2 --multiview 3 --encoder_name 'snerl' --seed 1
```

soccer-v2
```
CUDA_VISIBLE_DEVICES=0 python snerl/train.py --env_name soccer-v2 --encoder_type nerf --save_tb --frame_stack 2 --eval_freq 10000 --batch_size 128 --save_video --save_model --image_size 128 --camera_name cam_1_1 cam_7_4 cam_14_2 --multiview 3 --encoder_name 'snerl' --seed 1
```


## Reference
Our code is built upon the codebase of [SNeRL](https://github.com/jayLEE0301/snerl_official/).
