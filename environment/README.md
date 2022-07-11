# Sem2NeRF Environment Installation


## Manual installation

The provided exported anoconda environment file is mainly tested on `NVIDIA GeForce RTX 3090` and `Tesla V100` with cuda 11.4. If it does not work on your machine, please consider installing the environment manually by following the instructions below. 

* initialize a python 3.6 environment

```bash
conda create -n sem2nerf python=3.6.7
conda activate sem2nerf
```

* install pytorch 1.7.0. Choose a correct CUDA version from [here](https://pytorch.org/get-started/previous-versions/#v170). Below we provide the commands for cuda11

```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```

* install related python dependencies

```bash
pip install matplotlib==3.2.1 pyyaml==5.3.1 opencv-python==4.2.0.34 timm==0.5.4 tensorboard==2.2.1 tqdm==4.46.0 scikit-video==1.1.11 
```

## Miscs

The `scripts/inference3d.py` leverages `ffmpeg` to generate video files. If you do not have ffmpeg installed on your machine, please install it via conda, e.g.,

```bash
conda install -c conda-forge ffmpeg
```

