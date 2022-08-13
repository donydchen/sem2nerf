# Sem2NeRF

Official PyTorch implementation of [ECCV 2022] *Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields* by [Yuedong Chen](https://donydchen.github.io/), [Qianyi Wu](https://qianyiwu.github.io/), [Chuanxia Zheng](https://www.chuanxiaz.com/), [Tat-Jen Cham](https://personal.ntu.edu.sg/astjcham/) and [Jianfei Cai](https://jianfei-cai.github.io/).

<a href="https://arxiv.org/abs/2203.10821"><img src="https://img.shields.io/badge/arXiv-2203.10821-b31b1b.svg" height=22.5></a> 
<a href="https://www.youtube.com/watch?v=cYr3Dz8N_9E"><img src="https://img.shields.io/badge/YouTube-Demo Video-blue.svg" height=22.5></a>
<a href="https://donydchen.github.io/sem2nerf/"><img src="https://img.shields.io/badge/Web-Project Page-brightgreen.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a> 


<img src="docs/sem2nerf.gif">


## Abstract
Image translation and manipulation have gain increasing attention along with the rapid development of deep generative models. Although existing approaches have brought impressive results, they mainly operated in 2D space. In light of recent advances in NeRF-based 3D-aware generative models, we introduce a new task, **Semantic-to-NeRF translation**, that aims to reconstruct a 3D scene modelled by NeRF, conditioned on one single-view semantic mask as input. To kick-off this novel task, we propose the **Sem2NeRF** framework. In particular, Sem2NeRF addresses the highly challenging task by encoding the semantic mask into the latent code that controls the 3D scene representation of a pretrained decoder. To further improve the accuracy of the mapping, we integrate a new region-aware learning strategy into the design of both the encoder and the decoder. We verify the efficacy of the proposed Sem2NeRF and demonstrate that it outperforms several strong baselines on two benchmark
datasets.



<!-- ## Recent Updates

* `14-Jul-2022`: released CatMask dataset with related training scripts.
* `11-Jul-2022`: released Sem2NeRF codes and models for CelebAMask-HQ and CatMask datasets.
* `22-Mar-2022`: initialize the Sem2NeRF repository with demo and arxiv manuscript. -->

<details>
  <summary>Recent Updates</summary>
        
* `14-Jul-2022`: released CatMask dataset with related training scripts.
        
* `11-Jul-2022`: released Sem2NeRF codes and models for CelebAMask-HQ and CatMask datasets.
        
* `22-Mar-2022`: initialize the Sem2NeRF repository with demo and arxiv manuscript.
</details>

## Getting Started

### Installation

We recommend to use [Anaconda](https://www.anaconda.com) to create the running environment for the project, and all related dependencies are provided in `environment/sem2nerf.yml`, kindly run

```bash
git clone https://github.com/donydchen/sem2nerf.git
cd sem2nerf
conda env create -f environment/sem2nerf.yml
conda activate sem2nerf
```

**Note**: The above environment contains *PyTorch 1.7 with CUDA 11*, if it does not work on your machine, please refer to [environment/README.md](https://github.com/donydchen/sem2nerf/blob/main/environment/README.md) for manual installation and trouble shootings.

### Download Pretrained Weights

Download the pretrained models from [here](https://drive.google.com/drive/folders/15oqMkBunN7jU3qgFTCWbSLZW2rvjheYR), and save them to `pretrained_models/`. Details of files are provided in [pretrained_models/README.md](https://github.com/donydchen/sem2nerf/blob/main/pretrained_models/README.md).

### Quick Test

We have provided some input semantic masks for a quick test, kindly run

```bash
python scripts/inference3d.py --use_merged_labels --infer_paths_conf=data/CelebAMask-HQ/val_paths.txt 
```

If the environment is setup correctly, this command should function properly and generate some results in the folder `out/sem2nerf_qtest`. For more details regarding datasets, training and more tunning options for inference, kindly walk through the following sections.

------

## Datasets

### CelebAMask-HQ

* Download the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset, and extract it to `data/CelebAMask-HQ/`. The folder should have the following structures

```bash
data/CelebAMask-HQ/
        |__ CelebA-HQ-img/
        |__ CelebAMask-HQ-mask-anno/
        |__ CelebAMask-HQ-pose-anno.txt
        |__ mask_samples/
        |__ test_paths.txt
        |__ train_paths.txt
        |__ val_paths.txt
```

* Preprocess the semantic mask data by running

```bash
python scripts/build_celeba_mask.py
```

This script will save the combined mask labels to `data/CelebAMask-HQ/masks` for training the networks.

### CatMask

* Download the pseudo [CatMask](https://drive.google.com/drive/folders/1hpQEMxb-VIz-lvI51ErLvcgujWghLcwr) dataset, and extract it to `data/CatMask/`. 
* For more details, please refer to [data/CatMask/README.md](https://github.com/donydchen/sem2nerf/blob/main/data/CatMask/README.md).


## Inference

### CelebAMask-HQ

Render high quality images and videos. 

```bash
python scripts/inference3d.py \
--exp_dir=out/sem2nerf_celebahq_test \
--checkpoint_path=pretrained_models/sem2nerf_celebahq_pretrained.pt \
--data_path=data/CelebAMask-HQ/mask_samples \
--test_output_size=512 \
--pigan_infer_ray_step=72 \
--use_merged_labels \
--use_original_pose \
--latent_mask=8 \
--inject_code_seed=92 \
--render_videos
```

Use `--render_videos` to render videos with predefined camera trajetory. Change `inject_code_seed` and `latent_mask` to generate multi-modal results, e.g., `--latent_mask=6,7,8 --inject_code_seed=711`. More options and descriptions can be found by running `python scripts/inference3d.py --help`

### CatMask

```bash
python scripts/inference3d.py \
--exp_dir=out/sem2nerf_catmask_test \
--dataset_type=catmask_seg_to_3dface \
--pigan_curriculum_type=CatMask \
--checkpoint_path=pretrained_models/sem2nerf_catmask_pretrained.pt \
--data_path=data/CatMask/mask_samples \
--test_output_size=512 \
--pigan_infer_ray_step=72 \
--use_merged_labels \
--use_original_pose \
--latent_mask=7,8 \
--inject_code_seed=390234 \
--render_videos
```

## Training

### CelebAMask-HQ

We use 8x32G V100 GPUs to train and fine-tune the whole framework for better visual quality. Run the following comand to run the training,

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
scripts/train3d.py \
--exp_dir=out/sem2nerf_celebahq \
--workers=2 \
--batch_size=2 \
--test_output_size=128 \
--train_paths_conf=data/CelebAMask-HQ/train_paths.txt \
--test_paths_conf=data/CelebAMask-HQ/val_paths.txt \
--pigan_steps_conf=configs/pigan_steps/sem2nerf.yaml \
--val_latent_mask=8 \
--train_rand_pose_prob=0.2 \
--use_contour \
--use_merged_labels \
--patch_train \
--start_from_latent_avg
```

If you only have limited GPU resources, e.g., 1 GPU, and still decide to try the training process, you are recommended to set `--nproc_per_node=1 --batch_size=1 --dis_lambda=0.`. If it still does not work, you may consider tuning down the decoder patch size by setting `resolution_vol: 64` in `configs/pigan_steps/sem2nerf.yaml`. Note that this may harm the performance. 

Our framework also supports running without the `torch.distributed.launch` module for easily debugging, kindly start the program as something like `python scripts/train3d.py --exp_dir=out/sem2nerf_celebahq ...`. Besider, it also supports training with *multiple nodes multiple GPUs*, dive into `options/train_options.py` or drop us a message if you need further instructions in this regards.

### CatMask

Configurations are in general similar to CelebAMask-HQ, but it mainly needs to change some options accordingly, e.g., `dataset_type, pigan_curriculum_type, train_paths_conf, test_paths_conf, label_nc, input_nc`. We provide a example as below,

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
scripts/train3d.py \
--exp_dir=out/sem2nerf_catmask \
--dataset_type=catmask_seg_to_3dface \
--pigan_curriculum_type=CatMask \
--train_paths_conf=data/CatMask/train_paths.txt \
--test_paths_conf=data/CatMask/val_paths.txt \
--label_nc=8 \
--input_nc=10 \
--workers=2 \
--batch_size=2 \
--dis_lambda=0.1 \
--w_norm_lambda=0.008 \
--val_latent_mask=8 \
--train_rand_pose_prob=0.5 \
--use_contour \
--use_merged_labels \
--patch_train \
--ray_min_scale=0.08 \
--start_from_latent_avg
```


## Misc

### Citations

If you use this project for your research, please cite our paper.

```bibtex
@article{chen2022sem2nerf,
    title={Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields},
    author={Chen, Yuedong and Wu, Qianyi and Zheng, Chuanxia and Cham, Tat-Jen and Cai, Jianfei},
    journal={arXiv preprint arXiv:2203.10821},
    year={2022}
}
```

### Pull Request

You are more than welcome to contribute to this project by sending a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests). 

### Related Work

If you are interested in **NeRF / neural implicit representions + semantic map**, we would also like to recommend you to check out other related works:

* Object-compositional implicit neural surfaces: [ECCV 2022] [ObjectSDF](https://qianyiwu.github.io/objectsdf).

* Digital human animation: [ECCV 2022 oral] [SSPNeRF](https://alvinliu0.github.io/projects/SSP-NeRF).



### Acknowledgments

Our implementation was mainly inspired by [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel), we also borrowed many codes from [pi-GAN](https://github.com/marcoamonteiro/pi-GAN), [GRAF](https://github.com/autonomousvision/graf), [GIRAFFE](https://github.com/autonomousvision/giraffe) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). Many thanks for all the above mentioned projects.



