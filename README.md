# Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields

*Official pytorch implementation, code is coming soon, stay tuned*

<img src="docs/input.png" width="1024">
<img src="docs/output.gif" width="1024" style="margin-top: -6px">


[[arXiv Manuscript](https://arxiv.org/#)] 
[[Project Page](https://donydchen.github.io/sem2nerf/)] 
[[Demo Video](https://www.youtube.com/watch?v=cYr3Dz8N_9E)]



## Abstract
Image translation and manipulation have gain increasing attention along with the rapid development of deep generative models. Although existing approaches have brought impressive results, they mainly operated in 2D space. In light of recent advances in NeRF-based 3D-aware generative models, we introduce a new task, **Semantic-to-NeRF translation**, that aims to reconstruct a 3D scene modelled by NeRF, conditioned on one single-view semantic mask as input. To kick-off this novel task, we propose the **Sem2NeRF** framework. In particular, Sem2NeRF addresses the highly challenging task by encoding the semantic mask into the latent code that controls the 3D scene representation of a pretrained decoder. To further improve the accuracy of the mapping, we integrate a new region-aware learning strategy into the design of both the encoder and the decoder. We verify the efficacy of the proposed Sem2NeRF and demonstrate that it outperforms several strong baselines on two benchmark
datasets.

## Citation

```bibtex
@article{chen2022sem2nerf,
    title={Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields},
    author={Chen, Yuedong and Wu, Qianyi and Zheng, Chuanxia and Cham, Tat-Jen and Cai, Jianfei},
    journal={arXiv},
    year={2022}
}
```


