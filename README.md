<div align="center">

  <h1 align="center">GaussianRoom: Improving 3D Gaussian Splatting with SDF Guidance and Monocular Cues for Indoor Scene Reconstruction</h1>
<!--   <h2 align="center">ICML 2024</h2> -->
  


[Haodong Xiang*](xxx), [Xinghui Li*](xxx), [Xiansong Lai](xxx), [Wanting Zhan](https://jianglh-whu.github.io/), [Zhichao Liao](xxx), [Kai Cheng](https://cklibra.github.io/)✉, [Xueping Liu](https://www.sigs.tsinghua.edu.cn/lxp/main.htm)✉ <br />


### [[`Project Page`](https://xhd0612.github.io/GaussianRoom.github.io/)][[`arxiv`](https://arxiv.org/abs/xxxx)]

</div>

## Overview
<p align="center">
<img src="./figure/overview.png" width=100% height=100% 
class="center">
</p>

GaussianRoom integrates neural SDF within 3DGS and forms a positive cycle improving each other. 
    (a) We employ the geometric information from the SDF to constrain the Gaussian primitives, ensuring their spatial distribution aligns with the scene surface.
    (b) We utilize rasterized depth from Gaussian to efficiently provide coarse geometry information, narrowing down the sampling range to accelerate the optimization of neural SDF.
    (c) We introduce monocular normal prior and edge prior, addressing the challenges of texture-less areas and fine structures indoors.


### Our code is coming soon.
    
