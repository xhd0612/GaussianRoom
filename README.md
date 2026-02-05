<div align="center">

<h1 align="center">GaussianRoom: Improving 3D Gaussian Splatting with SDF Guidance and Monocular Cues for Indoor Scene Reconstruction</h1>
<h2 align="center">ICRA 2025</h2>

[Haodong Xiang*](https://github.com/xhd0612), [Xinghui Li*](xxx), [Kai Cheng*](https://cklibra.github.io/), [Xiansong Lai](xxx), [Wanting Zhang](xxx), [Zhichao Liao](https://lzc-sg.github.io/), [Long Zeng](xxx)✉, [Xueping Liu](https://www.sigs.tsinghua.edu.cn/lxp/main.htm)✉

### [[`Project Page`](https://xhd0612.github.io/GaussianRoom.github.io/)][[`arxiv`](https://arxiv.org/abs/2405.19671)][[`Paper`](https://arxiv.org/pdf/2405.19671)]

</div>

## 📃 Abstract

Recently, 3D Gaussian Splatting(3DGS) has revolutionized neural rendering with its high-quality rendering and real-time speed. However,  when it comes to indoor scenes with a significant number of textureless areas, 3DGS yields incomplete and noisy reconstruction results due to the poor initialization of the point cloud and under-constrained optimization. Inspired by the continuity of signed distance field (SDF), which naturally has advantages in modeling surfaces, we present a unified optimizing framework integrating neural SDF with 3DGS.This framework incorporates a learnable neural SDF field to guide the densification and pruning of Gaussians, enabling Gaussians to accurately model scenes even with poor initialized point clouds. At the same time, the geometry represented by Gaussians improves the efficiency of the SDF field by piloting its point sampling. Additionally, we regularize the optimization with normal and edge priors to eliminate geometry ambiguity in textureless areas and improve the details. Extensive experiments in ScanNet and ScanNet++ show that our method achieves state-of-the-art performance in both surface reconstruction and novel view synthesis.

## 🧭 Overview

<p align="center">
<img src="./figure/overview.png" width=100% height=100% 
class="center">
</p>

GaussianRoom integrates neural SDF within 3DGS and forms a positive cycle improving each other.
    (a) We employ the geometric information from the SDF to constrain the Gaussian primitives, ensuring their spatial distribution aligns with the scene surface.
    (b) We utilize rasterized depth from Gaussian to efficiently provide coarse geometry information, narrowing down the sampling range to accelerate the optimization of neural SDF.
    (c) We introduce monocular normal prior and edge prior, addressing the challenges of texture-less areas and fine structures indoors.

## 🌏 News

- [ ✔️ ] 20241209 Code version0  release
- [ ✔️ ] 20241209 Dataset sample

## 📌 Setup

### Clone this repo

```
git clone https://github.com/xhd0612/GaussianRoom.git
```

### Environment setup

It is recommended to manually install some modules here, especially the submodules required by 3DGS.

```
conda create -n GaussianRoom python=3.9
conda activate GaussianRoom 
pip install -r requirements.txt
```

## 📁 Data Preparation

### Data structure

```shell
data/
├── intrinsic_depth.txt
├── scene_name-GS/
│   ├── images/
│   └── sparse/
├── scene_name-SDF/
│   ├── image/
│   ├── pred_edge/
│   ├── pred_normal/
│   ├── cameras_sphere.npz
│   ├── intrinsic_depth.txt
│   ├── scene1_vh_clean_2.ply
│   └── trans_n2w.txt
...
```

### Data Preprocessing

The raw indoor scene dataset we used is sourced from [ScanNet](https://github.com/ScanNet/ScanNet) and [ScanNet++](https://github.com/scannetpp/scannetpp). We also provide a processed dataset as a [sample](https://drive.google.com/drive/folders/1eD2cgCKEHn6tnpQUesq5RDmczeVaGJao?usp=drive_link) here.

Follow the data processing in [NeuRIS](https://github.com/jiepengwang/NeuRIS) for the SDF branch, and follow the data processing in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) for the GS branch.

### edge & normal

Please refer to [pidinet](https://github.com/hellozhuo/pidinet) for obtaining edge maps and [surface_normal_uncertainty](https://github.com/baegwangbin/surface_normal_uncertainty) for obtaining normal maps. Alternatively, you may choose other similar methods to obtain these two monocular priors required for GaussianRoom.

We save the edge maps in `.png` format to `data/scene_name-SDF/pred_edge `, and the normal maps in `.npz` format to `data/scene_name-SDF/pred_normal`.

## 🎮 Run the codes

The required GPU memory for running the code might be quite large, so it's recommended to run it on a GPU with more than 32GB of VRAM.

Before running `run.sh`, you need to modify the file paths in it to match your local paths.

```
cd GaussianRoom
bash run.sh
```

## 📜 Acknowledgement

Thanks to excellent open-source projects like [3dgs](https://github.com/graphdeco-inria/gaussian-splatting), [neus](https://github.com/Totoro97/NeuS), [neuris](https://github.com/jiepengwang/NeuRIS), and [gaussianpro](https://github.com/kcheng1021/GaussianPro), the open-sourcing of this work is a small contribution back to the open-source community.

## 🖊 Citation

If our work is helpful to your research, please consider citing:

```
@inproceedings{xiang2025gaussianroom,
  title={Gaussianroom: Improving 3d gaussian splatting with sdf guidance and monocular cues for indoor scene reconstruction},
  author={Xiang, Haodong and Li, Xinghui and Cheng, Kai and Lai, Xiansong and Zhang, Wanting and Liao, Zhichao and Zeng, Long and Liu, Xueping},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={2686--2693},
  year={2025},
  organization={IEEE}
}
```
