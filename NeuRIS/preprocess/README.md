# Data preparation

We provide a pipeline to prepare training data for NeuRIS from 1) ScanNet data and 2) private data, including a video or a set of uncalibrated images. For private data, openMVG and openMVS are required. Please follow the steps below to prepare the running environment.

1. Download the normal network [TiltedSN](https://github.com/MARSLab-UMN/TiltedImageSurfaceNormal) or [SNU](https://github.com/baegwangbin/surface_normal_uncertainty). For private data, we can download and use their officially provided pretrained models. For ScanNet data, retraining the normal network is required and our pretrained models can be downloaded from [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jiepeng_connect_hku_hk/EpVwkZqo_vtIiWCGm2w91AIBDhAHIftPVxHeYmONnlI2sg?e=MPceUx). 
2. (Optional for private data) Follow the build tutorial of [openMVG](https://github.com/openMVG/openMVG/blob/develop/BUILD.md) and [openMVS](https://github.com/cdcseacave/openMVS/blob/master/BUILD.md) to build these two libraries using our provided packages in this [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jiepeng_connect_hku_hk/Ekh5W5iIv1tLnMswnyllFpQB0lV-2BK7Zu5qUb8RtyQmgQ?e=tAw8bn).
3. Update the paths in the file `confs/path.py`. 
4. For ScanNet data, please refer to the official [website](https://github.com/ScanNet/ScanNet) to download the used scenes and put all the scenes under the same folder. For private data, a video or a set of uncalibrated images should be put into the folder `<scene_name>/tmp_sfm_mvs` as shown in the data organization part.
5. Run the command below to generate files for NeuRIS.

```
python exp_preprocess.py --data_type scannet
```
6. (Optional) In our pipeline, we also provide some other options for users' interests, including depth priors from openMVS, plane priors w/o Manhattan-world assumption from normal priors using k-means and superpixels. More details can be found in `exp_preprocess.py`.


Data organization:

```
<scene_name>
|-- tmp_sfm_mvs         # optional, for private data or ScanNet data with unknown poses
    |-- video.MOV       # optional, a video of an indoor scene
    |-- images      # a set of uncalibrated images from a video or captured individually, or a set of ScanNet images
        |-- 0000.png        # target image for each view
        |-- 0001.png
        ...
|-- cameras_sphere.npz   # camera parameters
|-- image
    |-- 0000.png        # target image for each view
    |-- 0001.png
    ...
|-- depth
    |-- 0000.png        # target depth for each view, 0000.npy for private data
    |-- 0001.png
    ...
|-- pose
    |-- 0000.txt        # camera pose for each view
    |-- 0001.txt
    ...
|-- pred_normal
    |-- 0000.npz        # predicted normal for each view
    |-- 0001.npz
    ...
|-- neighbors.txt       # nearest neighbors of source view, optional for private data
|-- xxx.ply     # GT mesh or point cloud from MVS
|-- trans_n2w.txt		# transformation matrix from normalized coordinates to world coordinates
```