# 0417
python ./exp_runner.py --mode train --conf ./confs/neuris.conf --gpu 0 --scene_name scene0009_01
/data/xhd/2-3dRecon/1-3DGS/NeuRIS/dataset/indoor/scene0009_01

灵异事件：NeuRIS/confs/neuris.conf 文件中98行 125行注释换位才能运行
否则报错 pyparsing.exceptions.ParseException: Expected '}', found '='  (at char 2030), (line:98, col:14)

/data/xhd/2-3dRecon/1-3DGS/NeuRIS/models/dataset.py 126 127
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # n_images, H, W, 3   # Save GPU memory
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # n_images, H, W, 3   # Save GPU memory
导致后续数据分布在cpu和GPU而报错


# 0419

/data/xhd/2-3dRecon/1-3DGS/NeuRIS/dataset/indoor/scene0009_01

python ./exp_runner.py --mode train --conf ./confs/scene0085_00.conf --gpu 0 --scene_name scene0085_00


python exp_runner.py --mode validate_mesh --conf ./confs/scene0085_00.conf --is_continue

python ./exp_evaluation.py --mode eval_3D_mesh_metrics


-0.998134 -0.025826 -0.055325 2.355182
0.043890 0.326427 -0.944203 2.984659
0.042444 -0.944870 -0.324684 1.395898
0.000000 0.000000 0.000000 1.000000

load_K_Rt_from_P poseall 1
tensor([[-0.9981, -0.0258, -0.0553, -0.1321],
        [ 0.0439,  0.3264, -0.9442,  0.1816],
        [ 0.0424, -0.9449, -0.3247,  0.0723],
        [ 0.0000,  0.0000,  0.0000,  1.0000]], device='cpu')


-0.998736 -0.023407 -0.044478 2.326904
0.034248 0.330750 -0.943097 2.990230
0.036786 -0.943428 -0.329531 1.390010
0.000000 0.000000 0.000000 1.000000

tensor([[-0.9987, -0.0234, -0.0445, -0.1391],
        [ 0.0342,  0.3308, -0.9431,  0.1830],
        [ 0.0368, -0.9434, -0.3295,  0.0708],
        [ 0.0000,  0.0000,  0.0000,  1.0000]], device='cpu')