from typing import Any, Dict, Sequence
import open3d as o3d
import numpy as np
import os, logging
from argparse import ArgumentParser

from datetime import datetime
import numpy as np

import evaluation.EvalScanNet as EvalScanNet
from evaluation.renderer import render_depthmaps_pyrender

import utils.utils_geometry as GeoUtils
import utils.utils_image  as ImageUtils
import utils.utils_io as IOUtils
import utils.utils_normal as NormalUtils

def transform(
    mesh: o3d.geometry.TriangleMesh, scale: float, offset: Sequence[float]
) -> o3d.geometry.TriangleMesh:
    v = np.asarray(mesh.vertices)
    v *= scale
    v += offset
    mesh.vertices = o3d.utility.Vector3dVector(v)
    return mesh

def transform_point_cloud(
    point_cloud: o3d.geometry.PointCloud, scale: float, offset: Sequence[float]
) -> o3d.geometry.PointCloud:
    # 将点云数据转换为numpy数组
    points = np.asarray(point_cloud.points)
    # 缩放点云
    points *= scale
    # 偏移点云
    points += offset
    # 更新点云数据
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def eval_log(message="0000000", file_path = "./gs_mesh_log/log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 如果文件不存在，创建文件
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w"):
            pass
    with open(file_path, "a") as f:
        f.write(f"{timestamp}\n {message}\n")

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)


    parser = ArgumentParser("poission reconstruction")
    parser.add_argument("--root_dir", default="/home/xhd/xhd/gs-output/")
    parser.add_argument("--neuris_dir", default="/home/xhd/xhd/0-dataset/neuris_data")
    args = parser.parse_args()

    root_dir = args.root_dir
    neuris_dir = args.neuris_dir
    scenes = os.listdir(root_dir)

    for scene in scenes:

        iteration_dirs = []
        # 遍历目录，获取所有iteration目录的名称
        for dir_name in os.listdir(os.path.join(root_dir, scene, "point_cloud")):
            if dir_name.startswith("iteration_") and os.path.isdir(os.path.join(root_dir, scene, "point_cloud", dir_name)):
                iteration_dirs.append(dir_name)
        # 提取iteration目录中的数字，并找到最大值
        max_iteration = max(int(iter_dir.split("_")[1]) for iter_dir in iteration_dirs)
        # 构建具有最大iteration的目录路径
        max_iteration_dir = os.path.join(root_dir, scene, "point_cloud", "iteration_" + str(max_iteration))
        print("具有最大迭代数的目录路径：", max_iteration_dir)

        # ply_path = os.path.join(root_dir, scene, "point_cloud", "iteration_15000", "point_cloud.ply") # 高斯点云
        ply_path = max_iteration_dir
        trans_path = os.path.join(neuris_dir, scene, "trans_n2w.txt")
        mesh_dir = os.path.join(root_dir, scene)
        os.makedirs(mesh_dir, exist_ok=True)
        path_mesh_gt = os.path.join(neuris_dir, scene, scene + "_vh_clean_2.ply") # /neuris_data/scene0050_00/scene0050_00_vh_clean_2.ply
        mesh_path = os.path.join(mesh_dir, scene + "_mesh_world.ply")
        pc_path = os.path.join(mesh_dir, scene + "_world_pc.ply")

        # Read a PLY file (replace 'your_point_cloud.ply' with your actual file path)
        point_cloud = o3d.io.read_point_cloud(ply_path + "/point_cloud.ply")

        radius = 0.1  # 搜索半径
        max_nn = 30  # 邻域内用于估算法线的最大点数
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 法线估计

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=11)

        # n2w
        trans_n2w = np.loadtxt(trans_path)
        scale = trans_n2w[0, 0]
        offset = trans_n2w[:3, 3]

        mesh = transform(mesh, scale, offset)
        pc = transform_point_cloud(point_cloud, scale, offset)

        # save mesh
        o3d.io.write_point_cloud(pc_path, pc)
        o3d.io.write_triangle_mesh(mesh_path, mesh)

        # clean mesh
        if path_mesh_gt:
            GeoUtils.clean_mesh_points_outside_bbox(mesh_path, mesh_path, path_mesh_gt, scale_bbox = 1.1, check_existence=False)

        # # Visualize the result
        # o3d.visualization.draw_geometries([point_cloud, mesh])

    # ---------------------------------eval 3D--------------------------------------------------------------------------------------
    lis_name_scenes = scenes

    dir_dataset = neuris_dir
    path_intrin = f'{dir_dataset}/intrinsic_depth.txt'
    eval_threshold = 0.05
    check_existence = True
    
    # dir_results_baseline = f'./exps/evaluation' # Read PLY failed: unable to open file: ./exps/evaluation/neus/scene0085_00.ply
    # /home/xhd/xhd/0-output/neuris_data_sdf/neus/scene0085_00/scene0085_00-20k/meshes/00020000_reso512_scene0085_00_world.ply
    dir_results_baseline = root_dir
    exp_name = "gs_mesh"
    metrics_eval_all = []
    for scene_name in lis_name_scenes:
        logging.info(f'\n\nProcess: {scene_name}')

        # path_mesh_pred = f'{dir_results_baseline}/{name_baseline}/{scene_name}.ply'
        path_mesh_pred = mesh_path
        metrics_eval =  EvalScanNet.evaluate_3D_mesh(path_mesh_pred, scene_name, dir_dataset = dir_dataset,
                                                            eval_threshold = 0.05, reso_level = 2, 
                                                            check_existence = check_existence)
        msg = ("scene_name: {} \n \
            path_mesh_pred: {} \n \
            dir_dataset: {}".format(scene_name, path_mesh_pred, dir_dataset))
        eval_log(msg)

        metrics_eval_all.append(metrics_eval)
    metrics_eval_all = np.array(metrics_eval_all)
    str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # path_log = f'{dir_results_baseline}/eval_{name_baseline}_thres{eval_threshold}_{str_date}.txt'
    path_log = f'{dir_results_baseline}/eval_thres{eval_threshold}_{str_date}.txt'
    EvalScanNet.save_evaluation_results_to_latex(path_log, 
                                                    header = f'{exp_name}\n                     Accu.      Comp.      Prec.     Recall     F-score \n', 
                                                    results = metrics_eval_all, 
                                                    names_item = lis_name_scenes, 
                                                    save_mean = True, 
                                                    mode = 'w',
                                                    eval_log = eval_log)
    eval_log("\n--------------------------------------------------\n")
