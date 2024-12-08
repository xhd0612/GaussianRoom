import os, argparse, logging
from datetime import datetime
import numpy as np

import preprocess.neuris_data  as neuris_data
import evaluation.EvalScanNet as EvalScanNet
from evaluation.renderer import render_depthmaps_pyrender

import utils.utils_geometry as GeoUtils
import utils.utils_image  as ImageUtils
import utils.utils_io as IOUtils
import utils.utils_normal as NormalUtils

from confs.path import lis_name_scenes

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval_3D_mesh_metrics')

    args = parser.parse_args()
    
    if args.mode == 'eval_3D_mesh_metrics':
        dir_dataset = './dataset/indoor'
        path_intrin = f'{dir_dataset}/intrinsic_depth.txt'
        name_baseline = 'neus' # manhattansdf neuris
        exp_name = name_baseline
        eval_threshold = 0.05
        check_existence = True
        
        dir_results_baseline = f'./exps/evaluation'

        metrics_eval_all = []
        for scene_name in lis_name_scenes:
            logging.info(f'\n\nProcess: {scene_name}')

            path_mesh_pred = f'{dir_results_baseline}/{name_baseline}/{scene_name}.ply'
            metrics_eval =  EvalScanNet.evaluate_3D_mesh(path_mesh_pred, scene_name, dir_dataset = './dataset/indoor',
                                                                eval_threshold = 0.05, reso_level = 2, 
                                                                check_existence = check_existence)
    
            metrics_eval_all.append(metrics_eval)
        metrics_eval_all = np.array(metrics_eval_all)
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log = f'{dir_results_baseline}/eval_{name_baseline}_thres{eval_threshold}_{str_date}.txt'
        EvalScanNet.save_evaluation_results_to_latex(path_log, 
                                                        header = f'{exp_name}\n                     Accu.      Comp.      Prec.     Recall     F-score \n', 
                                                        results = metrics_eval_all, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'w')
     
    if args.mode == 'eval_mesh_2D_metrices':
        dir_dataset = 'dataset/indoor'
        path_intrin = f'{dir_dataset}/intrinsic_depth.txt'
        
        name_baseline = 'neus'
        eval_type_baseline = 'mesh'
        scale_depth = False

        dir_results_baseline = f'./exps/evaluation/results_baselines/{name_baseline}'
        results_all =  []
        for scene_name in lis_name_scenes:
            # scene_name += '_corner'
            print(f'Processing {scene_name}...')
            dir_scan = f'{dir_dataset}/{scene_name}'
            if eval_type_baseline == 'mesh':
                # use rendered depth map
                path_mesh_baseline =  f'{dir_results_baseline}/{scene_name}.ply'
                pred_depths = render_depthmaps_pyrender(path_mesh_baseline, path_intrin, 
                                                            dir_poses=f'{dir_scan}/pose')
                img_names = IOUtils.get_files_stem(f'{dir_scan}/depth', '.png')
            elif eval_type_baseline == 'depth':
                dir_depth_baseline =  f'{dir_results_baseline}/{scene_name}'
                pred_depths = GeoUtils.read_depth_maps_np(dir_depth_baseline)
                img_names = IOUtils.get_files_stem(dir_depth_baseline, '.npy')
                
            # evaluation
            dir_gt_depth = f'{dir_scan}/depth'
            gt_depths, _ = EvalScanNet.load_gt_depths(img_names, dir_gt_depth)
            err_gt_depth_scale = EvalScanNet.depth_evaluation(gt_depths, pred_depths, dir_results_baseline, scale_depth=scale_depth)
            results_all.append(err_gt_depth_scale)
            
        results_all = np.array(results_all)
        print('All results: ', results_all)

        count = 0
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log_all = f'./exps/evaluation/results_{name_baseline}-scale_{scale_depth}_{eval_type_baseline}_{str_date}.txt'
        EvalScanNet.save_evaluation_results_to_latex(path_log_all, header = f'{str_date}\n\n',  mode = 'a')
        
        precision = 3
        results_all = np.round(results_all, decimals=precision)
        EvalScanNet.save_evaluation_results_to_latex(path_log_all, 
                                                        header = f'{name_baseline}', 
                                                        results = results_all, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'a',
                                                        precision = precision)

    if args.mode == 'evaluate_normal':
        # compute normal errors
        exp_name = 'exp_neuris'
        name_normal_folder = 'normal_render'
        
        dir_root_dataset = './dataset/indoor'
        dir_root_normal_gt = '../TiltedImageSurfaceNormal/datasets/scannet-frames'

        err_neus_all, err_pred_all = [], []
        num_imgs_eval_all = 0
        
        for scene_name in lis_name_scenes:
            # scene_name = 'scene0085_00'
            print(f'Process: {scene_name}')
           
            dir_normal_neus = f'./exps/indoor/neus/{scene_name}/{exp_name}/{name_normal_folder}'
            
            dir_normal_pred = f'{dir_root_dataset}/{scene_name}/pred_normal' 
            dir_poses = f'{dir_root_dataset}/{scene_name}/pose' 
            dir_normal_gt = f'{dir_root_normal_gt}/{scene_name}'
            error_neus, error_pred, num_imgs_eval = NormalUtils.evauate_normal(dir_normal_neus, dir_normal_pred, dir_normal_gt, dir_poses)
            err_neus_all.append(error_neus)
            err_pred_all.append(error_pred)
            
            num_imgs_eval_all += num_imgs_eval

        error_neus_all = np.concatenate(err_neus_all).reshape(-1)
        err_pred_all = np.concatenate(err_pred_all).reshape(-1)
        metrics_neus = NormalUtils.compute_normal_errors_metrics(error_neus_all)
        metrics_pred = NormalUtils.compute_normal_errors_metrics(err_pred_all)
        NormalUtils.log_normal_errors(metrics_neus, first_line='metrics_neus fianl')
        NormalUtils.log_normal_errors(metrics_pred, first_line='metrics_pred final')
        print(f'Total evaluation images: {num_imgs_eval_all}')

    if args.mode == 'evaluate_nvs':
           # compute normal errors
        name_img_folder = 'image_render'
        sample_interval = 1

        exp_name_nerf = 'exp_nerf'
        exp_name_neuris = 'exp_neuris'
        exp_name_neus  = 'exp_neus'

        evals_nvs_all ={
            'nerf': exp_name_nerf,
            'neus': exp_name_neus,
            'neuris': exp_name_neuris
        }
        psnr_all_methods = {}
        psnr_imgs = []
        psnr_imgs_stem = []
        np.set_printoptions(precision=3)
        for key in evals_nvs_all:
            exp_name = evals_nvs_all[key]
            model_type = 'nerf' if key == 'nerf' else 'neus'

            print(f"Start to eval: {key}. {exp_name}")
            err_neus_all, err_pred_all = [], []
            num_imgs_eval_all = 0
            
            psnr_scenes_all = []
            psnr_mean_all = []
            for scene_name in lis_name_scenes:
                # scene_name = 'scene0085_00'
                scene_name = scene_name + '_nvs'
                print(f'Process: {scene_name}')
                
                dir_img_gt = f'./dataset/indoor/{scene_name}/image'
                dir_img_neus = f'./exps/indoor/{model_type}/{scene_name}/{exp_name}/{name_img_folder}'
                psnr_scene, vec_stem_eval = ImageUtils.eval_imgs_psnr(dir_img_neus, dir_img_gt, sample_interval)
                print(f'PSNR: {scene_name} {psnr_scene.mean()}  {psnr_scene.shape}')
                psnr_scenes_all.append(psnr_scene)
                psnr_imgs.append(psnr_scene)
                psnr_imgs_stem.append(vec_stem_eval)
                psnr_mean_all.append(psnr_scene.mean())
                # input("anything to continue")
            
            psnr_scenes_all = np.concatenate(psnr_scenes_all)
            psnr_mean_all = np.array(psnr_mean_all)
            print(f'\n\n Mean of scnes: {psnr_mean_all.mean()}. PSNR of all scenes: {psnr_mean_all} \n mean of images:{psnr_scenes_all.mean()} image numbers: {len(psnr_scenes_all)} ')

            psnr_all_methods[key] = (psnr_scenes_all.mean(), psnr_mean_all.mean(),  psnr_mean_all)
        
        # psnr_imgs = vec_stem_eval + psnr_imgs
        path_log_temp = f'./exps/indoor/evaluation/nvs/evaluation_temp_{lis_name_scenes[0]}.txt'
        flog_temp = open(path_log_temp, 'w')
        for i in range(len(vec_stem_eval)):
            try:
                flog_temp.write(f'{psnr_imgs_stem[0][i][9:13]}  {psnr_imgs_stem[1][i][9:13]}  {psnr_imgs_stem[2][i][9:13]}: {psnr_imgs[0][i]:.1f}:  {psnr_imgs[1][i]:.1f}  {psnr_imgs[2][i]:.1f}\n')
            except Exception:
                print(f'Error: skip {vec_stem_eval[i]}')
                continue
        flog_temp.close()
        input('Continue?')

        print(f'Finish NVS evaluation')
        # print eval information
        path_log = f'./exps/indoor/evaluation/nvs/evaluation.txt'
        flog_nvs = open(path_log, 'a')
        flog_nvs.write(f'sample interval: {sample_interval}. Scenes number: {len(lis_name_scenes)}. Scene names: {lis_name_scenes}\n\n')
        flog_nvs.write(f'Mean_img  mean_scenes  scenes-> \n')
        for key in psnr_all_methods:
            print(f'[{key}] Mean of all images: {psnr_all_methods[key][0]}. Mean of scenes: {psnr_all_methods[key][1]} \n Scenes: {psnr_all_methods[key][2]}\n')
            flog_nvs.write(f'{key:10s} {psnr_all_methods[key][0]:.03f} {psnr_all_methods[key][1]:.03f}. {psnr_all_methods[key][2]}.\n')
        flog_nvs.write(f'Images number: {len(psnr_scenes_all)}\n')
        flog_nvs.close()
    
    print('Done')
