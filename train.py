#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_gs_depth
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.general_utils import vis_depth, read_propagted_depth
from utils.general_utils import safe_state, load_pairs_relation
from utils.graphics_utils import depth_propagation, check_geometric_consistency

import torchvision
import numpy as np
import imageio
# from submodules.surface_normal_uncertainty.models.NNET import NNET
# import submodules.surface_normal_uncertainty.utils.utils as sne_utils
# from submodules.surface_normal_uncertainty.data.dataloader_custom import CustomLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import logging
from datetime import datetime
from NeuRIS.exp_runner import Runner
import NeuRIS.utils.utils_io as IOUtils

from scene.dataset_readers import fetchPly, fetchPly_no_color

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # train_sdf()
    
    #=======================================================SDF Init=======================================================
    runner = Runner(args.conf, args.scene_name, args.mode, args.model_type, args.is_continue, args.checkpoint_id , args)
    runner.writer = SummaryWriter(log_dir=os.path.join(runner.base_exp_dir, 'logs'))
    runner.update_learning_rate()
    runner.update_iter_step()
    res_step = runner.end_iter - runner.iter_step

    if runner.dataset.cache_all_data:
        runner.dataset.shuffle() 

    # runner.validate_mesh() # save mesh at iter 0
    logs_summary = {}
    image_perm = torch.randperm(runner.dataset.n_images) 
    #======================================================================================================================

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)


    # sne_imgs_dir = os.path.join(args.source_path, 'images')
    # sne_results_dir = os.path.join(args.source_path, 'sne_results')
    # normal_gt_dict = normal_estimation(input_height = 480, input_width = 640, imgs_dir = sne_imgs_dir, results_dir = sne_results_dir, architecture = args.architecture) 
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        # gaussians.random_prune(threshold_pts=100_0000, prune_percent=0.3)
        # gaussians.random_prune(threshold_pts=60_0000, prune_percent=0.2)
    GS_START_FROM = first_iter
    SDF_START_FROM = runner.iter_step
    # GAUSSIAN_ORI = copy.deepcopy(gaussians)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) 
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # with torch.no_grad(): 你把梯度掐断了憨憨
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                                return_normal=True, return_opacity=True, return_depth=True)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        # -----------------------------------original loss-------------------------------------------------
        if not args.is_gs_edge:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            gs_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # -----------------------------------egde guided loss-------------------------------------------------
        else:
            gs_img_idx = viewpoint_cam.image_name
            try:
                idx_img = runner.dataset.vec_stem_files.index(gs_img_idx) # 135 0810
                # print("Index of gs_img_idx:", idx_img)
            except ValueError:
                print("gs_img_idx not found in vec_stem_files.")
            edge = runner.dataset.edges[idx_img]
            edge = edge.sum(dim=-1)
            edge_weight = 2 * torch.sigmoid(edge) 
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(edge_weight * image, edge_weight * gt_image)
            gs_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # -------------------------------------------------------------------------------------------------

        # gs normal loss
        if args.is_gs_norm:
            rendered_normal = render_pkg['render_normal']

            #---------------------------------------------------------------------------------------------------------pkl
            # if normal_gt_dict[str(viewpoint_cam.image_name)] is not None:
            #     normal_gt = -normal_gt_dict[str(viewpoint_cam.image_name)]['pred_norm'].cuda()
            #     if viewpoint_cam.sky_mask is not None:
            #         filter_mask = viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
            #         normal_gt[~(filter_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
            #     filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
            #     l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
            #     cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
            #     gs_loss += opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal # 惩罚模长 方向
            #---------------------------------------------------------------------------------------------------------npz
  
            # ------------------------------------- .npz normal -------------------------------------
            gs_img_idx = viewpoint_cam.image_name
            idx_img = runner.dataset.vec_stem_files.index(gs_img_idx)
            normal_gt = runner.dataset.normals[idx_img]
            normal_gt = - normal_gt.permute(2, 0, 1)

            if viewpoint_cam.sky_mask is not None:
                filter_mask = viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
                normal_gt[~(filter_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
            filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
            l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
            cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
            gs_loss += opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal # 惩罚模长 方向

            # show_normal("/home/xhd/xhd/normal_gt.png", normal_gt)
            # show_normal("/home/xhd/xhd/rendered_normal.png", rendered_normal)
            
            #---------------------------------------------------------------------------------------------------------

        gs_loss.backward()

        #=======================================================SDF backward=======================================================
        iter_i = iteration - GS_START_FROM
        
        logs_summary.clear()
 
        view_cams = scene.getTrainCameras().copy()
        gs_render_conf = scene, view_cams, gaussians, pipe, background, args.gs2sdf_from 
        # gs_render_conf = scene, view_cams, GAUSSIAN_ORI, pipe, background, args.gs2sdf_from 
        input_model, logs_input = runner.get_model_input(image_perm, iter_i, gs_render_conf) # input_model.pixels_x pixels_y pixels_uv pixels_vu torch.Size([512]) 这些pixels是啥意思
        logs_summary.update(logs_input)

        render_out, logs_render = runner.renderer.render(input_model['rays_o'], input_model['rays_d'],    # input_model['rays_o'] input_model['rays_d'] torch.Size([512, 3]) 512是batch_size?
                                        input_model['near'], input_model['far'],                        # input_model['near'], input_model['far'] torch.Size([512, 1]) 标量表示远近 near全是0 far全是2 范围是多少？
                                        background_rgb=input_model['background_rgb'],
                                        alpha_inter_ratio=runner.get_alpha_inter_ratio())
        logs_summary.update(logs_render)

        patchmatch_out, logs_patchmatch = runner.patch_match(input_model, render_out) # 是啥作用 计算Normalized Cross Correlation 用于衡量 visual consistency
        logs_summary.update(logs_patchmatch)

        sdf_loss, logs_loss, mask_keep_gt_normal = runner.loss_neus(input_model, render_out, runner.sdf_network_fine, patchmatch_out)
        logs_summary.update(logs_loss)

        sdf_loss.backward()
        #======================================================================================================================

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * gs_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                if runner.sample_info: 
                    sample_info = [f"{tensor:.2f}" for tensor in runner.sample_info]
                else:
                    sample_info = None

                progress_bar.set_postfix({  "Loss": f"{ema_loss_for_log:.{7}f}",
                                            "pts": f"{gaussians.get_xyz.shape[0]}",
                                            "exp_info": f"{args.exp_conf}",
                                            "cnt": f"{runner.cnt}",
                                            "k": f"{runner.k}",
                                            "sample_info": f"{sample_info}"
                                          })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, gs_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            
            #=======================================================SDF D&P=======================================================
            # self.sdf_network_fine.sdf(pts)[:, 0]
            
            if args.is_geometry_gui:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter]) # 只在可见视角内做DP
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # TODO 这里计算压力稍大可以适当提高interval
                if (iteration - GS_START_FROM + SDF_START_FROM) >= args.sdf2gs_from and (iteration - GS_START_FROM  + SDF_START_FROM) <= args.sdf2gs_end and (iteration - GS_START_FROM  + SDF_START_FROM) % args.geo_interval == 3: # iter > 500 iter % 100 
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # opt.opacity_reset_interval = 3000
                    gaussians.sdf_densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, lambda pts : runner.sdf_network_fine.sdf(pts)[:, 0])
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity() # 将所有opacity置为0.01 深度失效

                # sdf预热后 gs指导sdf之前 修剪15k的gs
                if (iteration - GS_START_FROM) < args.gs2sdf_from and (iteration - GS_START_FROM) >= args.gs2sdf_from - 200 and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # opt.opacity_reset_interval = 3000
                    gaussians.sdf_densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, lambda pts : runner.sdf_network_fine.sdf(pts)[:, 0])

            else:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (iteration - GS_START_FROM + SDF_START_FROM) >= args.sdf2gs_from and (iteration - GS_START_FROM  + SDF_START_FROM) <= args.sdf2gs_end and (iteration - GS_START_FROM  + SDF_START_FROM) % args.geo_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            # =======================================================anchor=======================================================
            if args.is_anchor:
                if (iteration - GS_START_FROM + SDF_START_FROM) >= args.sdf2gs_from \
                    and (iteration - GS_START_FROM  + SDF_START_FROM) <= args.sdf2gs_end - 2*args.anchor_interval \
                    and (iteration - GS_START_FROM + SDF_START_FROM)  % args.anchor_interval == 0 \
                    or (iteration - GS_START_FROM + SDF_START_FROM) == args.sdf2gs_from + 1 : 
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter]) # 只在可见视角内做DP
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    with torch.no_grad():
                        anchor_pts = runner.gen_anchor_pts(gaussians, world_space=True, resolution=128, threshold=args.threshold)
                        # runner.show_anchor_pts(gaussians.get_xyz, msg=f"gs_xyz_{runner.iter_step}")
                        new_xyz = gaussians.anchor2gs(anchor_pts, args.anchor_extend, k = args.anchor_knn)
                        # runner.show_anchor_pts(new_xyz, msg=f"new_xyz_{runner.iter_step}")
                        
                        # gaussians.random_prune(threshold_pts=50_0000, prune_percent=0.2)

                        new_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool)
                        new_mask[-new_xyz.shape[0]:] = True

                        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                                return_normal=True, return_opacity=True, return_depth=True)
                        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                        out_vis = ~visibility_filter
                        p_mask = new_mask & out_vis
                        gaussians.prune_points(p_mask)
            # =====================================================================================================================

            # 原版DP 容易导致高斯过多
            if (iteration - GS_START_FROM + SDF_START_FROM) > args.gs_post_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration - GS_START_FROM  > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            #=====================================================================================================================

            # Densification 15k以内做原版D&P
            # if iteration - GS_START_FROM < opt.densify_until_iter: # 15000
            if iteration < opt.densify_until_iter: # 15000
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration - GS_START_FROM  > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        #=======================================================SDF Post=======================================================
        runner.optimizer.step()
        runner.optimizer.zero_grad()

        runner.iter_step += 1

        logs_val = runner.validate(input_model, logs_loss, render_out)
        logs_summary.update(logs_val)
        
        logs_summary.update({'Log/lr': runner.optimizer.param_groups[0]['lr']})
        runner.write_summary(logs_summary)

        runner.update_learning_rate()
        runner.update_iter_step()
        runner.accumulate_rendered_results(input_model, render_out, patchmatch_out,
                                                b_accum_render_difference = False,
                                                b_accum_ncc = False,
                                                b_accum_normal_pts = False)

        if runner.iter_step % runner.dataset.n_images == 0:
            image_perm = torch.randperm(runner.dataset.n_images)

        # if runner.iter_step % 1000 == 0:
        #     torch.cuda.empty_cache()
        
        if runner.iter_step % 99 == 0:
            torch.cuda.empty_cache()

    logging.info(f'Done. [{runner.base_exp_dir}]')
        #======================================================================================================================


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def normal_estimation(input_height, input_width, imgs_dir, results_dir, architecture='BN'):
    checkpoint = './submodules/surface_normal_uncertainty/checkpoints/%s.pt' % "scannet"
    print('loading checkpoint... {}'.format(checkpoint))
    model = NNET(architecture).to(device="cuda")
    model = sne_utils.load_checkpoint(checkpoint, model)
    model.eval()
    print('loading checkpoint... / done')
    os.makedirs(results_dir, exist_ok=True)
    test_loader = CustomLoader(input_height, input_width, imgs_dir).data
    normal_gt_dict = sne_test(model,test_loader, "cuda", results_dir)
    return normal_gt_dict


def sne_test(model, test_loader, device, results_dir):
    alpha_max = 60
    kappa_max = 30
    
    # 检查是否存在已保存的normal_gt_dict文件
    normal_gt_dict_path = os.path.join(results_dir, 'normal_gt_dict.pkl')
    if os.path.exists(normal_gt_dict_path):
        with open(normal_gt_dict_path, 'rb') as f:
            normal_gt_dict = pickle.load(f)
        print(f"Loaded normal_gt_dict from {normal_gt_dict_path}")
        return normal_gt_dict
    
    # 计算normal_gt并保存
    normal_gt_dict = {}

    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            img_info = {}

            img = data_dict['img'].to(device)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]

            pred_norm = norm_out[:, :3, :, :]               # (B, 3, H, W)
            pred_kappa = norm_out[:, 3:, :, :]
            img_info.update({'pred_norm': pred_norm[0]})       # (3, H, W)
            # img_info.update({'pred_norm': pred_norm.cpu().numpy()})
            # img_info.update({'pred_kappa': pred_kappa.cpu().numpy()})

            # to numpy arrays
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()


            img_info.update({'pred_norm_ndarray': pred_norm})

            # save results
            img_name = data_dict['img_name'][0]
            normal_gt_dict.update({img_name: img_info})

    # 保存normal_gt_dict
    with open(normal_gt_dict_path, 'wb') as f:
        pickle.dump(normal_gt_dict, f)
    print(f"Saved normal_gt_dict to {normal_gt_dict_path}")

    return normal_gt_dict

def show_normal(img_path, tensor_CHW):
    tensor_HWC = tensor_CHW.detach().cpu().permute(1, 2, 0).numpy() 
    normal_rgb = ((tensor_HWC + 1) * 0.5) * 255
    normal_rgb = np.clip(normal_rgb, a_min=0, a_max=255)
    normal_rgb = normal_rgb.astype(np.uint8)

    plt.imsave(img_path, normal_rgb)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[80_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[80_000])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000 + 15000, 10_000 + 15000, 50_000 + 15000, 80_000+ 15000]) # 比较占空间
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[40_000 + 15000, 80_000 + 15000])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    #======================================SNE arg====================================
    parser.add_argument('--architecture', default='BN', type=str, help='{BN, GN}')
    #======================================SDF arg====================================
    # parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    # parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint_id', type=int, default=-1)
    parser.add_argument('--mc_reso', type=int, default=512, help='Marching cube resolution')
    parser.add_argument('--reset_var', action= 'store_true', help='Reset variance for validate_iamge()' )
    parser.add_argument('--nvs', action= 'store_true', help='Novel view synthesis' )
    parser.add_argument('--save_render_peak', action= 'store_true', help='Novel view synthesis' )
    parser.add_argument('--scene_name', type=str, default='', required=True, help='Scene or scan name')

    #======================================sdf conf config====================================
    parser.add_argument('--conf', type=str, default='./0-confs/template.conf', help='path to conf_template')
    parser.add_argument('--conf_dataset_type', type=str, default='indoor', help='dtu or indoor')
    # parser.add_argument('--scan_name', type=str, default='scene0085_00', required=True)    # --scene_name
    parser.add_argument('--exp_name', type=str, default='scene0085_00-test000', required=True)
    parser.add_argument('--data_dir', type=str, default='/home/xhd/xhd/0-dataset/neuris_data', required=True)
    parser.add_argument('--exp_dir', type=str, default='/home/xhd/xhd/0-output/neuris_data_sdf', required=True)

    parser.add_argument('--k', type=float, default=7, help='采样范围系数')
    parser.add_argument('--kk', type=float, default=0, help='采样范围衰减率')
    parser.add_argument('--min_sam_dis', type=float, default=0.2, help='最小采样距离')

    #======================================anchor config====================================
    parser.add_argument('--anchor_thres', type=float, default=0.01, help='sdf 值小于阈值就放置gs seed')
    parser.add_argument('--anchor_interval', type=int, default=999, help='生成anchor的iter')          # 和其他剪枝不要在同一iter 
    parser.add_argument('--anchor_knn', type=int, default=10, help='anchor 衍生 gs 数目') 
    parser.add_argument('--anchor_extend', type=float, default=0.0156, help='anchor 扩展的距离') 
    
    

    #======================================iter config====================================
    parser.add_argument('--sdf2gs_from', type=int, default=5_000, help='iter sdf starts guiding gs')
    parser.add_argument('--sdf2gs_end', type=int, default=60_000, help='iter sdf stops guiding gs')
    parser.add_argument('--gs2sdf_from', type=int, default=5_000, help='iter gs starts guiding sdf')  
    parser.add_argument('--gs2sdf_end', type=int, default=5_000, help='iter gs starts guiding sdf')   # 考虑启用   
    parser.add_argument('--gs_post_iter', type=int, default=100_000, help='最后做一些原始高斯dp')  
         
    parser.add_argument('--geo_interval', type=int, default=100, help='sdf geo gui interval')               # 0505
    parser.add_argument('--no_sam_iter', type=int, default=1000, help='3k清空opacityi之后 不指导iter数')
    parser.add_argument('--sam_add_len', type=float, default=1.0, help='采样异常两端加长射线值')


    #======================================Ablation Switch====================================
    parser.add_argument('--exp_conf',type=str, default='1234567', required=True)
    # 默认true 传参false
    parser.add_argument('--is_sdf_norm',     type=bool, default=True)  # 学习低纹理部分
    parser.add_argument('--is_gs_norm',      type=bool, default=True)
    parser.add_argument('--is_sdf_edge',     type=bool, default=True)  # 学习高频特征
    parser.add_argument('--is_gs_edge',      type=bool, default=True)
    parser.add_argument('--is_geometry_gui', type=bool, default=True)  # sdf指导高斯densify & prune
    parser.add_argument('--is_sample_gui',   type=bool, default=True)  # gs depth指导sdf采样
    parser.add_argument('--is_anchor',       type=bool, default=True)  # sdf生成锚点利于高斯在正常位置生长

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    exp = [int(x) for x in args.exp_conf]
    args.is_sdf_norm        = bool(exp[0])
    args.is_gs_norm         = bool(exp[1])
    args.is_sdf_edge        = bool(exp[2])
    args.is_gs_edge         = bool(exp[3])  
    args.is_geometry_gui    = bool(exp[4])
    args.is_sample_gui      = bool(exp[5])
    args.is_anchor          = bool(exp[6])

    args.model_path = os.path.join(args.model_path, args.exp_name)
    #======================================SDF====================================
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    #==============================================================================
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

