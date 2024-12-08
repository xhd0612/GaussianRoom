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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
import torch.nn.functional as F

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           return_depth = False, return_normal = False, return_opacity = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_dict =  {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


    '''
    viewpoint_camera.world_view_transform
    tensor([[ 0.6018,  0.4079, -0.6866,  0.0000],
            [ 0.7947, -0.2199,  0.5658,  0.0000],
            [ 0.0798, -0.8861, -0.4565,  0.0000],
            [-0.2794, -0.1310,  0.3024,  1.0000]])
    pose
    6.017839999999999856e-01 4.079389999999999961e-01 -6.866160000000000041e-01 4.291945841007517304e-01
    7.946579999999999755e-01 -2.198859999999999981e-01 5.658349999999999769e-01 2.213524814575151822e-02
    7.984900000000000331e-02 -8.861350000000000060e-01 -4.564960000000000129e-01 4.428542983722438819e-02
    0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00
    projvect1 包含了相机在世界坐标系中的方向信息，通常表示相机的视线方向。因为从变换矩阵中提取出来的是第三列，所以它代表了相机在世界坐标系中的 Z 轴方向（或者说是相机的前向方向）。因此，projvect1 可以用来表示相机拍摄图像时的视线方向。
    projvect2 是相机在世界坐标系中的位置信息，通常表示相机的位置偏移。由于从变换矩阵中提取出来的是第三列的最后一个元素，所以它代表了相机在世界坐标系中 Z 轴方向上的偏移量。因此，projvect2 可以用来表示相机拍摄图像时的位置偏移。
    '''
    if return_depth:
        projvect1 = viewpoint_camera.world_view_transform[:,2][:3].detach()                         # projvect1 torch.Size([3]) viewpoint_camera.world_view_transform torch.Size([4, 4])
        projvect2 = viewpoint_camera.world_view_transform[:,2][-1].detach()                         # tensor(0.3024)
        means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1,keepdim=True) + projvect2
        means3D_depth = means3D_depth.repeat(1,3) # torch.Size([924950, 3])
        render_depth, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = means3D_depth,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_depth = render_depth.mean(dim=0) 
        return_dict.update({'render_depth': render_depth}) # torch.Size([480, 640])
    
    if return_normal:
        rotations_mat = build_rotation(rotations)
        scales = pc.get_scaling
        min_scales = torch.argmin(scales, dim=1)
        indices = torch.arange(min_scales.shape[0])
        normal = rotations_mat[indices, :, min_scales]

        # convert normal direction to the camera; calculate the normal in the camera coordinate
        view_dir = means3D - viewpoint_camera.camera_center
        normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]

        R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32)
        normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)
  
        render_normal, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = normal,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_normal = F.normalize(render_normal, dim = 0)                                                                                                                                                   
        return_dict.update({'render_normal': render_normal})

    if return_opacity:
        density = torch.ones_like(means3D)
  
        render_opacity, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = density,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        return_dict.update({'render_opacity': render_opacity.mean(dim=0)})

    return return_dict

def render_gs_depth(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           return_depth = True):
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    '''
    projvect1 包含了相机在世界坐标系中的方向信息，通常表示相机的视线方向。因为从变换矩阵中提取出来的是第三列，所以它代表了相机在世界坐标系中的 Z 轴方向（或者说是相机的前向方向）。因此，projvect1 可以用来表示相机拍摄图像时的视线方向。
    projvect2 是相机在世界坐标系中的位置信息，通常表示相机的位置偏移。由于从变换矩阵中提取出来的是第三列的最后一个元素，所以它代表了相机在世界坐标系中 Z 轴方向上的偏移量。因此，projvect2 可以用来表示相机拍摄图像时的位置偏移。
    '''
    # if viewpoint_camera.image_name == '1032':
    #     print("!")
    if return_depth:
        projvect1 = viewpoint_camera.world_view_transform[:,2][:3].detach()                         # projvect1 torch.Size([3]) viewpoint_camera.world_view_transform torch.Size([4, 4])
        projvect2 = viewpoint_camera.world_view_transform[:,2][-1].detach()                         # tensor(0.3024)
        means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1,keepdim=True) + projvect2
        means3D_depth = means3D_depth.repeat(1,3) # torch.Size([924950, 3])
        render_depth, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = means3D_depth,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_depth = render_depth.mean(dim=0) 

    # 计算归一化pose
    # pose_R = viewpoint_camera.world_view_transform[:3, :3] # c2w == viewpoint_camera.R    torch.Size([3, 3])  
    # pose_t = viewpoint_camera.camera_center # c2w_t != viewpoint_camera.T (w2c)           torch.Size([3])
    # pose_c2w = torch.eye(4)
    # pose_c2w[:3, :3] = pose_R
    # pose_c2w[:3, 3] = pose_t
    # # pose_norm = 
        # return_dict.update({'render_depth': render_depth}) # torch.Size([480, 640])

    return render_depth
    # get_near_far处发出求解某图像深度请求
    # viewpoint_camera负责对应 有c2w(Rt) 和内参 K 可以将深度信息映射到世界坐标系
    # 计算射线与世界坐标系下深度表面的交点 pts = ray_o + D·ray_v与距离D o-------->|
    # s = sdf(pts) k|D-s|作为采样范围
    # near = ray_o + (D - k|s|)
    # far = ray_o + (D + k|s|)

'''
3DGS_SDF/submodules/NeuRIS/exp_runner.py
class Runner
get_near_far()

3DGS_SDF/gaussian_renderer/__init__.py
render_gs_depth()

def get_near_far(self, rays_o, rays_d,  image_perm = None, iter_i= None, pixels_x = None, pixels_y= None):
    log_vox = {}
    log_vox.clear()
    batch_size = len(rays_o)
    if  self.dataset_type == 'dtu':
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
    elif self.dataset_type == 'indoor':
        near, far = torch.zeros(batch_size, 1), self.sample_range_indoor * torch.ones(batch_size, 1)
        #------------------------------------------应该就在此处改采样远近-----------------------------------
        # get_pose ray_o ray_v D = depth s = sdf(o + Dv)
        # rays_o, rays_d = self.gen_rays()
        # 实际上形参已经有rays_o, rays_d
        # img_idx 计算gs_depth然后转换到世界坐标 射线查询深度 射线查询sdf表面所在位置 根据差值计算判据 规定采样near far 

        #------------------------------------------应该就在此处改采样远近-----------------------------------
    else:
        NotImplementedError


def gen_rays_at(self, img_idx, pose = None, resolution_level=1):
    pose_cur = self.get_pose(img_idx, pose) # 和dataset pose不一样 可能是世界pose

    l = resolution_level
    tx = torch.linspace(0, self.W - 1, self.W // l)
    ty = torch.linspace(0, self.H - 1, self.H // l)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
    # 根据相机的内参矩阵，将像素坐标转换为相机坐标系下的坐标。这一步可以将图像中的像素坐标转换为相机坐标系下的射线方向。
    p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3 
    
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    rays_v = torch.matmul(pose_cur[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = pose_cur[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3 # torch.Size([320, 240, 3])
    return rays_o.transpose(0, 1), rays_v.transpose(0, 1)'''

'''
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

'''
