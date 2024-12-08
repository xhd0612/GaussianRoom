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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", K=None, 
                 sky_mask=None, normal=None, depth=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.sky_mask = sky_mask
        self.normal = normal
        self.depth = depth

        # if image_name == '1356':
        #     print("!")

        '''
        R
        array([[ 0.60178366,  0.40793929, -0.68661632],
       [ 0.79465752, -0.2198862 ,  0.56583521],
       [ 0.07984896, -0.8861352 , -0.45649595]])

       T
       array([-0.27940821, -0.13097533,  0.30238335])

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
    
    viewpoint_camera.camera_center
    tensor([0.4292, 0.0221, 0.0443])
    
    tensor([[ 0.6018,  0.4079, -0.6866,  0.0000],
        [ 0.7947, -0.2199,  0.5658,  0.0000],
        [ 0.0798, -0.8861, -0.4565,  0.0000],
        [-0.2794, -0.1310,  0.3024,  1.0000]])
        '''
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.K = torch.tensor([[K[0], 0, K[2]],
                               [0, K[1], K[3]],
                               [0, 0, 1]]).to(self.data_device).to(torch.float32)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda() # 世界坐标下的平移
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# import torch
# from torch import nn
# import numpy as np
# from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# class Camera(nn.Module):
#     def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
#                  image_name, uid,
#                  trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
#                  ):
#         super(Camera, self).__init__()

#         self.uid = uid
#         self.colmap_id = colmap_id
#         self.R = R
#         self.T = T
#         self.FoVx = FoVx
#         self.FoVy = FoVy
#         self.image_name = image_name

#         try:
#             self.data_device = torch.device(data_device)
#         except Exception as e:
#             print(e)
#             print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
#             self.data_device = torch.device("cuda")

#         self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
#         self.image_width = self.original_image.shape[2]
#         self.image_height = self.original_image.shape[1]

#         if gt_alpha_mask is not None:
#             self.original_image *= gt_alpha_mask.to(self.data_device)
#         else:
#             self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

#         self.zfar = 100.0
#         self.znear = 0.01

#         self.trans = trans
#         self.scale = scale

#         self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
#         self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
#         self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
#         self.camera_center = self.world_view_transform.inverse()[3, :3]

# class MiniCam:
#     def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
#         self.image_width = width
#         self.image_height = height    
#         self.FoVy = fovy
#         self.FoVx = fovx
#         self.znear = znear
#         self.zfar = zfar
#         self.world_view_transform = world_view_transform
#         self.full_proj_transform = full_proj_transform
#         view_inv = torch.inverse(self.world_view_transform)
#         self.camera_center = view_inv[3][:3]

