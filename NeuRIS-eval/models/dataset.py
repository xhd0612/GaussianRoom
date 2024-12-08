import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import cv2 as cv
import numpy as np
import os, copy, logging

from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from utils.utils_image import read_images, write_image, write_images
from utils.utils_io import checkExistence, ensure_dir_existence, get_path_components, get_files_stem
from utils.utils_geometry import get_pose_inv, get_world_normal, quat_to_rot, save_points

import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils
import models.patch_match_cuda as PatchMatch
import utils.utils_training as TrainingUtils
import utils.utils_image as ImageUtils


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P) # 投影矩阵分解为相机内参矩阵、旋转矩阵、平移矩阵和相机坐标系到世界坐标系的变换矩阵
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    '''Check normal and depth in folder depth_cloud
    '''
    def __init__(self, conf):
        super(Dataset, self).__init__()
        # logging.info('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf['data_dir']
        self.cache_all_data = conf['cache_all_data']
        assert self.cache_all_data == False
        self.mask_out_image = conf['mask_out_image']
        self.estimate_scale_mat = conf['estimate_scale_mat']
        self.piece_size = 2**20
        self.bbox_size_half = conf['bbox_size_half']
        self.use_normal = conf['use_normal']
        self.resolution_level = conf['resolution_level']

        self.denoise_gray_image = self.conf['denoise_gray_image']
        self.denoise_paras = self.conf['denoise_paras']

        self.use_planes = conf['use_planes']
        self.use_plane_offset_loss = conf['use_plane_offset_loss']
 
        path_cam = os.path.join(self.data_dir, './cameras_sphere.npz')  # cameras_sphere, cameras_linear_init
        camera_dict = np.load(path_cam)
        logging.info(f'Load camera dict: {path_cam.split("/")[-1]}')
        
        images_lis = None
        for ext in ['.png', '.JPG']:
            images_lis = sorted(glob(os.path.join(self.data_dir, f'image/*{ext}')))
            self.vec_stem_files = get_files_stem(f'{self.data_dir}/image', ext_file=ext)
            if len(images_lis) > 0:
                break
        assert len(images_lis) > 0
        
        self.n_images = len(images_lis)
        logging.info(f"Read {self.n_images} images.")
        self.images_lis = images_lis
        self.images_np = np.stack([self.read_img(im_name, self.resolution_level) for im_name in images_lis]) / 256.0
        masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        if len(masks_lis) ==0:
            self.masks_np = np.ones(self.images_np.shape[:-1])
            logging.info(f"self.masks_np.shape: {self.masks_np.shape}")
        else:
            self.masks_np = np.stack([self.read_img(im_name, self.resolution_level) for im_name in masks_lis])[:,:,:,0] / 256.0

        if self.mask_out_image:
            self.images_np[np.where(self.masks_np < 0.5)] = 0.0

        # world_mat: projection matrix: world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the object is inside the unit sphere at origin.
        if self.estimate_scale_mat:
            self.scale_mats_np = self.estimated_scale_mat()
        else:
            self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []


        # i = 0
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            # if np.array_equal(world_mat, world_mat_0000):
            #     print("stop")
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            if self.resolution_level > 1.0:
                intrinsics[:2,:3] /= self.resolution_level
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())

            self.pose_all.append(torch.from_numpy(pose).float())

        # self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # n_images, H, W, 3   # Save GPU memory
        # self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # n_images, H, W, 3   # Save GPU memory
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cuda()  # n_images, H, W, 3   
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cuda()   # n_images, H, W, 3   
        h_img, w_img, _ = self.images[0].shape
        logging.info(f"Resolution level: {self.resolution_level}. Image size: ({w_img}, {h_img})")

        if self.use_normal:
            logging.info(f'[Use normal] Loading estimated normals...')
            normals_np = []
            normals_npz, stems_normal = read_images(f'{self.data_dir}/pred_normal', target_img_size=(w_img, h_img), img_ext='.npz')
            assert len(normals_npz) == self.n_images
            for i in tqdm(range(self.n_images)):
                normal_img_curr = normals_npz[i]
        
                # transform to world coordinates
                ex_i = torch.linalg.inv(self.pose_all[i])
                img_normal_w = get_world_normal(normal_img_curr.reshape(-1, 3), ex_i).reshape(h_img, w_img,3)

                normals_np.append(img_normal_w)
                
            self.normals_np = -np.stack(normals_np)   # reverse normal
            # self.normals = torch.from_numpy(self.normals_np.astype(np.float32)).cpu()
            self.normals = torch.from_numpy(self.normals_np.astype(np.float32)).cuda()

            debug_ = True
            if debug_ and IOUtils.checkExistence(f'{self.data_dir}/depth'):
                self.depths_np, stems_depth = read_images(f'{self.data_dir}/depth', target_img_size=(w_img, h_img), img_ext='.png')
                dir_depths_cloud = f'{self.data_dir}/depth_cloud'
                ensure_dir_existence(dir_depths_cloud)
                
                for i in range(5):
                    ext_curr = get_pose_inv(self.pose_all[i].detach().cpu().numpy())
                    pts = GeoUtils.get_world_points( self.depths_np[i], self.intrinsics_all[i], ext_curr)
                    normals_curr = self.normals_np[i].reshape(-1,3)
                    colors = self.images_np[i].reshape(-1,3)
                    save_points(f'{dir_depths_cloud}/{stems_depth[i]}.ply', pts, colors, normals_curr)

        if self.use_planes:
            logging.info(f'Use planes: Loading planes...')  

            planes_np = []
            planes_lis = sorted(glob(f'{self.data_dir}/pred_normal_planes/*.png'))
            assert len(planes_lis) == self.n_images
            for i in range(self.n_images):
                path = planes_lis[i]
                img_plane = cv.imread(path, cv.IMREAD_UNCHANGED)

                if img_plane.shape[0] != h_img:
                    img_plane = cv.resize(img_plane, (w_img, h_img), interpolation=cv.INTER_NEAREST)

                planes_np.append(img_plane)
            self.planes_np = np.stack(planes_np)
            # if self.planes_np.max() > 40:
            #     self.planes_np = self.planes_np // 40
            assert self.planes_np.max() <= 20 
            self.planes = torch.from_numpy(self.planes_np.astype(np.float32)).cpu()

        if self.use_plane_offset_loss:
            logging.info(f'Use planes: Loading subplanes...')  

            subplanes_np, stem_subplanes = read_images(f'{self.data_dir}/pred_normal_subplanes', 
                                                            target_img_size=(w_img, h_img), 
                                                            interpolation=cv.INTER_NEAREST, 
                                                            img_ext='.png')
            # subplanes_np = subplanes_np // 40
            assert subplanes_np.max() <= 20 
            self.subplanes = torch.from_numpy(subplanes_np.astype(np.uint8)).cpu()

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # n, 4, 4
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all) # n, 4, 4
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # n_images, 4, 4
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        # for patch match
        self.min_neighbors_ncc = 3
        if IOUtils.checkExistence(self.data_dir + '/neighbors.txt'):
            self.min_neighbors_ncc = 1  # images are relatively sparse
            path_neighbors = self.data_dir + '/neighbors.txt'
            logging.info(f'Use openMVS neighbors.')
            self.dict_neighbors = {}
            with open(path_neighbors, 'r') as fneighbor:
                lines = fneighbor.readlines()
                for line in lines:
                    line = line.split(' ')
                    line = np.array(line).astype(np.int32)
                    if len(line) > 1:
                        self.dict_neighbors[line[0]] = line[1:]
                    else:
                        logging.info(f'[{line[0]}] No neighbors, use adjacent views')
                        self.dict_neighbors[line[0]] = [np.min([line[0] + 1, self.n_images - 1]), np.min([line[0] - 1, 0]) ]
                
                for i in range(self.n_images):
                    if i not in self.dict_neighbors:
                        print(f'[View {i}] error: No nerighbors. Using {i-1, i+1}')
                        self.dict_neighbors[i] = [i-1,i+1]
                        msg = input('Check neighbor view...[y/n]')
                        if msg == 'n':
                            exit()
                assert len(self.dict_neighbors) == self.n_images       
        else:
            logging.info(f'Use adjacent views as neighbors.')


        self.initialize_patchmatch()
        
        # Gen train_data
        self.train_data = None
        if self.cache_all_data:
            train_data = []
            # Calc rays
            rays_o, rays_d = self.gen_rays()
            self.train_idx = []

            for i in range(len(self.images)):
                cur_data = torch.cat([rays_o[i], rays_d[i], self.images[i].to(self.device), self.masks[i, :, :, :1].to(self.device)], dim=-1).detach()
                # cur_data: [H, W, 10]
                cur_data = cur_data[torch.randperm(len(cur_data))]
                train_data.append(cur_data.reshape(-1, 10).detach().cpu())
            # train_data.append(cur_data.reshape(-1, 10))
            self.train_data = torch.stack(train_data).reshape(-1, 10).detach().cpu()
            self.train_piece = None
            self.train_piece_np = None
            self.large_counter = 0
            self.small_counter = 0
            del self.images
            del self.masks

        self.sphere_radius =  conf['sphere_radius']
        if checkExistence(f'{self.data_dir}/bbox.txt'):
            logging.info(f"Loading bbox.txt")
            bbox = np.loadtxt(f'{self.data_dir}/bbox.txt')
            self.bbox_min = bbox[:3]
            self.bbox_max = bbox[3:6]
        else:
            self.bbox_min = np.array([-1.01*self.bbox_size_half, -1.01*self.bbox_size_half, -1.01*self.bbox_size_half])
            self.bbox_max = np.array([ 1.01*self.bbox_size_half,  1.01*self.bbox_size_half,  1.01*self.bbox_size_half])

        self.iter_step = 0
        
    def initialize_patchmatch(self):
        self.check_occlusion = self.conf['check_occlusion']
        
        logging.info(f'Prepare gray images...')
        self.extrinsics_all = torch.linalg.inv(self.pose_all)
        self.images_gray = []
        self.images_denoise_np = []

        if self.denoise_gray_image:
            dir_denoise = f'{self.data_dir}/image_denoised_cv{self.denoise_paras[0]:02d}{self.denoise_paras[1]:02d}{self.denoise_paras[2]:02d}{self.denoise_paras[3]:02d}'
            if not checkExistence(dir_denoise) and len(get_files_stem(dir_denoise, '.png'))==0:
                logging.info(f'Use opencv structural denoise...')
                for i in tqdm(range(self.n_images)):
                    img_idx = (self.images_np[i]*256)
                    img_idx = cv.fastNlMeansDenoisingColored(img_idx.astype(np.uint8), None, h = self.denoise_paras[2], 
                                                                                            hColor = self.denoise_paras[3], 
                                                                                            templateWindowSize = self.denoise_paras[0], 
                                                                                            searchWindowSize = self.denoise_paras[1])
                    self.images_denoise_np.append(img_idx)

                self.images_denoise_np = np.array(self.images_denoise_np)

                # save denoised images
                stems_imgs = get_files_stem(f'{self.data_dir}/image', '.png')
                write_images(dir_denoise, self.images_denoise_np, stems_imgs)
            else:
                logging.info(f'Load predenoised images by openCV structural denoise: {dir_denoise}')
                images_denoised_lis = sorted(glob(f'{dir_denoise}/*.png'))
                self.images_denoise_np = np.stack([self.read_img(im_name, self.resolution_level) for im_name in images_denoised_lis])
        else:
            logging.info(f'Use original image to generate gray image...')
            self.images_denoise_np = self.images_np * 255

        for i in tqdm(range(self.n_images)):
            img_gray = cv.cvtColor(self.images_denoise_np[i].astype(np.uint8), cv.COLOR_BGR2GRAY)
            self.images_gray.append(img_gray)

        # range: (0,255)
        self.images_gray_np = np.array(self.images_gray).astype(np.float32)
        self.images_gray = torch.from_numpy(self.images_gray_np).cuda()

        # For cache rendered depths and normals
        self.confidence_accum = None
        self.samples_accum = None
        self.normals_accum = None
        self.depths_accum = None
        self.points_accum = None
        self.render_difference_accum = None
        self.samples_accum = torch.zeros_like(self.masks, dtype=torch.int32).cuda()
        
        b_accum_all_data = False
        if b_accum_all_data:
            self.colors_accum = torch.zeros_like(self.images, dtype=torch.float32).cuda()
                    
    def read_img(self, path, resolution_level):
        img = cv.imread(path)
        H, W = img.shape[0], img.shape[1]

        if resolution_level > 1.0:
            img = cv.resize(img, (int(W/resolution_level), int(H/resolution_level)), interpolation=cv.INTER_LINEAR)

            # save resized iamge for visulization
            ppath, stem, ext = get_path_components(path)
            dir_resize = ppath+f'_reso{int(resolution_level)}'
            logging.debug(f'Resize dir: {dir_resize}')
            os.makedirs(dir_resize, exist_ok=True)
            write_image(os.path.join(dir_resize, stem+ext), img)

        return img

    def estimated_scale_mat(self):
        assert len(self.world_mats_np) > 0
        rays_o = []
        rays_v = []
        for world_mat in self.world_mats_np:
            P = world_mat[:3, :4]
            intrinsics, c2w = load_K_Rt_from_P(None, P)
            rays_o.append(c2w[:3, 3])
            rays_v.append(c2w[:3, 0])
            rays_o.append(c2w[:3, 3])
            rays_v.append(c2w[:3, 1])

        rays_o = np.stack(rays_o, axis=0)   # N * 3
        rays_v = np.stack(rays_v, axis=0)   # N * 3
        dot_val = np.sum(rays_o * rays_v, axis=-1, keepdims=True)  # N * 1
        center, _, _, _ = np.linalg.lstsq(rays_v, dot_val)
        center = center.squeeze()
        radius = np.max(np.sqrt(np.sum((rays_o - center[None, :])**2, axis=-1)))
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center
        scale_mat = scale_mat.astype(np.float32)
        scale_mats = [scale_mat for _ in self.world_mats_np]

        return scale_mats

    def gen_rays(self):
        tx = torch.linspace(0, self.W - 1, self.W)
        ty = torch.linspace(0, self.H - 1, self.H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[:, None, None, :3, :3], p[None, :, :, :, None]).squeeze() # n, W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # n, W, H, 3
        rays_v = torch.matmul(self.pose_all[:, None, None, :3, :3],  rays_v[:, :, :, :, None]).squeeze()  # n, W, H, 3
        rays_o = self.pose_all[:, None, None, :3, 3].expand(rays_v.shape)  # n, W, H, 3
        return rays_o.transpose(1, 2), rays_v.transpose(1, 2)

    def get_pose(self, img_idx, pose):
        pose_cur = None
        if pose == None:
            pose_cur = self.pose_all[img_idx]
        elif pose is not None:
            if pose.dim() == 1:
                pose = pose.unsqueeze(0)
            assert pose.dim() == 2
            if pose.shape[1] == 7: #In case of quaternion vector representation
                cam_loc = pose[:, 4:]
                R = quat_to_rot(pose[:,:4])
                p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
                p[:, :3, :3] = R
                p[:, :3, 3] = cam_loc
            else: # In case of pose matrix representation
                cam_loc = pose[:, :3, 3]
                p = pose
            pose_cur = p
        else:
            NotImplementedError 

        return pose_cur.squeeze()

    def gen_rays_at(self, img_idx, pose = None, resolution_level=1):
        pose_cur = self.get_pose(img_idx, pose)

        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose_cur[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose_cur[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        key_rots = [rot_0, rot_1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10
        
    def random_get_rays_at(self, img_idx, batch_size, pose = None):
        pose_cur = self.get_pose(img_idx, pose)
        
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
            
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)][:,None]     # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(pose_cur[None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = pose_cur[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        
        normal_sample = None
        if self.use_normal:
            normal_sample = self.normals[img_idx][(pixels_y, pixels_x)].cuda()

        planes_sample = None
        if self.use_planes:
            planes_sample = self.planes[img_idx][(pixels_y, pixels_x)].unsqueeze(-1).cuda()
        
        subplanes_sample = None
        if self.use_plane_offset_loss:
            subplanes_sample = self.subplanes[img_idx][(pixels_y, pixels_x)].unsqueeze(-1).cuda()

        return torch.cat([rays_o, rays_v, color, mask], dim=-1).cuda(), pixels_x, pixels_y, normal_sample, planes_sample, subplanes_sample    # batch_size, 10

    def near_far_from_sphere(self, rays_o, rays_d):
        # torch
        assert self.sphere_radius is not None
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        c = torch.sum(rays_o ** 2, dim=-1, keepdim=True) - self.sphere_radius**2
        mid = 0.5 * (-b) / a
        near = mid - self.sphere_radius
        far = mid + self.sphere_radius
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def shuffle(self):
        r = torch.randperm(len(self.train_data))
        self.train_data = self.train_data[r]
        self.large_counter = 0
        self.small_counter = 0

    def next_train_batch(self, batch_size):
        if self.train_piece == None or self.small_counter + batch_size >= len(self.train_piece):
            if self.train_piece == None or self.large_counter + self.piece_size >= len(self.train_data):
                self.shuffle()
            self.train_piece_np = self.train_data[self.large_counter: self.large_counter + self.piece_size]
            self.train_piece = self.train_piece_np.cuda()
            self.small_counter = 0
            self.large_counter += self.piece_size

        curr_train_data = self.train_piece[self.small_counter: self.small_counter + batch_size]
        curr_train_data_np = self.train_piece_np[self.small_counter: self.small_counter + batch_size]
        self.small_counter += batch_size

        return curr_train_data, curr_train_data_np


    def score_pixels_ncc(self, idx, pts_world, normals_world, pixels_coords_vu, reso_level = 1.0, _debug = False):
        '''Use patch-match to evaluate the geometry: Smaller, better
        Return:
            scores_all_mean: N*1
            diff_patch_all: N*1
            mask_valid_all: N*1
        '''
        K = copy.deepcopy(self.intrinsics_all[0][:3,:3])
        img_ref = self.images_gray[idx]
        H, W = img_ref.shape
        window_size, window_step= 11, 2
        if reso_level > 1:
            K[:2,:3] /= reso_level
            img_ref = self.images_gray_np[idx]
            img_ref = cv.resize(img_ref, (int(W/reso_level), int(H/reso_level)), interpolation=cv.INTER_LINEAR)
            img_ref = torch.from_numpy(img_ref).cuda()
            window_size, window_step= (5, 1) if reso_level== 2 else (3, 1)

        if hasattr(self, 'dict_neighbors'):
            idx_neighbors = self.dict_neighbors[int(idx)]
            if len(idx_neighbors) < self.min_neighbors_ncc:
                return torch.ones(pts_world.shape[0]), torch.zeros(pts_world.shape[0]), torch.zeros(pts_world.shape[0]).bool()
        else:
            idx_neighbors = [idx-3, idx-2, idx-1, idx+1, idx+2, idx+3]
            if idx < 3:
                idx_neighbors = [idx+1, idx+2, idx+3]
            if idx > self.n_images-4:
                idx_neighbors = [idx-3, idx-2, idx-1]

        assert pixels_coords_vu.ndim == 2
        num_patches = pixels_coords_vu.shape[0]

        extrin_ref = self.extrinsics_all[idx]
        pts_ref = (extrin_ref[None,...] @ TrainingUtils.convert_to_homo(pts_world)[..., None]).squeeze()[:,:3]
        normals_ref = (extrin_ref[:3,:3][None,...] @ normals_world[..., None]).squeeze()
        
        patches_ref, idx_patch_pixels_ref, mask_idx_inside = PatchMatch.prepare_patches_src(img_ref, pixels_coords_vu, window_size, window_step)
        scores_all_mean, diff_patch_all, count_valid_all = torch.zeros(num_patches, dtype=torch.float32), torch.zeros(num_patches, dtype=torch.float32), torch.zeros(num_patches, dtype=torch.uint8)
        for idx_src in idx_neighbors:
            img_src = self.images_gray[idx_src]
            if reso_level > 1:
                img_src = cv.resize(self.images_gray_np[idx_src], (int(W/reso_level), int(H/reso_level)), interpolation=cv.INTER_LINEAR)
                img_src = torch.from_numpy(img_src).cuda()

            extrin_src = self.extrinsics_all[idx_src]

            homography = PatchMatch.compute_homography(pts_ref, normals_ref, K, extrin_ref, extrin_src)
            idx_patch_pixels_src = PatchMatch.warp_patches(idx_patch_pixels_ref, homography)
            patches_src = PatchMatch.sample_patches(img_src, idx_patch_pixels_src, sampling_mode = 'grid_sample')
            scores_curr, diff_patch_mean_curr, mask_patches_valid_curr = PatchMatch.compute_NCC_score(patches_ref, patches_src)

            # check occlusion
            if self.check_occlusion:
                mask_no_occlusion = scores_curr < 0.66
                mask_patches_valid_curr = mask_patches_valid_curr & mask_no_occlusion
                scores_curr[mask_no_occlusion==False] = 0.0
                diff_patch_mean_curr[mask_no_occlusion==False] = 0.0

            scores_all_mean += scores_curr
            diff_patch_all += diff_patch_mean_curr
            count_valid_all += mask_patches_valid_curr

            if _debug:
                corords_src = idx_patch_pixels_src[:,3,3].cpu().numpy().astype(int)
                img_sample_ref = PatchMatch.visualize_sampled_pixels(self.images[idx].numpy()*255, pixels_coords_vu.cpu().numpy())
                img_sample_src = PatchMatch.visualize_sampled_pixels(self.images[idx_src].numpy()*255, corords_src)
                ImageUtils.write_image_lis(f'./test/ncc/{idx}_{idx_src}.png', [img_sample_ref, img_sample_src])

                # save patches
                ImageUtils.write_image_lis(f'./test/ncc/patches_{idx}.png',[ patches_ref[i].cpu().numpy() for i in range(len(patches_ref))], interval_img = 5 )
                ImageUtils.write_image_lis(f'./test/ncc/patches_{idx_src}.png',[ patches_ref[i].cpu().numpy() for i in range(len(patches_src))], interval_img = 5 )

        
        # get the average scores of all neighbor views
        mask_valid_all = count_valid_all>=self.min_neighbors_ncc
        scores_all_mean[mask_valid_all] /= count_valid_all[mask_valid_all]
        diff_patch_all[mask_valid_all]  /= count_valid_all[mask_valid_all]

        # set unvalid scores and diffs to zero
        scores_all_mean = scores_all_mean*mask_valid_all
        diff_patch_all  = diff_patch_all*mask_valid_all


        scores_all_mean[mask_valid_all==False] = 1.0 # average scores for pixels without patch.

        return scores_all_mean, diff_patch_all, mask_valid_all
            