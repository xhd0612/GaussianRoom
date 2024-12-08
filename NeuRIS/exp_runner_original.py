import cv2 as cv
import numpy as np
import os, logging, argparse, trimesh, copy

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.dataset import Dataset
from models.fields import SDFNetwork, RenderingNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer, extract_fields
from models.nerf_renderer import NeRFRenderer
from models.loss import NeuSLoss

import models.patch_match_cuda as PatchMatch
from shutil import copyfile
from tqdm import tqdm
from icecream import ic
from datetime import datetime
from pyhocon import ConfigFactory, HOCONConverter

import utils.utils_io as IOUtils
import utils.utils_geometry as GeoUtils
import utils.utils_image as ImageUtils
import utils.utils_training as TrainingUtils


class Runner:
    def __init__(self, conf_path, scene_name = '', mode='train', model_type='', is_continue=False, checkpoint_id = -1):
        # Initial setting: Genreal
        self.device = torch.device('cuda')
        self.conf_path = conf_path
           
        self.conf = ConfigFactory.parse_file(conf_path)
        self.dataset_type = self.conf['general.dataset_type']
        self.scan_name = self.conf['general.scan_name']
        if len(scene_name)>0:
            self.scan_name = scene_name
            print(f"**********************Scan name: {self.scan_name}********************** \n\n")

        self.exp_name = self.conf['general.exp_name']

        # parse cmd args
        self.is_continue = is_continue
        self.checkpoint_id = checkpoint_id
        self.mode = mode
        self.model_type = self.conf['general.model_type']
        if model_type != '':  # overwrite
            self.model_type = model_type
        self.model_list = []
        self.writer = None
        self.curr_img_idx = -1

        self.parse_parameters()
        self.build_dataset()
        self.build_model()

        if self.is_continue:
            self.load_pretrained_model()
                               
        # recording
        if self.mode[:5] == 'train':
            self.file_backup()
            with open(f'{self.base_exp_dir}/recording/config_modified.conf', 'w') as configfile:
                configfile.write(HOCONConverter.to_hocon(self.conf))

    def load_pretrained_model(self):
        # Load checkpoint
        latest_model_name = None

        model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
        model_list = []
        for model_name in model_list_raw:
            if 'pretrained_sdf' in model_name:
                continue
            if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                model_list.append(model_name)
        model_list.sort()
        if self.checkpoint_id == -1: 
            latest_model_name = model_list[-1]
        else:
            if f'ckpt_{self.checkpoint_id:06d}.pth' in model_list:
                latest_model_name = f'ckpt_{self.checkpoint_id:06d}.pth'
            else:
                logging.error(f"path_ckpt is not exist. (f'ckpt_{self.checkpoint_id:06d}.pth')")
                exit()

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def parse_parameters(self):
        # patch-match
        self.thres_robust_ncc = self.conf['dataset.patchmatch_thres_ncc_robust']
        logging.info(f'Robust ncc threshold: {self.thres_robust_ncc}')
        self.patchmatch_start = self.conf['dataset.patchmatch_start']
        self.patchmatch_mode =  self.conf['dataset.patchmatch_mode']

        self.nvs = self.conf.get_bool('train.nvs', default=False)
        self.save_normamap_npz = self.conf['train.save_normamap_npz']

        self.base_exp_dir = os.path.join(self.conf['general.exp_dir'], self.dataset_type, self.model_type, self.scan_name, str(self.exp_name))
        os.makedirs(self.base_exp_dir, exist_ok=True)
        logging.info(f'Exp dir: {self.base_exp_dir}')

        self.iter_step = 0
        self.radius_norm = 1.0

        # trainning parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_milestone = self.conf.get_list('train.learning_rate_milestone')
        self.learning_rate_factor = self.conf.get_float('train.learning_rate_factor')
        
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        logging.info(f'Warm up end: {self.warm_up_end}; learning_rate_alpha: {self.learning_rate_alpha}')

        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.anneal_start = self.conf.get_float('train.anneal_start', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        self.n_samples = self.conf.get_int('model.neus_renderer.n_samples')

        # validate parameters
        for attr in ['save_freq','report_freq', 'val_image_freq', 'val_mesh_freq', 'val_fields_freq', 
                        'freq_valid_points', 'freq_valid_weights',
                        'freq_save_confidence' ]:
            setattr(self, attr,  self.conf.get_int(f'train.{attr}'))

    def build_dataset(self):
        # dataset and directories
        if self.dataset_type == 'indoor':
            self.sample_range_indoor =  self.conf['dataset']['sample_range_indoor']
            self.conf['dataset']['use_normal'] = True if self.conf['model.loss.normal_weight'] > 0 else False
            self.use_indoor_data = True
            logging.info(f"Ray sample range: {self.sample_range_indoor}")
        elif self.dataset_type == 'dtu':
            self.use_indoor_data = False
            self.conf['model.sdf_network']['reverse_geoinit'] = False
            self.conf['dataset']['use_normal'] = True if self.conf['model.loss.normal_weight'] > 0 else False

            self.scan_id = int(self.scan_name[4:])
            logging.info(f"DTU scan ID: {self.scan_id}")
        else:
            raise NotImplementedError
        logging.info(f"Use normal: {self.conf['dataset']['use_normal']}")

        
        self.conf['dataset']['use_planes'] = True if self.conf['model.loss.normal_consistency_weight'] > 0 else False
        self.conf['dataset']['use_plane_offset_loss'] = True if self.conf['model.loss.plane_offset_weight'] > 0 else False

        self.conf['dataset']['data_dir']  = os.path.join(self.conf['general.data_dir'] , self.dataset_type, self.scan_name)
        self.dataset = Dataset(self.conf['dataset'])

    def build_model(self):
        # Networks
        params_to_train = []
        if self.model_type == 'neus':
            self.nerf_outside = NeRF(**self.conf['model.tiny_nerf']).to(self.device)
            self.sdf_network_fine = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
            self.variance_network_fine = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
            self.color_network_fine = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
            params_to_train += list(self.nerf_outside.parameters())
            params_to_train += list(self.sdf_network_fine.parameters())
            params_to_train += list(self.variance_network_fine.parameters())
            params_to_train += list(self.color_network_fine.parameters())

            self.renderer = NeuSRenderer(self.nerf_outside,
                                self.sdf_network_fine,
                                self.variance_network_fine,
                                self.color_network_fine,
                                **self.conf['model.neus_renderer'])
        
        elif self.model_type == 'nerf':  # NeRF
            self.nerf_coarse = NeRF(**self.conf['model.nerf']).to(self.device)
            self.nerf_fine = NeRF(**self.conf['model.nerf']).to(self.device)
            self.nerf_outside = NeRF(**self.conf['model.tiny_nerf']).to(self.device)
            params_to_train += list(self.nerf_coarse.parameters())
            params_to_train += list(self.nerf_fine.parameters())
            params_to_train += list(self.nerf_outside.parameters())

            self.renderer = NeRFRenderer(self.nerf_coarse,
                                self.nerf_fine,
                                self.nerf_outside,
                                **self.conf['model.nerf_renderer'])
                                         
        else:
            NotImplementedError

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
                
        self.loss_neus = NeuSLoss(self.conf['model.loss'])

    def train(self):
        if self.model_type == 'neus':
            self.train_neus()
        elif self.model_type == 'nerf':
            self.train_nerf()
        else:
            NotImplementedError

    def get_near_far(self, rays_o, rays_d,  image_perm = None, iter_i= None, pixels_x = None, pixels_y= None):
        log_vox = {}
        log_vox.clear()
        batch_size = len(rays_o)
        if  self.dataset_type == 'dtu':
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        elif self.dataset_type == 'indoor':
            near, far = torch.zeros(batch_size, 1), self.sample_range_indoor * torch.ones(batch_size, 1)
        else:
            NotImplementedError

        logging.debug(f"[{self.iter_step}] Sample range: max, {torch.max(far -near)}; min, {torch.min(far -near)}")
        return near, far, log_vox

    def get_model_input(self, image_perm, iter_i):
        input_model = {}

        idx_img = image_perm[self.iter_step % self.dataset.n_images]
        self.curr_img_idx = idx_img
        data, pixels_x, pixels_y,  normal_sample, planes_sample, subplanes_sample = self.dataset.random_get_rays_at(idx_img, self.batch_size)
        
        rays_o, rays_d, true_rgb, true_mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
        true_mask =  (true_mask > 0.5).float()
        mask = true_mask

        if self.conf['model.loss.normal_weight'] > 0:
            input_model['normals_gt'] = normal_sample
        
        if self.conf['model.loss.normal_consistency_weight'] > 0:
            input_model['planes_gt'] = planes_sample

        if self.conf['model.loss.plane_offset_weight'] > 0:
            input_model['subplanes_gt'] = subplanes_sample

        near, far, logs_input = self.get_near_far(rays_o, rays_d,  image_perm, iter_i, pixels_x, pixels_y)
        
        background_rgb = None
        if self.use_white_bkgd:
            background_rgb = torch.ones([1, 3])

        if self.conf['model.loss.mask_weight'] > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)

        mask_sum = mask.sum() + 1e-5

        pixels_uv = torch.stack([pixels_x, pixels_y], dim=-1)
        pixels_vu = torch.stack([pixels_y, pixels_x], dim=-1)
        input_model.update({
            'rays_o': rays_o,
            'rays_d': rays_d,
            'near': near,
            'far': far,
            'mask': mask,
            'mask_sum': mask_sum,
            'true_rgb': true_rgb,
            'background_rgb': background_rgb,
            'pixels_x': pixels_x,  # u
            'pixels_y': pixels_y,   # v,
            'pixels_uv': pixels_uv,
            'pixels_vu': pixels_vu
        })
        return input_model, logs_input

    def train_neus(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        self.update_iter_step()
        res_step = self.end_iter - self.iter_step

        if self.dataset.cache_all_data:
            self.dataset.shuffle()

        # self.validate_mesh() # save mesh at iter 0
        logs_summary = {}
        image_perm = torch.randperm(self.dataset.n_images) # 打乱索引
        for iter_i in tqdm(range(res_step)):  
            logs_summary.clear()

            input_model, logs_input = self.get_model_input(image_perm, iter_i)
            logs_summary.update(logs_input)

            render_out, logs_render = self.renderer.render(input_model['rays_o'], input_model['rays_d'], 
                                            input_model['near'], input_model['far'],
                                            background_rgb=input_model['background_rgb'],
                                            alpha_inter_ratio=self.get_alpha_inter_ratio())
            logs_summary.update(logs_render)

            patchmatch_out, logs_patchmatch = self.patch_match(input_model, render_out) # 是啥作用 衡量视觉一致性判断是否进行指导
            logs_summary.update(logs_patchmatch)

            loss, logs_loss, mask_keep_gt_normal = self.loss_neus(input_model, render_out, self.sdf_network_fine, patchmatch_out)
            logs_summary.update(logs_loss)

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()


            self.iter_step += 1

            logs_val = self.validate(input_model, logs_loss, render_out)
            logs_summary.update(logs_val)
            
            logs_summary.update({'Log/lr': self.optimizer.param_groups[0]['lr']})
            self.write_summary(logs_summary)

            self.update_learning_rate()
            self.update_iter_step()
            self.accumulate_rendered_results(input_model, render_out, patchmatch_out,
                                                    b_accum_render_difference = False,
                                                    b_accum_ncc = False,
                                                    b_accum_normal_pts = False)

            if self.iter_step % self.dataset.n_images == 0:
                image_perm = torch.randperm(self.dataset.n_images)

            if self.iter_step % 1000 == 0:
                torch.cuda.empty_cache()

        logging.info(f'Done. [{self.base_exp_dir}]')

    def calculate_ncc_samples(self, input_model, render_out, 
                                    use_peak_value = True, 
                                    use_normal_prior = False):
        pixels_coords_vu =input_model['pixels_vu'] 
        
        if use_peak_value:
            pts_render = render_out['point_peak']
            normals_render = render_out['normal_peak']
        else:
            raise NotImplementedError
        
        if use_normal_prior:
            normals_render = input_model['normals_gt']

        pts_all = pts_render[:, None, :]
        normals_all = normals_render[:, None, :]
        coords_all = pixels_coords_vu[:, None, :]

        with torch.no_grad():
            scores_ncc_all, diff_patch_all, mask_valid_all = self.dataset.score_pixels_ncc(self.curr_img_idx, pts_all.reshape(-1, 3), 
                                                                                                normals_all.reshape(-1, 3), 
                                                                                                coords_all.reshape(-1, 2))
            num_valid = mask_valid_all.sum()
            scores_ncc_all = scores_ncc_all.reshape(self.batch_size, -1)
            mask_valid_all = mask_valid_all.reshape(self.batch_size, -1)
        return scores_ncc_all, diff_patch_all, mask_valid_all
                      
    def patch_match(self, input_model, render_out):
        patchmatch_out, logs_summary = None, {}
        if self.iter_step > self.patchmatch_start:
            # ensure initialization of confidence_map, depth_map and points_map
            if self.dataset.confidence_accum is None:
                self.initialize_accumulated_results(mode=self.conf['dataset.mode_init_accum'], 
                                                    iter_step_npz=self.conf['dataset.init_accum_step_npz'],
                                                    resolution_level=self.conf['dataset.init_accum_reso_level'])

            if self.patchmatch_mode == 'use_geocheck':
                scores_ncc, diffs_patch, mask_valid = self.calculate_ncc_samples(input_model, render_out) # Normalized Cross Correlation 用于衡量 visual consistency

                # (1) mask of pixels with high confidence
                mask_high_confidence_curr = scores_ncc < self.thres_robust_ncc

                # (2) check previous ncc confidence
                pixels_u, pixels_v = input_model['pixels_x'].cpu().numpy(), input_model['pixels_y'].cpu().numpy()
                ncc_prev = self.dataset.confidence_accum[self.curr_img_idx][pixels_v, pixels_u]
                mask_high_confidence_prev = ncc_prev < self.thres_robust_ncc

                # (3) remove normals with large difference between rendered normal and gt normal
                # large difference means large inconsistency of normal between differenct views
                normal_render = render_out['normal_peak']
                angles_diff = TrainingUtils.get_angles(normal_render, input_model['normals_gt'])
                MAX_ANGLE_DIFF = 30.0 / 180.0 * 3.1415926
                mask_small_normal_diffence = angles_diff < MAX_ANGLE_DIFF

                # (4) merge masks
                mask_use_gt = mask_high_confidence_curr & torch.from_numpy(mask_high_confidence_prev[:,None]).cuda() & mask_small_normal_diffence
                
                # (5) update confidence, discard the normals
                mask_not_use_gt = (mask_use_gt==False)
                scores_ncc_all2 = copy.deepcopy(scores_ncc)
                scores_ncc_all2[mask_not_use_gt] = 1.0
                self.dataset.confidence_accum[self.curr_img_idx][pixels_v, pixels_u] = scores_ncc_all2.squeeze().cpu().numpy()
        
                idx_scores_min = mask_use_gt.long()  # 1 use gt, 0 not
                patchmatch_out = {
                    'patchmatch_mode': self.patchmatch_mode,
                    'scores_ncc_all': scores_ncc,
                    'idx_scores_min': idx_scores_min,
                }
                logs_summary.update({
                    'Log/pixels_use_volume_rendering_only': (idx_scores_min==0).sum(),
                    'Log/pixels_use_prior': (idx_scores_min==1).sum(),
                })
                
            else:
                raise NotImplementedError

        return patchmatch_out, logs_summary

    def accumulate_rendered_results(self, input_model, render_out, patchmatch_out,
                                        b_accum_render_difference = False,
                                        b_accum_ncc = False,
                                        b_accum_normal_pts = False):
        '''Cache rendererd depth, normal and confidence (if avaliable) in the training process
        
        Args:
            scores: (N,)
        '''
        pixels_u, pixels_v = input_model['pixels_x'], input_model['pixels_y']
        pixels_u_np, pixels_v_np = pixels_u.cpu().numpy(), pixels_v.cpu().numpy()
        if b_accum_render_difference:
            color_gt = self.dataset.images_denoise_np[self.curr_img_idx.item()][pixels_v_np, pixels_u_np]
            color_render = render_out['color_fine'].detach().cpu.numpy()
            diff = np.abs(color_render - color_gt).sum(axis=-1)
            self.dataset.render_difference_accum[self.curr_img_idx.item()][pixels_v_np, pixels_u_np] = diff

        if b_accum_ncc:
            scores_vr = patchmatch_out['scores_samples'] 
            self.dataset.confidence_accum[self.curr_img_idx.item()][pixels_v_np, pixels_u_np] = scores_vr.cpu().numpy()
   
        if b_accum_normal_pts:
            point_peak = render_out['point_peak']
            self.dataset.points_accum[self.curr_img_idx][pixels_v, pixels_u]  = point_peak.detach().cpu()

            normal_peak = render_out['normal_peak']
            self.dataset.normals_accum[self.curr_img_idx][pixels_v, pixels_u] = normal_peak.detach().cpu()

        b_accum_all_data = False
        if b_accum_all_data:
            self.dataset.depths_accum[self.curr_img_idx][pixels_v, pixels_u]  = render_out['depth_peak'].squeeze()
            self.dataset.colors_accum[self.curr_img_idx][pixels_v, pixels_u]  = render_out['color_peak']

    def write_summary(self, logs_summary):
        for key in logs_summary:
            self.writer.add_scalar(key, logs_summary[key], self.iter_step )

    def update_iter_step(self):
        self.sdf_network_fine.iter_step = self.iter_step
        self.sdf_network_fine.end_iter = self.end_iter
        
        self.loss_neus.iter_step = self.iter_step
        self.loss_neus.iter_end = self.end_iter

        self.dataset.iter_step = self.iter_step

    def get_alpha_inter_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        elif self.iter_step < self.anneal_start:
            return 0.0
        else:
            return np.min([1.0, (self.iter_step - self.anneal_start) / (self.anneal_end - self.anneal_start)])

    def get_alpha_inter_ratio_decrease(self):
        anneal_end = 5e4
        anneal_start = 0
        
        if anneal_end == 0.0:
            return 0.0
        elif self.iter_step < anneal_start:
            return 1.0
        else:
            return 1.0 - np.min([1.0, (self.iter_step - anneal_start) / (anneal_end - anneal_start)])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def train_nerf(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data, _, _, normal_sample, planes_sample, subplanes_sample, mask_normal_certain_sample = self.dataset.random_get_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            
            if self.dataset_type == 'dtu':
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
            elif self.dataset_type == 'indoor':
                near, far = torch.zeros(len(rays_o), 1), self.sample_range_indoor * torch.ones(len(rays_o), 1)
            else:
                NotImplementedError

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.conf['model.loss.mask_weight'] > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb)

            color_coarse = render_out['color_coarse']
            color_fine = render_out['color_fine']
            weight_sum = render_out['weight_sum']

            # Color loss
            color_coarse_error = (color_coarse - true_rgb) * mask
            color_coarse_loss = F.mse_loss(color_coarse_error, torch.zeros_like(color_coarse_error), reduction='sum') / mask_sum
            color_fine_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.mse_loss(color_fine_error, torch.zeros_like(color_fine_error), reduction='sum') / mask_sum

            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            # Mask loss, optional
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_coarse_loss + color_fine_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/color_coarse_loss', color_coarse_loss, self.iter_step)

            self.writer.add_scalar('Log/psnr', psnr, self.iter_step)
            self.writer.add_scalar('Log/lr', self.optimizer.param_groups[0]['lr'], self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                logging.info('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_image_freq == 0:
                self.validate_image_nerf()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh_nerf()

            if self.iter_step % self.freq_eval_mesh == 0:
                self.validate_mesh_nerf(world_space=False, resolution=512, threshold=25)

            self.update_learning_rate()
            self.dataset.iter_step = self.iter_step

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        if not self.nvs:
            return torch.randperm(self.dataset.n_images)
        else:
            lis = [i for i in range(self.dataset.n_images) if not i in [8, 13, 16, 21, 26, 31, 34, 56]]
            lis = torch.tensor(lis, dtype=torch.long)
            return lis[torch.randperm(len(lis))]

    def file_backup(self):
        # copy python file
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        # copy configs
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        if self.model_type == 'neus':
            self.nerf_outside.load_state_dict(checkpoint['nerf'])
            self.sdf_network_fine.load_state_dict(checkpoint['sdf_network_fine'])
            self.variance_network_fine.load_state_dict(checkpoint['variance_network_fine'])
            self.color_network_fine.load_state_dict(checkpoint['color_network_fine'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.iter_step = checkpoint['iter_step']

        elif self.model_type == 'nerf':
            self.nerf_coarse.load_state_dict(checkpoint['nerf_coarse'])
            self.nerf_fine.load_state_dict(checkpoint['nerf_fine'])
            self.nerf_outside.load_state_dict(checkpoint['nerf_outside'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.iter_step = checkpoint['iter_step']
            
        else:
            NotImplementedError

    def save_checkpoint(self):
        checkpoint = None
        if self.model_type == 'neus':
            checkpoint = {
                'nerf': self.nerf_outside.state_dict(),
                'sdf_network_fine': self.sdf_network_fine.state_dict(),
                'variance_network_fine': self.variance_network_fine.state_dict(),
                'color_network_fine': self.color_network_fine.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
            }

        elif self.model_type == 'nerf':
            checkpoint = {
                'nerf_coarse': self.nerf_coarse.state_dict(),
                'nerf_fine': self.nerf_fine.state_dict(),
                'nerf_outside': self.nerf_outside.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
            }

        else:
            NotImplementedError

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate(self, input_model, loss_out, render_out):
        mask, rays_o, rays_d, near, far = input_model['mask'], input_model['rays_o'], input_model['rays_d'],  \
                                                    input_model['near'], input_model['far']
        mask_sum = mask.sum() + 1e-5
        loss, psnr = loss_out['Loss/loss'], loss_out['Log/psnr']

        log_val = {}
        if self.iter_step % self.report_freq == 0:
            print('\n', self.base_exp_dir)
            logging.info('iter:{:8>d} loss={:.03f} lr={:.06f} var={:.04f}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr'], render_out['variance'].mean()))
            ic((render_out['weight_sum'] * mask).sum() / mask_sum)
            ic(self.get_alpha_inter_ratio())
            ic(psnr)

        if self.iter_step % self.save_freq == 0:
            self.save_checkpoint()

        if self.iter_step % self.val_mesh_freq == 0 or self.iter_step == 1:
            self.validate_mesh()
        
        if self.iter_step % self.val_image_freq == 0:
            self.validate_image(save_normalmap_npz=self.save_normamap_npz)

        if self.iter_step % self.val_fields_freq == 0:
            self.validate_fields()
        
        if self.iter_step % self.freq_valid_points == 0:
            self.validate_points(rays_o, rays_d, near, far)
            
        if self.iter_step % self.freq_save_confidence == 0:
            self.save_accumulated_results()
            self.save_accumulated_results(idx=self.dataset.n_images//2)
    
        return log_val

    def validate_image(self, idx=-1, resolution_level=-1, 
                                save_normalmap_npz = False, 
                                save_peak_value = False, 
                                validate_confidence = True,
                                save_image_render = False):
        # validate image
        ic(self.iter_step, idx)
        logging.info(f'Validate begin: idx {idx}...')
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        
        imgs_render = {}
        for key in ['color_fine', 'confidence', 'normal', 'depth', 'variance_surface', 'confidence_mask']:
            imgs_render[key] = []
        
        if save_peak_value:
            imgs_render.update({
                'color_peak': [], 
                'normal_peak': [], 
                'depth_peak':[]
            })
            pts_peak_all = []

        # (1) render images
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)

        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        idx_pixels_vu = torch.tensor([[i, j] for i in range(0, H) for j in range(0, W)]).split(self.batch_size)
        for rays_o_batch, rays_d_batch, idx_pixels_vu_batch in zip(rays_o, rays_d, idx_pixels_vu):
            near, far, _ = self.get_near_far(rays_o = rays_o_batch, rays_d = rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out, _ = self.renderer.render(rays_o_batch, rays_d_batch, near, far, alpha_inter_ratio=self.get_alpha_inter_ratio(), background_rgb=background_rgb)
            feasible = lambda key: ((key in render_out) and (render_out[key] is not None))

            for key in imgs_render:
                if feasible(key):
                    imgs_render[key].append(render_out[key].detach().cpu().numpy())
                    
            if validate_confidence:
                pts_peak = rays_o_batch + rays_d_batch * render_out['depth_peak']
                scores_all_mean, diff_patch_all, mask_valid_all = self.dataset.score_pixels_ncc(idx, pts_peak, render_out['normal_peak'], idx_pixels_vu_batch, reso_level=resolution_level)
                
                imgs_render['confidence'].append(scores_all_mean.detach().cpu().numpy()[:,None])
                imgs_render['confidence_mask'].append(mask_valid_all.detach().cpu().numpy()[:,None])
                if save_peak_value:
                    pts_peak_all.append(pts_peak.detach().cpu().numpy())

            del render_out

        # (2) reshape rendered images
        for key in imgs_render:
            if len(imgs_render[key]) > 0:
                imgs_render[key] = np.concatenate(imgs_render[key], axis=0)
                if imgs_render[key].shape[1] == 3: # for color and normal
                    imgs_render[key] = imgs_render[key].reshape([H, W, 3])
                elif imgs_render[key].shape[1] == 1:  
                    imgs_render[key] = imgs_render[key].reshape([H, W])

        # confidence map
        if save_normalmap_npz:
            # For each view, save point(.npz), depthmap(.png) point cloud(.ply), normalmap(.png), normal(.npz)
            # rendered depth in volume rednering and projected depth of points in world are different
            shape_depthmap = imgs_render['depth'].shape[:2]
            pts_world = torch.vstack(rays_o).cpu().numpy() +  torch.vstack(rays_d).cpu().numpy() * imgs_render['depth'].reshape(-1,1)
            os.makedirs(os.path.join(self.base_exp_dir, 'depth'), exist_ok=True)
            GeoUtils.save_points(os.path.join(self.base_exp_dir, 'depth', f'{self.iter_step:0>8d}_{idx}_reso{resolution_level}.ply'),
                                    pts_world.reshape(-1,3),
                                    colors = imgs_render['color_fine'].squeeze().reshape(-1,3),
                                    normals= imgs_render['normal'].squeeze().reshape(-1,3),
                                    BRG2RGB=True)
            
            # save peak depth and normal
            if save_peak_value:
                pts_world_peak = torch.vstack(rays_o).cpu().numpy() +  torch.vstack(rays_d).cpu().numpy() * imgs_render['depth_peak'].reshape(-1,1)
                os.makedirs(os.path.join(self.base_exp_dir, 'depth'), exist_ok=True)
                GeoUtils.save_points(os.path.join(self.base_exp_dir, 'depth', f'{self.iter_step:0>8d}_{idx}_reso{resolution_level}_peak.ply'),
                                        pts_world_peak.reshape(-1,3),
                                        colors = imgs_render['color_peak'].squeeze().reshape(-1,3),
                                        normals= imgs_render['normal_peak'].squeeze().reshape(-1,3),
                                        BRG2RGB=True)

                os.makedirs(os.path.join(self.base_exp_dir, 'normal_peak'), exist_ok=True)
                np.savez(os.path.join(self.base_exp_dir, 'normal_peak', f'{self.iter_step:08d}_{self.dataset.vec_stem_files[idx]}_reso{resolution_level}.npz'), 
                            imgs_render['normal_peak'].squeeze())
                
                os.makedirs(os.path.join(self.base_exp_dir, 'normal_render'), exist_ok=True)
                np.savez(os.path.join(self.base_exp_dir, 'normal_render', f'{self.iter_step:08d}_{self.dataset.vec_stem_files[idx]}_reso{resolution_level}.npz'), 
                            imgs_render['normal'].squeeze())
                
                pts_world2 = pts_world_peak.reshape([H, W, 3])
                np.savez(os.path.join(self.base_exp_dir, 'depth', f'{self.iter_step:08d}_{idx}_reso{resolution_level}_peak.npz'),
                            pts_world2)
            
        if save_image_render:
            os.makedirs(os.path.join(self.base_exp_dir, 'image_render'), exist_ok=True)
            ImageUtils.write_image(os.path.join(self.base_exp_dir, 'image_render', f'{self.iter_step:08d}_{self.dataset.vec_stem_files[idx]}_reso{resolution_level}.png'), 
                        imgs_render['color_fine']*255)
            psnr_render = 20.0 * torch.log10(1.0 / (((self.dataset.images[idx] - torch.from_numpy(imgs_render['color_fine']))**2).sum() / (imgs_render['color_fine'].size * 3.0)).sqrt())
            
            os.makedirs(os.path.join(self.base_exp_dir, 'image_peak'), exist_ok=True)
            ImageUtils.write_image(os.path.join(self.base_exp_dir, 'image_peak', f'{self.iter_step:08d}_{self.dataset.vec_stem_files[idx]}_reso{resolution_level}.png'), 
                        imgs_render['color_peak']*255)    
            psnr_peak = 20.0 * torch.log10(1.0 / (((self.dataset.images[idx] - torch.from_numpy(imgs_render['color_peak']))**2).sum() / (imgs_render['color_peak'].size * 3.0)).sqrt())
            
            print(f'PSNR (rener, peak): {psnr_render}  {psnr_peak}')
        
        # (3) save images
        lis_imgs = []
        img_sample = np.zeros_like(imgs_render['color_fine'])
        img_sample[:,:,1] = 255
        for key in imgs_render:
            if len(imgs_render[key]) == 0 or key =='confidence_mask':
                continue

            img_temp = imgs_render[key]
            if key in ['normal', 'normal_peak']:
                img_temp = (((img_temp + 1) * 0.5).clip(0,1) * 255).astype(np.uint8)
            if key in ['depth', 'depth_peak', 'variance']:
                img_temp = img_temp / (np.max(img_temp)+1e-6) * 255
            if key == 'confidence':
                img_temp2 = ImageUtils.convert_gray_to_cmap(img_temp)
                img_temp2 = img_temp2 * imgs_render['confidence_mask'][:,:,None]
                lis_imgs.append(img_temp2)

                img_temp3_mask_use_prior = (img_temp<self.thres_robust_ncc)
                lis_imgs.append(img_temp3_mask_use_prior*255)
                img_sample[img_temp==1.0] = (255,0,0)
                logging.info(f'Pixels: {img_temp3_mask_use_prior.size}; Use prior: {img_temp3_mask_use_prior.sum()}; Invalid patchmatch: {(img_temp==1.0).sum()}')
                logging.info(f'Ratio: Use prior: {img_temp3_mask_use_prior.sum()/img_temp3_mask_use_prior.size}; Invalid patchmatch: {(img_temp==1.0).sum()/img_temp3_mask_use_prior.size}')

                img_temp = img_temp.clip(0,1) * 255
            if key in ['color_fine', 'color_peak', 'confidence_mask']:
                img_temp *= 255

            cv.putText(img_temp, key,  (img_temp.shape[1]-100, img_temp.shape[0]-20), 
                fontFace= cv.FONT_HERSHEY_SIMPLEX, 
                fontScale = 0.5, 
                color = (0, 0, 255), 
                thickness = 2)

            lis_imgs.append(img_temp)
        
        dir_images = os.path.join(self.base_exp_dir, 'images')
        os.makedirs(dir_images, exist_ok=True)
        img_gt = ImageUtils.resize_image(self.dataset.images[idx].cpu().numpy(), 
                                            (lis_imgs[0].shape[1], lis_imgs[0].shape[0]))
        img_sample[img_temp3_mask_use_prior] = img_gt[img_temp3_mask_use_prior]*255
        ImageUtils.write_image_lis(f'{dir_images}/{self.iter_step:08d}_reso{resolution_level}_{idx:08d}.png',
                                        [img_gt, img_sample] + lis_imgs)

        if save_peak_value:
            pts_peak_all = np.concatenate(pts_peak_all, axis=0)
            pts_peak_all = pts_peak_all.reshape([H, W, 3])

            return imgs_render['confidence'], imgs_render['color_peak'], imgs_render['normal_peak'], imgs_render['depth_peak'], pts_peak_all, imgs_render['confidence_mask']

    def compare_ncc_confidence(self, idx=-1, resolution_level=-1):
        # validate image
        ic(self.iter_step, idx)
        logging.info(f'Validate begin: idx {idx}...')
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        # (1) render images
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)

        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        normals_gt = self.dataset.normals[idx].cuda()
        normals_gt = normals_gt.reshape(-1, 3).split(self.batch_size)
        scores_render_all, scores_gt_all = [], []

        idx_pixels_vu = torch.tensor([[i, j] for i in range(0, H) for j in range(0, W)]).split(self.batch_size)
        for rays_o_batch, rays_d_batch, idx_pixels_vu_batch, normals_gt_batch in zip(rays_o, rays_d, idx_pixels_vu, normals_gt):
            near, far, _ = self.get_near_far(rays_o = rays_o_batch, rays_d = rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out, _ = self.renderer.render(rays_o_batch, rays_d_batch, near, far, alpha_inter_ratio=self.get_alpha_inter_ratio(), background_rgb=background_rgb)


            pts_peak = rays_o_batch + rays_d_batch * render_out['depth_peak']
            scores_render, diff_patch_all, mask_valid_all = self.dataset.score_pixels_ncc(idx, pts_peak, render_out['normal_peak'], idx_pixels_vu_batch, reso_level=resolution_level)
            scores_gt, diff_patch_all_gt, mask_valid_all_gt = self.dataset.score_pixels_ncc(idx, pts_peak, normals_gt_batch, idx_pixels_vu_batch, reso_level=resolution_level)
            
            scores_render_all.append(scores_render.cpu().numpy())
            scores_gt_all.append(scores_gt.cpu().numpy())
        scores_render_all = np.concatenate(scores_render_all, axis=0).reshape([H, W])
        scores_gt_all = np.concatenate(scores_gt_all, axis=0).reshape([H, W])

        mask_filter =( (scores_gt_all -0.01) < scores_render_all)

        img_gr=np.zeros((H, W, 3), dtype=np.uint8)
        img_gr[:,:, 0] = 255 
        img_gr[mask_filter==False] = (0,0,255)

        img_rgb =np.zeros((H, W, 3), dtype=np.uint8) 
        img_rgb[mask_filter==False] = self.dataset.images_np[idx][mask_filter==False]*255

        ImageUtils.write_image_lis(f'./test/sampling/rgb_{idx:04d}.png', [self.dataset.images_np[idx]*256, mask_filter*255, img_gr, img_rgb])

    def validate_mesh(self, world_space=False, resolution=128, threshold=0.0):
        bound_min = torch.tensor(self.dataset.bbox_min, dtype=torch.float32) #/ self.sdf_network_fine.scale
        bound_max = torch.tensor(self.dataset.bbox_max, dtype=torch.float32) # / self.sdf_network_fine.scale
        vertices, triangles, sdf = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        path_mesh = os.path.join(self.base_exp_dir, 'meshes', f'{self.iter_step:0>8d}_reso{resolution}_{self.scan_name}.ply')
        path_mesh_gt = IOUtils.find_target_file(self.dataset.data_dir, '_vh_clean_2_trans.ply')

        if world_space:
            path_trans_n2w = f"{self.conf['dataset']['data_dir']}/trans_n2w.txt"
            scale_mat = self.dataset.scale_mats_np[0]
            if IOUtils.checkExistence(path_trans_n2w):
                scale_mat = np.loadtxt(path_trans_n2w)
            vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
            
            path_mesh = os.path.join(self.base_exp_dir, 'meshes', f'{self.iter_step:0>8d}_reso{resolution}_{self.scan_name}_world.ply')
            path_mesh_gt = IOUtils.find_target_file(self.dataset.data_dir, '_vh_clean_2.ply')

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(path_mesh)
    
        if path_mesh_gt:
            GeoUtils.clean_mesh_points_outside_bbox(path_mesh, path_mesh, path_mesh_gt, scale_bbox = 1.1, check_existence=False)

        return path_mesh

    def validate_fields(self, iter_step=-1):
        if iter_step < 0:
            iter_step = self.iter_step
        bound_min = torch.tensor(self.dataset.bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.bbox_max, dtype=torch.float32)
        sdf = extract_fields(bound_min, bound_max, 128, lambda pts: self.sdf_network_fine.sdf(pts)[:, 0])
        os.makedirs(os.path.join(self.base_exp_dir, 'fields'), exist_ok=True)
        np.save(os.path.join(self.base_exp_dir, 'fields', '{:0>8d}_sdf.npy'.format(iter_step)), sdf)

    def validate_points(self, rays_o, rays_d, near, far):
        logging.info(f"[{self.iter_step}]: validate sampled points...")
        sample_dist = self.radius_norm / self.n_samples

        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)

        mid_z_vals = z_vals + dists * 0.5
        mid_dists = mid_z_vals[..., 1:] - mid_z_vals[..., :-1]

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3

        dir_pts = f"{self.base_exp_dir}/points"
        os.makedirs(dir_pts, exist_ok=True)

        path_pts = f"{dir_pts}/{self.iter_step:08d}_{self.curr_img_idx}.ply"
        pts = pts.reshape(-1, 3)
        GeoUtils.save_points(path_pts, pts.detach().cpu().numpy())

    def validate_weights(self, weight):
        dir_weights = os.path.join(self.base_exp_dir, "weights")
        os.makedirs(dir_weights, exist_ok=True)

    def validate_image_nerf(self, idx=-1, resolution_level=-1, save_render = False):
        print('Validate:')
        ic(self.iter_step, idx)
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = torch.zeros(len(rays_o_batch), 1), 1 * torch.ones(len(rays_o_batch), 1)
            if self.dataset_type == 'dtu':
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            elif self.dataset_type == 'indoor':
                near, far = torch.zeros(len(rays_o_batch), 1), 1 * torch.ones(len(rays_o_batch), 1)
            else:
                NotImplementedError

            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              background_rgb=background_rgb)

            def feasible(key): return ((key in render_out) and (render_out[key] is not None))

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'image_render'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        f'{self.iter_step:08d}_{self.dataset.vec_stem_files[idx]}_reso{resolution_level}.png'), # '{:0>8d}_{}_{}.png'.format(self.iter_step, i, self.dataset.vec_stem_files[idx])),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
                if save_render:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'image_render',
                                            f'{self.iter_step:08d}_{self.dataset.vec_stem_files[idx]}_reso{resolution_level}.png'), # '{:0>8d}_{}_{}.png'.format(self.iter_step, i, self.dataset.vec_stem_files[idx])),
                            img_fine[..., i])

    def validate_mesh_nerf(self, world_space=False, resolution=128, threshold=25.0):
        logging.info(f'Validate mesh (Resolution:{resolution}； Threhold: {threshold})....')
        bound_min = torch.tensor(self.dataset.bbox_min, dtype=torch.float32) #/ self.sdf_network_fine.scale
        bound_max = torch.tensor(self.dataset.bbox_max, dtype=torch.float32) # / self.sdf_network_fine.scale

        vertices, triangles, sdf =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        
        os.makedirs(os.path.join(self.base_exp_dir, 'contour'), exist_ok=True)
        for ax in ['x', 'y', 'z']:
            path_contour_img = os.path.join(self.base_exp_dir, 'contour', f'{ax}_{self.iter_step:0>8d}_reso{resolution}.png')
            if ax == 'x':
                GeoUtils.visualize_sdf_contour(path_contour_img, -sdf[int(sdf.shape[0]/2), :,:], levels=[i-50 for i in range(0, 100, 10)])
            if ax == 'y':
                GeoUtils.visualize_sdf_contour(path_contour_img, -sdf[:, int(sdf.shape[1]/2),:], levels=[i-50 for i in range(0, 100, 10)])
            if ax == 'z':
                GeoUtils.visualize_sdf_contour(path_contour_img, -sdf[:,:, int(sdf.shape[2]/2)], levels=[i-50 for i in range(0, 100, 10)])

        path_mesh_gt = IOUtils.find_target_file(self.dataset.data_dir, '_vh_clean_2_trans.ply')

        if world_space:
            path_trans_n2w = f"{self.conf['dataset']['data_dir']}/trans_n2w.txt"
            print(f'LOad normalization matrix: {path_trans_n2w}')
            scale_mat = self.dataset.scale_mats_np[0]
            if IOUtils.checkExistence(path_trans_n2w):
                scale_mat = np.loadtxt(path_trans_n2w)
            vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
            
            path_mesh = os.path.join(self.base_exp_dir, 'meshes', f'{self.iter_step:0>8d}_reso{resolution}_thres{int(threshold):03d}_{self.scan_name}_world.ply')
            path_mesh_gt = IOUtils.find_target_file(self.dataset.data_dir, '_vh_clean_2.ply')

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(path_mesh)

        path_mesh_clean = IOUtils.add_file_name_suffix(path_mesh, '_clean')
        if path_mesh_gt:
            print(f'Clena points outside bbox')
            GeoUtils.clean_mesh_points_outside_bbox(path_mesh_clean, path_mesh, path_mesh_gt, scale_bbox = 1.1, check_existence=False)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(path_mesh_clean)

    def save_accumulated_results(self, idx=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        logging.info(f'Save confidence: idx {idx}...')
              
        accum_results = {}
        accum_results['confidence'] = self.dataset.confidence_accum[idx]

        b_accum_all_data = False
        if b_accum_all_data:
            accum_results['color'] = self.dataset.colors_accum[idx].cpu().numpy() * 255
            # accum_results['normal'] = self.dataset.normals_accum[idx].cpu().numpy()
            accum_results['depth'] = self.dataset.depths_accum[idx].cpu().numpy()

        img_gt = self.dataset.images[idx].cpu().numpy()
        lis_imgs = []
        for key in accum_results:
            if len(accum_results[key]) == 0:
                continue

            img_temp = accum_results[key].squeeze()
            if key == 'normal':
                img_temp = (((img_temp + 1) * 0.5).clip(0,1) * 255).astype(np.uint8)
            if key in ['depth']:
                img_temp = img_temp / (np.max(img_temp)+1e-6) * 255
            if key in ['confidence', 'samples']:
                if key == 'confidence':
                    img_temp2 = ImageUtils.convert_gray_to_cmap(img_temp.clip(0,1), vmax=1.0)
                    lis_imgs.append(img_temp2)
                    img_temp = img_temp.clip(0,1) * 255
                if key == 'samples':
                    img_temp = ImageUtils.convert_gray_to_cmap(img_temp)
            cv.putText(img_temp, key,  (img_temp.shape[1]-100, img_temp.shape[0]-20), 
                fontFace= cv.FONT_HERSHEY_SIMPLEX, 
                fontScale = 0.5, 
                color = (0, 0, 255), 
                thickness = 2)

            accum_results[key] = img_temp
            lis_imgs.append(img_temp)

        # save images
        dir_accum = os.path.join(self.base_exp_dir, 'images_accum')
        os.makedirs(dir_accum, exist_ok=True)
        ImageUtils.write_image_lis(f'{dir_accum}/{self.iter_step:0>8d}_{idx}.png', 
                                        [img_gt] + lis_imgs)

        if self.iter_step ==  self.end_iter:
            dir_accum_npz = f'{self.base_exp_dir}/checkpoints/npz'
            os.makedirs(dir_accum_npz, exist_ok=True)
            np.savez(f'{dir_accum_npz}/confidence_accum_{self.iter_step:08d}.npz', self.dataset.confidence_accum)
            np.savez(f'{dir_accum_npz}/samples_accum_{self.iter_step:08d}.npz', self.dataset.samples_accum.cpu().numpy())

            b_accum_all_data = False
            if b_accum_all_data:
                np.savez(f'{dir_accum_npz}/colors_accum_{self.iter_step:08d}.npz', self.dataset.colors_accum.cpu().numpy())
                np.savez(f'{dir_accum_npz}/depths_accum_{self.iter_step:08d}.npz', self.dataset.depths_accum.cpu().numpy())
                

    def initialize_accumulated_results(self, mode = 'MODEL', iter_step_npz = None, resolution_level = 2):
        # For cache rendered depths and normals
        self.dataset.confidence_accum = np.ones_like(self.dataset.masks_np, dtype=np.float32)
        
        b_accum_all_data = False
        if b_accum_all_data:
            self.dataset.colors_accum = torch.zeros_like(self.dataset.images, dtype=torch.float32).cuda()
            self.depths_accum = torch.zeros_like(self.masks, dtype=torch.float32).cpu()

        self.dataset.normals_accum = torch.zeros_like(self.dataset.images, dtype=torch.float32).cpu()  # save gpu memory
        self.dataset.points_accum = torch.zeros_like(self.dataset.images, dtype=torch.float32).cpu()  # world coordinates

        if mode == 'model':
            # compute accumulated results from pretrained model
            logging.info(f'Start initializeing accumulated results....[mode: {mode}; Reso: {resolution_level}]')
            self.dataset.render_difference_accum = np.ones_like(self.dataset.masks_np, dtype=np.float32)

            t1 = datetime.now()
            num_neighbors_half = 0
            for idx in tqdm(range(num_neighbors_half, self.dataset.n_images-num_neighbors_half)):
                confidence, color_peak, normal_peak, depth_peak, points_peak, _ = self.validate_image(idx, resolution_level=resolution_level, save_peak_value=True)
                H, W = confidence.shape
                resize_arr = lambda in_arr : cv.resize(in_arr, (W*resolution_level, H*resolution_level), interpolation=cv.INTER_NEAREST) if resolution_level > 1 else in_arr
                self.dataset.confidence_accum[idx] = resize_arr(confidence)
                
                b_accum_all_data = False
                if b_accum_all_data:
                    self.dataset.colors_accum[idx]     = torch.from_numpy(resize_arr(color_peak)).cuda()
            logging.info(f'Consumed time: {IOUtils.get_consumed_time(t1)/60:.02f} min.') 
            
            dir_accum_npz = f'{self.base_exp_dir}/checkpoints/npz'
            os.makedirs(dir_accum_npz, exist_ok=True)
            np.savez(f'{dir_accum_npz}/confidence_accum_{self.iter_step:08d}.npz', self.dataset.confidence_accum)
            
        elif mode == 'npz':
            # load accumulated results from files
            logging.info(f'Start initializeing accumulated results....[mode: {mode}; Reso: {resolution_level}]')
            iter_step_npz = int(iter_step_npz)
            assert iter_step_npz is not None
            dir_accum_npz = f'{self.base_exp_dir}/checkpoints/npz'
            
            self.dataset.confidence_accum = np.load(f'{dir_accum_npz}/confidence_accum_{int(iter_step_npz):08d}.npz')['arr_0']

            b_accum_all_data = False
            if b_accum_all_data:
                self.dataset.colors_accum =     torch.from_numpy(np.load(f'{dir_accum_npz}/colors_accum_{iter_step_npz:08d}.npz')['arr_0']).cuda()
                self.dataset.depths_accum =     torch.from_numpy(np.load(f'{dir_accum_npz}/depths_accum_{iter_step_npz:08d}.npz')['arr_0']).cuda()
        else:
            logging.info('Initialize all accum data to ones.')

            
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint_id', type=int, default=-1)
    parser.add_argument('--mc_reso', type=int, default=512, help='Marching cube resolution')
    parser.add_argument('--reset_var', action= 'store_true', help='Reset variance for validate_iamge()' )
    parser.add_argument('--nvs', action= 'store_true', help='Novel view synthesis' )
    parser.add_argument('--save_render_peak', action= 'store_true', help='Novel view synthesis' )
    parser.add_argument('--scene_name', type=str, default='', help='Scene or scan name')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.scene_name, args.mode, args.model_type, args.is_continue, args.checkpoint_id)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        if runner.model_type == 'neus':
            t1= datetime.now()
            runner.validate_mesh(world_space=True, resolution=args.mc_reso, threshold=args.threshold)
            logging.info(f"[Validate mesh] Consumed time (Step: {runner.iter_step}; MC resolution: {args.mc_reso}): {IOUtils.get_consumed_time(t1):.02f}(s)")

        elif runner.model_type == 'nerf':
            thres = args.threshold
            t1= datetime.now()
            runner.validate_mesh_nerf(world_space=True, resolution=args.mc_reso, threshold=thres)
            logging.info(f"[Validate mesh] Consumed time (MC resolution: {args.mc_reso}； Threshold: {thres}): {IOUtils.get_consumed_time(t1):.02f}(s)")
    
    elif args.mode.startswith('validate_image'):
        if runner.model_type == 'neus':
            for i in range(0, runner.dataset.n_images, 2):
                t1 = datetime.now()
                runner.validate_image(i, resolution_level=1, 
                                            save_normalmap_npz=args.save_render_peak, 
                                            save_peak_value=True,
                                            save_image_render=args.nvs)
                logging.info(f"validating image time is : {(datetime.now()-t1).total_seconds()}")
        
        elif runner.model_type == 'nerf':
            for i in range(0, runner.dataset.n_images, 2):
                t1 = datetime.now()
                runner.validate_image_nerf(i, resolution_level=1, save_render=True)
                logging.info(f"validating image time is : {(datetime.now()-t1).total_seconds()}")
            
    elif args.mode == 'validate_fields':
        runner.validate_fields()