import cv2, logging, copy, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import utils.utils_io as IOUtils
import torch.nn.functional as F

def prepare_patches_src(img_gray, indices_pixel, window_size=11, window_step=2):
    '''
    Args:
        img: H*W, gray image
        indices_pixel: N*2
        size_window: int, patch size
        size_step: int, interval sampling
    Return:
        rgb_pathes: N*M*M, gray values of images
    '''


    assert img_gray.dtype == torch.float32
        
    size_img = img_gray.shape[:2][::-1]

    # get kernel indices of a patch
    window_size+=1
    x = torch.arange(-(window_size//2), window_size//2 + 1, window_step) # weired -1//2=-1
    xv, yv = torch.meshgrid(x,x)
    kernel_patch = torch.stack([xv,yv], axis=-1) # M*M*2 
    # print(kernel_patch)
    num_pixels = len(indices_pixel)
    indices_patch = indices_pixel.reshape(num_pixels,1,1,2) + kernel_patch
    
    # sample img
    mask_indices_inside = is_inside_border(indices_patch.reshape(-1,2), size_img).reshape(indices_patch.shape[:3]) # N*M*M
    indices_patch_inside = (indices_patch * mask_indices_inside[...,None]).reshape(-1,2)   # N*2. let the indices of outside pixels be zero
    
    if img_gray.ndim ==2:
        rgb_patches = img_gray[indices_patch_inside[:, 0], indices_patch_inside[:, 1]].reshape(indices_patch.shape[:3]) # N*M*M
    elif img_gray.ndim ==3:
        rgb_patches = img_gray[indices_patch_inside[:, 0], indices_patch_inside[:, 1]].reshape(indices_patch.shape[:3] + tuple([3])) # N*M*M
    else:
        raise NotImplementedError

    rgb_patches[mask_indices_inside==False] = -1
    return rgb_patches.cuda(), indices_patch, mask_indices_inside

def normalize_coords_vu(idx_vu, shape_img):
    '''Normalize coords to interval [-1, 1]
    Args:
        idx_vu: shape [..., 2]
    '''
    assert idx_vu.dtype == torch.float32
    H, W = shape_img

    idx_vu_norm = idx_vu.clone().detach()
    temp = 2 * idx_vu[...,0]  / (H - 1) - 1
    
    # convert vu to xy
    idx_vu_norm[..., 1] = 2 * idx_vu[...,0]  / (H - 1) - 1
    idx_vu_norm[..., 0] = 2 * idx_vu[...,1]  / (W - 1) - 1
    return idx_vu_norm

def sample_patches(img, idx_patches_input, sampling_mode = 'nearest'): 
    '''
    Args:
        img: W*H or W*H*3
        idx_patches_warp: N*M*M*2, warped indices
    Return:
        rgb_patches: N*M*M, sampled pixels
    '''
    # sample img
    size_img = img.shape[:2][::-1]
    shape_patches = idx_patches_input.shape

    # TODO: Add inter-linear interploation of sampling
    if idx_patches_input.dtype not in [torch.int, torch.int16, torch.int32, torch.int64]:
        idx_patches_input_round = torch.round(idx_patches_input).long()
        idx_patches = idx_patches_input_round
    else:
        # logging.info(f'Grid sampling')
        idx_patches = idx_patches_input
        sampling_mode = 'nearest'
    
    mask_indices_inside = is_inside_border(idx_patches.reshape(-1,2), size_img).reshape(idx_patches.shape[:-1]) # N*M*M
    indices_patch_inside = (idx_patches * mask_indices_inside[...,None]).reshape(-1,2)   # N*2. let the indices of outside pixels be zero
    
    # grid sampling
    if sampling_mode == 'grid_sample':
        assert img.ndim == 2  # 1*1*H*W
        idx_patches_input_norm = normalize_coords_vu(idx_patches_input, img.shape).clip(-1,1)
        rgb_patches = F.grid_sample(img[None, None,:,:], idx_patches_input_norm.reshape(1,1,-1,2), align_corners=False).squeeze()
        rgb_patches = rgb_patches.reshape(shape_patches[:3])
    elif sampling_mode == 'nearest':
        if img.ndim ==2:
            rgb_patches = img[indices_patch_inside[:, 0], indices_patch_inside[:, 1]].reshape(idx_patches.shape[:-1]) # N*M*M
        elif img.ndim ==3:
            rgb_patches = img[indices_patch_inside[:, 0], indices_patch_inside[:, 1]].reshape(idx_patches.shape[:-1] + tuple([3])) # N*M*M
        else:
            raise NotImplementedError

    rgb_patches[mask_indices_inside==False] = -1
    return rgb_patches

def is_inside_border(ind_pixels, size_img):
    '''
    Args:
        ind_pixel: N*2
        size_img: (W,H)
    Return:
        mask: N*1, bool
    '''
    assert (ind_pixels.ndim ==2 & ind_pixels.shape[1] ==2)
    W, H = size_img
    return (ind_pixels[:, 0] >= 0) & (ind_pixels[:,0] < H) & \
           (ind_pixels[:, 1] >= 0) & (ind_pixels[:,1] < W)

def compute_NCC_score(patches_ref, patches_src):
    '''
    Args:
        pathces_ref: N*M*M
        patches_src: N*M*M
    Return:
        score: float
    '''
    num_patches = patches_ref.shape[0]
    size_patch = (patches_ref.shape[1] * patches_ref.shape[2])

    # ensure pixels inside border
    mask_inside_ref, mask_inside_src = patches_ref >= 0, patches_src >= 0
    mask_inside = mask_inside_ref & mask_inside_src # different for diffenert neighbors
    
    # ensure half patch inside border
    mask_patches_valid = (mask_inside.reshape(num_patches, -1).sum(axis=-1) == size_patch)[...,None,None]
    mask_inside = mask_inside & mask_patches_valid
    # logging.info(f'Pixels inside ref and src: {mask_inside.sum()}/{mask_inside.size}; Ref: {mask_inside_ref.sum()}/{mask_inside_ref.size}; Src: {mask_inside_src.sum()}/{mask_inside_src.size}')

    calculate_patches_mean = lambda patches, mask: (patches*mask).reshape(patches.shape[0], -1).sum(axis=1) / (mask.reshape(patches.shape[0], -1).sum(axis=1)+1e-6)
    mean_patches_ref = calculate_patches_mean(patches_ref, mask_inside).reshape(-1,1,1)
    mean_patches_src = calculate_patches_mean(patches_src, mask_inside).reshape(-1,1,1)
    
    normalized_patches_ref = patches_ref - mean_patches_ref
    normalized_patches_src = patches_src - mean_patches_src
    norm = ((normalized_patches_ref * normalized_patches_src)*mask_inside).reshape(num_patches,-1).sum(axis=-1)
    denom = torch.sqrt( 
                (torch.square(normalized_patches_ref)*mask_inside).reshape(num_patches,-1).sum(axis=-1) * \
                (torch.square(normalized_patches_src)*mask_inside).reshape(num_patches,-1).sum(axis=-1)
            )
    norm = norm.clip(10, 1e6)
    ncc = (norm + 20) / (denom+1e-3)
    if (denom == 0).sum() > 0:
        # check quadratic moments to avoid degeneration.
        # pathes with same values 
        # [Nonlinear systems], https://en.wikipedia.org/wiki/Cross-correlation
        mask_degenerade = ( denom == 0)
        ncc[denom == 0] = 1.0
        # logging.info(f'Deneraded patches: {mask_degenerade.sum()}/{mask_degenerade.size}.')  # pixels with invalid patchmatch
    score = 1-ncc.clip(-1.0,1.0) # 0->2: smaller, better
    diff_mean = torch.abs(mean_patches_ref - mean_patches_src).squeeze()
    return score, diff_mean, mask_patches_valid.squeeze()

def update_confmap(confmap, idx_pixels, ncc_score, idx_patches = None):
    '''
    Args:
        confmap: H*W
        idx_pixels: N*2, numpy array index
        ncc_score: N*1,
        idx_patches: None. (optional: N*M*M*2) 
    Return:
        updated_confmap: H*W
    '''
    # ncc_score = ncc_score[...,None, None]* np.ones_like(mask_inside_ref)
     
    # indices_inside = indices_patch[mask_inside]
    # score_inside = ncc_score[mask_inside]
    confmap = copy.deepcopy(confmap)
    prev_conf = confmap[idx_pixels[:,0], idx_pixels[:,1]]
    confmap[idx_pixels[:,0], idx_pixels[:,1]] = ncc_score  # TODO: check duplicated values here
    return confmap
    
def prepare_neigbor_pixels(img, indices_pixel):
    '''Get 4 neighbor indices of given pixels
    Args:
        img: H*W, ndarray
        indices_pixel: N*2
    Return:
        indices_neighbors: N*4*2
    '''
    raise NotImplementedError

def sample_pixels_by_probability(prob_pixels_all, batch_size, clip_range = (0.22, 0.66), shuffle = True):
    '''Sample pixels based on NCC confidence. Sample more pixels on low confidence areas
    Args:
        scores_ncc: W*H, narray
    Return:
        uv_sample: 2*N
    '''
    # Guarantee that every pixels can be sampled
    prob = prob_pixels_all.clip(*clip_range) 

    H,W = prob.shape
    sample_range = W * H
    prob = prob / prob.sum()
    
    samples_indices = np.random.choice(sample_range, batch_size, p=prob.reshape(-1))
    pixels_v = samples_indices // W
    pixels_u = samples_indices % W

    uv_sample = np.stack((pixels_u, pixels_v), axis=-1) #.transpose()
    if shuffle:
        # uv_sample = np.copy(uv_sample)
        np.random.shuffle(uv_sample)
    uv_sample = uv_sample.transpose()
    return uv_sample

def visualize_sampled_pixels(img_input, indices_pixel, path = None):
    '''
    Two different format of coordinates:
        (1) indices_pixel: in vu format
        (2) uvs_pixel: in opencv format
    Args:
        indices_pixel: N*2
    '''
    img = copy.deepcopy(img_input)
    for i in range(indices_pixel.shape[0]):
        cv2.circle(img, (indices_pixel[i, 1], indices_pixel[i,0]), 
                        radius = 2, 
                        color = (255,0,0), # BGR
                        thickness=2)  
        cv2.putText(img, f'{i}',  (indices_pixel[i, 1] + 2 if indices_pixel[i, 1] + 2 < img.shape[1] else img.shape[1]-1, indices_pixel[i,0]), 
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.75, 
                        color = (0, 0, 255), 
                        thickness = 1) # int
    if path is not None:
        cv2.imwrite(path, img)
    return img

def visualize_samples_lis(img_input, lis_samples_uv, path = None):
    '''
    Two different format of coordinates:
        (1) indices_pixel: in numpy format
        (2) uvs_pixel: in opencv format
    Args:
        lis_samples: M*N*2
    '''
    img = copy.deepcopy(img_input)
    lis_colors = [(0,0,255), (255,0,0),(0,255,0)] # R, G, B, in cv color mode

    assert len(lis_samples_uv) <= len(lis_colors)
    for idx_samples in range(len(lis_samples_uv)):
        samples_curr = lis_samples_uv[idx_samples]
        for i in range(samples_curr.shape[0]):
            cv2.circle(img, (samples_curr[i, 0], samples_curr[i,1]), 
                            radius = 2, 
                            color = lis_colors[idx_samples], # BGR
                            thickness=2)  
        #     cv2.putText(img, f'{i}',  (indices_pixel[i, 1] + 2 if indices_pixel[i, 1] + 2 < img.shape[1] else img.shape[1]-1, indices_pixel[i,0]), 
        #                     fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
        #                     fontScale = 0.75, 
        #                     color = (0, 0, 255), 
        #                     thickness = 2) # int
    if path is not None:
        dir, _, _ = IOUtils.get_path_components(path)
        IOUtils.ensure_dir_existence(dir)
        cv2.imwrite(path, img)
    return img

def save_patches(path, patches, interval = 2, color_mode = 'GRAY'):
    num_patches, w_p, h_p = patches.shape[:3]
    w_img = int( np.ceil(np.sqrt(num_patches)) * (w_p+interval) )
    h_img = int( np.ceil(np.sqrt(num_patches)) * (h_p+ interval) )

    img = np.zeros((h_img, w_img))
    if patches.shape[-1] ==3:
        img = np.zeros((h_img, w_img, 3))
    patch_num_row = int(np.ceil(np.sqrt(num_patches)))
    for i in range(patch_num_row):
        for j in range(patch_num_row):
            idx_patch = i*patch_num_row+j
            if idx_patch >= num_patches:
                continue
            img[i*(w_p+interval):(i+1)*(w_p+interval)-interval,j*(h_p+interval):(j+1)*(h_p+interval)-interval] = patches[idx_patch]
    cv2.imwrite(path, img)    
    return img

def concatenate_images(img1, img2, path = None):
    check_channel = lambda img : np.stack([img]*3, axis=-1) if img.ndim ==2 else img
    img1 = check_channel(img1)
    img2 = check_channel(img2)
    img_cat = np.concatenate([img1, img2], axis=0)
    if path is not None:
        cv2.imwrite(path, img_cat)
    return img_cat

def compute_homography(pt_ref, n_ref, K, extrin_ref, extrin_src):
    '''Compute homography from reference view to source view.
    Tranform from world to reference view coordinates.

    Args:
        n: N*3*1, reference view normal
        pt: N*3*1, reference view coordinates of a point

        K: 3*3, intrinsics
        extrin_ref: N*4*4
        extrin_src: N*4*4
    Return:
        mat_homography: N*3*3 
    '''
    # TODO: check this method later
    if False:
        def decompose_extrin(extrin):
            rot = extrin[:3,:3]
            trans = extrin[:3,3].reshape(3,1)
            cam_center = - np.linalg.inv(rot) @ trans
            return rot, cam_center
        
        R_ref, C_ref = decompose_extrin(extrin_ref)
        R_src, C_src = decompose_extrin(extrin_src)

        # rt = R_ref.transpose()
        Hl = K @ R_src @ R_ref.transpose()
        Hm = K @ R_src @ (C_ref - C_src)
        Hr = np.linalg.inv(K)

        num_pts = len(pt)
        denom = (n * pt).sum(axis=-1, keepdims=True)[0].reshape(1, 1,1)
        mat_homography = ( Hl[None, ...] + Hm[None, ...] @ (n.reshape(num_pts, 1,3) / denom) ) *Hr[None, ...];

    ref_pose = torch.linalg.inv(extrin_ref)
    inv_src_pose = extrin_src
    inv_ref_pose = extrin_ref

    relative_proj = inv_src_pose @ ref_pose
    R_rel = relative_proj[:3, :3]
    t_rel = relative_proj[:3, 3:]
    R_ref = inv_ref_pose[:3, :3]
    t_ref = inv_ref_pose[:3, 3:]

    num_pts = len(pt_ref)
    d = (n_ref * pt_ref).sum(axis=-1, keepdims=True).reshape(num_pts, 1,1)
    mat_homography = (K[None, ...] @ (R_rel[None, ...] + t_rel[None,...] @ n_ref.reshape(num_pts,1,3) / d))@torch.linalg.inv(K)
    return mat_homography

def warp_patches(idx_patches_xy, homography):
    '''Warp patches from reference image to source images
    Args:
        idx_patches: N*M*M*2, xy index coordinates (numpy array)
        homography: N*3*3, from reference to source
    Return:
        idx_patches_warp: N*M*M*2, xy index coordinates (numpy array)
    '''
    shape = idx_patches_xy.shape
    idx_patches = convert_xy_to_uv(idx_patches_xy)
    idx_patches_homo = convert_to_homo(idx_patches)
    num_patches, H_patch, W_patch = homography.shape
    
    if idx_patches_xy.ndim == 2:    #N*2
        idx_patches_warp = (homography @ idx_patches_homo[...,None]).squeeze() # N*3*1
    elif idx_patches_xy.ndim == 4:  #N*M*M*2
        idx_patches_warp = (homography.reshape(num_patches, 1, 1, H_patch, W_patch) @ idx_patches_homo[...,None]).squeeze() # N*M*M*3*1
    else:
        raise NotImplementedError

    idx_patches_warp = (idx_patches_warp / idx_patches_warp[..., 2][...,None])[...,:2]

    # TODO: update later fpr faster calculation. Refer to NeuralWarp
    # einsum is 30 times faster
    # tmp = (H.view(Nsrc, N, -1, 1, 3, 3) @ hom_uv.view(1, N, 1, -1, 3, 1)).squeeze(-1).view(Nsrc, -1, 3)
    # tmp = torch.einsum("vprik,pok->vproi", H, hom_uv).reshape(Nsrc, -1, 3)
    # patches_warp = patches_warp / patches_warp[..., 2:].clip(min=1e-8)

    # TODO: use the stragety of openMVS to simplify calculatoin. O(N*h^2)->O(H+h^2)
    # homography_step = homography*window_step
    # kernel_homography = 
    idx_patches_warp_xy = convert_uv_to_xy(idx_patches_warp)
    return idx_patches_warp_xy

def denoise_image(img):
    b,g,r = cv2.split(img)           # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb

    # img = cv2.fastNlMeansDenoisingColored(img.astype(np.uint8),None,10,10,7,21)

    # b,g,r = cv2.split(dst)           # get b,g,r
    # rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
    return img

def convert_uv_to_xy(coord_uv):
    '''
    Args:
        coord: N*2
    '''
    coord_xy = torch.stack([coord_uv[...,1], coord_uv[...,0]], axis=-1)
    return coord_xy

def convert_xy_to_uv(coord_xy):
    '''
    Args:
        coord: N*2
    '''
    coord_uv = torch.stack([coord_xy[...,1], coord_xy[...,0]], axis=-1)
    return coord_uv

def convert_to_homo(pts):
    pts_homo = torch.cat([pts, torch.ones(pts.shape[:-1] + tuple([1])) ], dim=-1)
    return pts_homo

