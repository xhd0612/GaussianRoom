import matplotlib.pyplot as plt
import torch
import glob, os
import cv2
import logging
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import copy

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from tqdm import tqdm

import utils.utils_io as IOUtils

DIR_FILE = os.path.abspath(os.path.dirname(__file__))

def calculate_normal_angle(normal1, normal2, use_degree = False):
    '''Get angle to two vectors
    Args:
        normal1: N*3
        normal2: N*3
    Return:
        angles: N*1
    '''
    check_dim = lambda normal : np.expand_dims(normal, axis=0) if normal.ndim == 1 else normal
    normal1 = check_dim(normal1)
    normal2 = check_dim(normal2)

    inner = (normal1*normal2).sum(axis=-1)
    norm1 = np.linalg.norm(normal1, axis=1, ord=2)
    norm2 = np.linalg.norm(normal2, axis=1, ord=2)
    angles_cos = inner / (norm1*norm2 + 1e-6)
    angles = np.arccos(angles_cos)
    if use_degree:
        angles = angles/np.pi * 180

    assert not np.isnan(angles).any()
    return angles

def read_image(path, target_img_size = None, interpolation=cv2.INTER_LINEAR, color_space = 'BGR'):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if target_img_size is not None:
        img= resize_image(img, target_img_size, interpolation=interpolation)
    if color_space == 'RGB':
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img

def write_image(path, img, color_space = None, 
                    target_img_size = None, interpolation = cv2.INTER_LINEAR):
    '''If color space is defined, convert colors to RGB mode
    Args:
        target_img_size: resize image if defined
    
    '''
    if color_space == 'RGB':
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if target_img_size is not None:
        img = resize_image(img, target_size=target_img_size, interpolation=interpolation)
    cv2.imwrite(path, img)

def write_images(dir, imgs, stems = None, ext_img = '.png', color_space = None):
    IOUtils.ensure_dir_existence(dir)
    for i in range(len(imgs)):
        if stems is None:
            path = f'{dir}/{i:04d}{ext_img}'
        else:
            path = f'{dir}/{stems[i]}{ext_img}'
        write_image(path, imgs[i], color_space)

def read_images(dir, target_img_size = None, interpolation=cv2.INTER_LINEAR, img_ext = '.png', use_rgb_mode = False, vec_stems = None):
    f'''Read images in directory with extrension {img_ext}
    Args:
        dir: directory of images
        target_img_size: if not none, resize read images to target size
        img_ext: defaut {img_ext}
        use_rgb_mode: convert brg to rgb if true
    Return:
        imgs: N*W*H
        img_stems
    '''
    if img_ext == '.npy':
        read_img = lambda path : np.load(path)
    elif img_ext == '.npz':
        read_img = lambda path : np.load(path)['arr_0']
    elif img_ext in ['.png', '.jpg']:
        read_img = lambda path : read_image(path)
    else:
        raise NotImplementedError

    vec_path = sorted(glob.glob(f"{dir}/**{img_ext}"))
    if vec_stems is not None:
        vec_path = []
        for stem_curr in vec_stems:
            vec_path.append(f'{dir}/{stem_curr}{img_ext}')
        vec_path = sorted(vec_path)

    rgbs = []
    img_stems = []
    for i in range(len(vec_path)):
        img = read_img(vec_path[i])
        if (target_img_size != None) and (target_img_size[0] != img.shape[1]):
            img= cv2.resize(img, target_img_size, interpolation=cv2.INTER_LINEAR)
        if use_rgb_mode:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgbs.append(img)

        _, stem, _ = IOUtils.get_path_components(vec_path[i])
        img_stems.append(stem)
    return np.array(rgbs), img_stems

def get_planes_from_normalmap(dir_pred, thres_uncertain = 0):
    '''Get normals from normal map for indoor dataset (ScanNet), which was got by the following paper:
    Surface normal estimation with uncertainty
    '''
    dir_normals = os.path.join(dir_pred, 'pred_normal')
    vec_path_imgs = sorted(glob.glob(f'{dir_normals}/**.png'))
    num_images = len(vec_path_imgs)
    assert num_images > 0
    logging.info(f'Found images: {num_images}')

    dir_mask_labels = os.path.join(dir_normals, '../pred_normal_planes')
    dir_mask_labels_rgb = os.path.join(dir_normals, '../pred_normal_planes_rgb')
    os.makedirs(dir_mask_labels, exist_ok=True)
    os.makedirs(dir_mask_labels_rgb, exist_ok=True)

    vec_path_kappa =  sorted(glob.glob(f'{dir_normals}/../pred_kappa/**.png'))
    dir_normals_certain = f'{dir_normals}/../pred_normal_certain'
    os.makedirs(dir_normals_certain, exist_ok=True)

    channel_threshold = 200
    channel_threshold_curr = channel_threshold

    for j in tqdm(range(num_images)):
        path = vec_path_imgs[j]
        _, stem, ext = IOUtils.get_path_components(path)

        img = read_image(path)
        
        if thres_uncertain > 0:
            img_kappa = read_image(vec_path_kappa[j])
            mask_uncertain = img_kappa < thres_uncertain
            img[mask_uncertain] = 0
            write_image(f'{dir_normals_certain}/{stem}{ext}', img)
        
        img_masks = []
        imgs_lables = np.zeros((img.shape[0], img.shape[1]))
        for i in range(3):
            ch = img[:,:, i]

            ch_mask = ch > channel_threshold
            test = ch_mask.sum()
            while ch_mask.sum() == 0:
                channel_threshold_curr -= 10
                ch_mask = ch > channel_threshold_curr
            channel_threshold_curr = channel_threshold
            
            if i==2:
                ch_mask = img[:,:, 0] > 0
                if ch_mask.sum() ==0:
                    ch_mask = img[:,:, 1] > 0

            ch_arr = ch[ch_mask]
            count_arr = np.bincount(ch[ch_mask])
            ch_value_most = np.argmax(count_arr)
            logging.info(f"[{j}-{i}] Channel value most: {ch_value_most}")

            # ch_value_most = np.max(ch)
            sample_range_half = 5
            range_min = ch_value_most - sample_range_half if ch_value_most > sample_range_half else 0
            range_max = ch_value_most + sample_range_half if ch_value_most < (255-sample_range_half) else 255
            logging.debug(f'Sample range: [{range_min}, {range_max}]')

            # ToDO: check other channels

            ch_mask = (ch > range_min) & (ch < range_max)
            ch = ch * ch_mask
            
            img_masks.append(ch_mask)
            imgs_lables[ch_mask] = i + 1

            img[ch_mask] = 0

        img = np.stack(img_masks, axis=-1).astype(np.float) * 255
        write_image(f'{dir_mask_labels_rgb}/{stem}{ext}', img) 
        write_image(f'{dir_mask_labels}/{stem}{ext}', imgs_lables) 

def cluster_normals_kmeans(path_normal, n_clusters=12, thres_uncertain = -1, folder_name_planes = 'pred_normal_planes'):
    '''Use k-means to cluster normals of images.
    Extract the maximum 6 planes, where the second largest 3 planes will remove the uncertain pixels by thres_uncertain
    '''
    PROP_PLANE = 0.03
    PROP_DOMINANT_PLANE = 0.05
    ANGLE_DIFF_DOMINANT_PLANE_MIN = 75
    ANGLE_DIFF_DOMINANT_PLANE_MAX = 105
    MAGNITUDE_PLANES_MASK = 1
    
    
    path_img_normal = path_normal[:-4]+'.png'
    img = np.load(path_normal)['arr_0']

    shape = img.shape
    num_pixels_img = shape[0] * shape [1]
    MIN_SIZE_PIXELS_PLANE_AREA = num_pixels_img*0.02

    if thres_uncertain > 0:
        path_alpha = IOUtils.add_file_name_prefix(path_normal, '../pred_alpha/')
        img_alpha = np.load(path_alpha)['arr_0']
        mask_uncertain = img_alpha > thres_uncertain

    path_rgb = IOUtils.add_file_name_prefix(path_img_normal, '../image/')
    img_rgb = read_image(path_rgb)[:,:,:3]
    img_rgb = resize_image(img_rgb, target_size=(shape[1], shape[0], 3))
        # img[mask_uncertain] = 0
        
        # path_uncertain = os.path.abspath( IOUtils.add_file_name_prefix(path_img_normal, '../pred_uncertain/') )
        # os.makedirs(os.path.dirname(path_uncertain),exist_ok=True)
        # write_image(path_uncertain, img)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img.reshape(-1, 3))
    pred = kmeans.labels_
    centers = kmeans.cluster_centers_

    num_max_planes = 7
    count_values = np.bincount(pred)
    max5 = np.argpartition(count_values,-num_max_planes)[-num_max_planes:]
    prop_planes = np.sort(count_values[max5] / num_pixels_img)[::-1]
    # logging.info(f'Proportion of planes: {prop_planes}; Proportion of the 3 largest planes: {np.sum(prop_planes[:3]):.04f}; Image name: {path_img_normal.split("/")[-1]}')
    centers_max5 = centers[max5]
    sorted_idx_max5 = np.argsort(count_values[max5] / num_pixels_img)
    sorted_max5 = max5[sorted_idx_max5][::-1]
    
    sorted_centers_max5  = centers_max5[sorted_idx_max5][::-1]
    # idx_center_uncertain_area = -1
    # for i in range(len(sorted_centers_max5)):
    #     # print(sorted_centers_max5[i].sum())
    #     if sorted_centers_max5[i].sum() < 1e-6:
    #         idx_center_uncertain_area = i
    # if idx_center_uncertain_area >= 0:
    #     sorted_max5 = np.delete(sorted_max5, idx_center_uncertain_area)

    
    colors_planes =[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]  # r,g,b, yellow, Cyan, Magenta
    planes_rgb = np.zeros(shape).reshape(-1,3)
    img_labels = np.zeros([*shape[:2]]).reshape(-1,1)
    count_planes = 0
    angles_diff = np.zeros(3)
    for i in range(6):
        curr_plane = (pred==sorted_max5[count_planes])

        # remove small isolated areas
        mask_clean = remove_small_isolated_areas((curr_plane>0).reshape(*shape[:2])*255, min_size=MIN_SIZE_PIXELS_PLANE_AREA).reshape(-1)
        curr_plane[mask_clean==0] = 0
    
        ratio_curr_plane = curr_plane.sum() / num_pixels_img

        check_sim = True
        if check_sim:
            # (1) check plane size
            if ratio_curr_plane < PROP_PLANE: # small planes: ratio > 2%
                continue
            if i < 3 and ratio_curr_plane < PROP_DOMINANT_PLANE: # dominant planes: ratio > 10%
                continue
            
            # (2) check normal similarity
            eval_metric = 'angle'
            if eval_metric == 'distance':
                curr_normal = sorted_centers_max5[count_planes]
                thres_diff = ANGLE_DIFF_DOMINANT_PLANE
                if i ==1: 
                    dist1 = np.linalg.norm(sorted_centers_max5[i-1]-curr_normal).sum()
                    angles_diff[0] = dist1
                    if dist1 < thres_diff:
                        continue
                if i == 2:
                    dist1 = np.linalg.norm(sorted_centers_max5[count_planes-1]-curr_normal).sum()
                    dist2 = np.linalg.norm(sorted_centers_max5[0]-curr_normal).sum()
                    angles_diff[1] = dist1
                    angles_diff[2] = dist2
                    if dist1 < thres_diff or dist2 < thres_diff:
                        continue
                        print(f'Dist1, dist2: {dist1, dist2}')
            if eval_metric == 'angle':
                curr_normal = sorted_centers_max5[count_planes]
                thres_diff_min = ANGLE_DIFF_DOMINANT_PLANE_MIN
                thres_diff_max = ANGLE_DIFF_DOMINANT_PLANE_MAX
                if i ==1: 
                    angle1 = calculate_normal_angle(sorted_centers_max5[i-1], curr_normal, use_degree=True)
                    angles_diff[0] = angle1
                    if angle1 < thres_diff_min or angle1 > thres_diff_max:
                        continue
                if i == 2:
                    angle1 = calculate_normal_angle(sorted_centers_max5[count_planes-1], curr_normal, use_degree=True)
                    angle2 = calculate_normal_angle(sorted_centers_max5[0], curr_normal, use_degree=True)
                    angles_diff[1] = angle1
                    angles_diff[2] = angle2
                    if angle1 < thres_diff_min or angle2 < thres_diff_min or \
                        angle1 >  thres_diff_max or angle2 >  thres_diff_max :
                        continue
                        print(f'Dist1, dist2: {dist1, dist2}')                
        count_planes += 1
        img_labels[curr_plane] = (i+1)*MAGNITUDE_PLANES_MASK

        if thres_uncertain > -1:
            # remove the influence of uncertainty pixels on small planes
            curr_plane[mask_uncertain.reshape(-1)] = 0
        planes_rgb[curr_plane] = colors_planes[i]

        # save current plane
        if img_rgb is not None:
            path_planes_visual_compose = IOUtils.add_file_name_prefix(path_img_normal, f"../{folder_name_planes}_visual_compose/{i+1}/",check_exist=True)
            mask_non_plane = (curr_plane < 1).reshape(shape[:2])
            img_compose = copy.deepcopy(img_rgb)
            img_compose[mask_non_plane] = 0
            write_image(path_planes_visual_compose, img_compose)



        # else:
        #     center0, center1, center2, center3 = centers[sorted_max5[0]], centers[sorted_max5[1]],centers[sorted_max5[2]], centers[sorted_max5[3]]
        #     diff = center1 - center2
        #     dist12 = np.linalg.norm(diff)
        #     if dist12 < 50:
        #         img_labels[pred==sorted_max5[3]]= i+1
        #         curr_channel = (pred==sorted_max5[3])
        #     else:
        #         img_labels[pred==sorted_max5[2]]= i+1
        #         curr_channel = (pred==sorted_max5[2])
        
    img_labels = img_labels.reshape(*shape[:2])
    path_labels = IOUtils.add_file_name_prefix(path_img_normal, f"../{folder_name_planes}/")
    write_image(path_labels, img_labels)
    msg_log = f'{path_img_normal.split("/")[-1]}: {prop_planes} {np.sum(prop_planes[:3]):.04f} {1.0 - (img_labels==0).sum() / num_pixels_img : .04f}. Angle differences (degrees): {angles_diff}'
    logging.info(msg_log)

    # planes_rgb = np.stack(planes_rgb, axis=-1).astype(np.float32)*255
    # visualization
    planes_rgb = planes_rgb.reshape(shape)
    path_planes_visual = IOUtils.add_file_name_prefix(path_img_normal, f"../{folder_name_planes}_visual/", check_exist=True)
    planes_rgb = cv2.cvtColor(planes_rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    mask_planes = planes_rgb.sum(axis=-1)>0
    # mask_planes = np.stack([mask_planes]*3, axis=-1)
    curre_img_ = copy.deepcopy(img_rgb)
    curre_img_[mask_planes==False] = (255,255,255)
    # planes_rgb_cat = np.concatenate((planes_rgb, curre_img_, img_rgb))
    # write_image(path_planes_visual, planes_rgb_cat)

    # visualize plane error
    img_normal_error = np.zeros(img.shape[:2])
    MAX_ANGLE_ERROR = 15
    for i in range(int(np.max(img_labels))):
        id_label = i+1
        mask_plane_curr = (img_labels==id_label)
        if mask_plane_curr.sum() ==0:
            continue
        normal_curr_plane = img[mask_plane_curr]

        mean_normal_curr = normal_curr_plane.mean(axis=0)
        angle_error = calculate_normal_angle(normal_curr_plane, mean_normal_curr, use_degree=True)
        img_normal_error[mask_plane_curr] = np.abs(angle_error)

    if thres_uncertain > -1:
        # remove the influence of uncertainty pixels on small planes
        img_normal_error[mask_uncertain] = 0
    
    path_planes_visual_error = IOUtils.add_file_name_prefix(path_img_normal, f"../{folder_name_planes}_visual_error/", check_exist=True)
    path_planes_visual_error2 = IOUtils.add_file_name_suffix(path_planes_visual_error, "_jet")
    
    img_normal_error_cmap =  convert_gray_to_cmap(img_normal_error.clip(0, MAX_ANGLE_ERROR))
    # img_normal_error_cmap = convert_color_BRG2RGB(img_normal_error_cmap)
    img_normal_error_cmap[mask_planes==False] = (255,255,255)
    
    img_normal_error_stack = np.stack([img_normal_error.clip(0, MAX_ANGLE_ERROR)]*3, axis=-1)/MAX_ANGLE_ERROR*255
    img_normal_error_stack[mask_planes==False] = (255,255,255)

    # img_gap = 255 * np.ones((img_normal_error.shape[0], 20, 3)).astype('uint8')
    # write_image(path_planes_visual_error, np.concatenate([img_rgb, img_gap, planes_rgb, img_gap, img_normal_error_cmap, img_gap, img_normal_error_stack], axis=1))
    write_image_lis(path_planes_visual, [img_rgb, planes_rgb, curre_img_, img_normal_error_cmap, img_normal_error_stack])
    # plt.imsave(path_planes_visual_error2, img_normal_error, vmin=0, vmax=MAX_ANGLE_ERROR, cmap = 'jet')
    # plt.imsave(path_planes_visual_error, img_normal_error, vmin=0, vmax=MAX_ANGLE_ERROR, cmap = 'gray')
    
    # img_normal_error = img_normal_error/np.max(img_normal_error) * 255
    # write_image(path_planes_visual_error, img_normal_error)  

    # if planes_rgb.shape[2] > 3:
    #     path_planes_visual2 = IOUtils.add_file_name_suffix(path_planes_visual, '_2')
    #     write_image(path_planes_visual2, planes_rgb[:,:, 3:])

    # img2 = copy.deepcopy((img_rgb))
    # mask_planes2_reverse = (planes_rgb[:,:, 3:].sum(axis=-1) == 0)
    # img2[mask_planes2_reverse] = 0
    # path_planes2_color = IOUtils.add_file_name_suffix(path_planes_visual, f"_color2")
    # write_image(path_planes2_color, img2)

    # img3 = copy.deepcopy((img_rgb))
    # img3[planes_rgb[:,:, 3:].sum(axis=-1) > 0] = 255
    # path_color2_reverse = IOUtils.add_file_name_suffix(path_planes_visual, '_color2_reverse')
    # write_image(path_color2_reverse, img3)

    # path_default = IOUtils.add_file_name_suffix(path_planes_visual, '_default')
    # write_image(path_default, img_rgb)
    return msg_log

def remove_image_background(path_png, path_mask, path_merge = None, reso_level=0):
    '''Remove image background using mask and resize image if reso_level > 0
    Args:
        path_png: path of source image
    Return:
        img_with_mask: image without background
    '''
    img = cv2.imread(path_png)
    mask = cv2.imread(path_mask)[:,:,-1]
    res = cv2.bitwise_and(img, img, mask=mask)
    mask = np.expand_dims(mask, -1)
    img_with_mask = np.concatenate((res, mask), axis=2)

    if reso_level > 0:
        shape_target = img_with_mask.shape[:2] // np.power(2, reso_level)
        shape_target = (shape_target[1], shape_target[0])
        img_with_mask = cv2.resize(img_with_mask, shape_target, interpolation=cv2.INTER_LINEAR)

    if path_merge != None:
        cv2.imwrite(path_merge, img_with_mask)
    logging.debug(f"Remove background of img: {path_png}")
    return img_with_mask

def resize_image(img, target_size, interpolation=cv2.INTER_LINEAR):
    '''Resize image to target size
    Args:
        target_size: (W,H)
    Return img_resize
    
    '''
    W,H = target_size[0], target_size[1]
    if img.shape[0] != H or img.shape[1] != W:
        img_resize = cv2.resize(img, (W,H), interpolation=interpolation)
        return img_resize
    else:
        return img

def convert_images_type(dir_imgs, dir_target, 
                        rename_mode = 'stem',
                        target_img_size = None, ext_source = '.png', ext_target = '.png', 
                        sample_interval = -1):
    '''Convert image type in directory from ext_imgs to ext_target
    '''
    IOUtils.ensure_dir_existence(dir_target)
    vec_path_files = sorted(glob.glob(f'{dir_imgs}/**{ext_source}'))
    stems = []
    for i in range(len(vec_path_files)):
        pp, stem, ext = IOUtils.get_path_components(vec_path_files[i])
        # id_img = int(stem)
        id_img = i
        if sample_interval > 1:
            # sample images
            if id_img % sample_interval !=0:
                continue
            
        if rename_mode == 'stem':
            path_target = f'{dir_target}/{stem}{ext_target}'
        elif rename_mode == 'order':
            path_target = f'{dir_target}/{i}{ext_target}'
        elif rename_mode == 'order_04d':
            path_target = f'{dir_target}/{i:04d}{ext_target}'
        else:
            NotImplementedError
        
        if ext_source[-4:] == ext_target:
            IOUtils.copy_file(vec_path_files[i], path_target)
        else:
            img = read_image(vec_path_files[i])
            write_image(path_target, img, target_img_size=target_img_size)
        
        stems.append(stem)
    return len(vec_path_files), stems

# image noise
def add_image_noise(image, noise_type='gauss', noise_std = 10):
    '''Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:
        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    Ref: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    '''
    if noise_type == "gauss":
        row,col,ch= image.shape
        mean = 0
        sigma = noise_std
        logging.debug(f'Gauss noise (mean, sigma): {mean,sigma}')
        gauss = np.random.normal(mean,sigma,(row,col, ch))
        gauss = gauss.reshape(row,col, ch)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

# lines
def extract_lines(dir_image, img_size, focal, ext_img = '.png'):
    dir_lines = dir_image + '_lines'
    IOUtils.ensure_dir_existence(dir_lines)

    dir_only_lines = dir_image + '_only_lines'
    IOUtils.ensure_dir_existence(dir_only_lines)

    vec_path_imgs = glob.glob(f'{dir_image}/**{ext_img}')
    for i in tqdm(range(len(vec_path_imgs))):
        path = vec_path_imgs[i]
        _, stem, ext = IOUtils.get_path_components(path)
        path_lines_img = f'{dir_lines}/{stem}.png'
        path_lines_txt = f'{dir_lines}/{stem}.txt'
        path_log =  f'{dir_lines}/num_lines.log'
       
        width = img_size[0]
        height = img_size[1]
        
        path_only_lines =  f'{dir_only_lines}/{stem}.png'
        os.system(f'{DIR_FILE}/VanishingPoint {path} {path_lines_img} {path_lines_txt} {width} {height} {focal} {path_log} {path_only_lines}')

# image segmentation
def extract_superpixel(path_img, CROP = 16):
    scales, markers = [1], [400]

    img = cv2.imread(path_img)
    image = copy.deepcopy(img)
    image = image[CROP:-CROP, CROP:-CROP, :]

    segments = []
    for s, m in zip(scales, markers):
        image = cv2.resize(image, (384//s, 288//s), interpolation=cv2.INTER_LINEAR)
        image = img_as_float(image)

        gradient = sobel(rgb2gray(image))
        # segment = watershed(gradient, markers=m, compactness=0.001)
        segment = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
        segments.append(segment)
    img_seg = segments[0].astype(np.int16)
    img_seg = resize_image(img_seg, target_size=img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return img, img_seg

def find_labels_max_clusters(pred, num_max_clusters):
    '''Find labels of the maximum n clusters with descending order
    Args:
        pred: N*1
        num_max_clusters: how many clusters to find
    '''
    count_values = np.bincount(pred)
    num_max_clusters = np.min([len(count_values), num_max_clusters])
    max_planes = np.argpartition(count_values,-num_max_clusters)[-num_max_clusters:]
    
    sorted_idx_max_planes = np.argsort(count_values[max_planes] / len(pred))
    sorted_max_planes = max_planes[sorted_idx_max_planes][::-1]
    prop_planes = np.sort(count_values[sorted_max_planes] / len(pred))[::-1]
    return sorted_max_planes, prop_planes, num_max_clusters

def visualize_semantic_label():
    # scannet
    img = read_image('/media/hp/HKUCS2/Dataset/ScanNet/scannet_whole/scene0079_00/label-filt/1.png')

    img_mask = np.zeros(img.shape + tuple([3]))
    for i in range(int(img.max()+1)):
        mask_curr = (img == i)
        img_mask[mask_curr] = np.random.randint(0, 255, 3)
    write_image('./test.png', img_mask)

def remove_small_isolated_areas(img, min_size = 3000):
    f'''Remove the small isolated areas with size smaller than defined {min_size}
    '''
    if img.ndim ==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = copy.deepcopy(img).astype(np.uint8)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img_clean = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img_clean[output == i + 1] = 255

    return img_clean

def convert_gray_to_cmap(img_gray, map_mode = 'jet', revert = True, vmax = None):
    '''Visualize point distances with 'hot_r' color map
    Args:
        cloud_source
        dists_to_target
    Return:
        cloud_visual: use color map, max
    '''
    img_gray = copy.deepcopy(img_gray)
    shape = img_gray.shape

    cmap = plt.get_cmap(map_mode)
    if vmax is not None:
        img_gray = img_gray / vmax
    else:
        img_gray = img_gray / (np.max(img_gray)+1e-6)
    if revert:
        img_gray = 1- img_gray      
    colors = cmap(img_gray.reshape(-1))[:, :3]

    # visualization
    colors = colors.reshape(shape+tuple([3]))*255
    return colors

#image editting
def crop_images(dir_images_origin, dir_images_crop, crop_size, img_ext = '.png'):
    IOUtils.ensure_dir_existence(dir_images_crop)
    imgs, stems_img = read_images(dir_images_origin, img_ext = img_ext)
    W_target, H_target = crop_size
    for i in range(len(imgs)):
        img_curr = imgs[i]
        H_origin, W_origin = img_curr.shape[:2]
        
        crop_width_half = (W_origin-W_target)//2
        crop_height_half = (H_origin-H_target) //2
        assert (W_origin-W_target)%2 ==0 and (H_origin- H_target) %2 == 0
        
        img_crop = img_curr[crop_height_half:H_origin-crop_height_half, crop_width_half:W_origin-crop_width_half]
        if img_ext == '.png':
            write_image(f'{dir_images_crop}/{stems_img[i]}.png', img_crop)
        elif img_ext == '.npy':
            np.save(f'{dir_images_crop}/{stems_img[i]}.npy', img_crop)
        else:
            raise NotImplementedError
        
    return crop_width_half, crop_height_half

def split_video_to_frames(path_video, dir_images):
    IOUtils.ensure_dir_existence(dir_images)
    vidcap = cv2.VideoCapture(path_video)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f'{dir_images}/{count:04d}.png', image)     # save frame as JPEG file      
        success,image = vidcap.read()
        logging.info(f'Read a new frame: {count}, {success}.')
        count += 1
    logging.info(f'End. Frames: {count}')

# concatenate images
def write_image_lis(path_img_cat, lis_imgs, use_cmap = False, interval_img = 20, cat_mode = 'horizontal', color_space = 'BGR'):
    '''Concatenate an image list to a single image and save it to the target path
    Args:
        cat_mode: horizonal/vertical
    '''
    img_cat = []
    for i in range(len(lis_imgs)):
        img = lis_imgs[i]
        H, W = img.shape[:2]
        
        if use_cmap:
            img = convert_gray_to_cmap(img) if img.ndim==2 else img
        else:
            img = np.stack([img]*3, axis=-1) if img.ndim==2 else img
        
        if img.max() <= 1.0:
            img *= 255

        img_cat.append(img)
        if cat_mode == 'horizontal':
            img_cat.append(255 * np.ones((H, interval_img, 3)).astype('uint8'))
        elif cat_mode == 'vertical':
            img_cat.append(255 * np.ones((interval_img, W, 3)).astype('uint8'))
    
    if cat_mode == 'horizontal':
        img_cat = np.concatenate(img_cat[:-1], axis=1)
    elif cat_mode == 'vertical':
        img_cat = np.concatenate(img_cat[:-1], axis=0)
    else:
        raise NotImplementedError
    if color_space == 'RGB':
        img_cat = cv2.cvtColor(img_cat.astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imwrite(path_img_cat, img_cat)


def calculate_psnr_nerf(path_img_src, path_img_gt, mask = None):
    img_src = read_image(path_img_src) / 255.0 #[:480]
    img_gt = read_image(path_img_gt) / 255.0
    # print(img_gt.shape)
    
    img_src = torch.from_numpy(img_src)
    img_gt = torch.from_numpy(img_gt)
    
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    err = img2mse(img_src, img_gt)
    # print(f'Error shape: {err.shape} {err}')
    psnr = mse2psnr(err)
    return float(psnr)
    
    
def eval_imgs_psnr(dir_img_src, dir_img_gt, sample_interval):
    vec_stems_imgs = IOUtils.get_files_stem(dir_img_src, '.png')   
    psnr_all = []
    for i in tqdm(range(0, len(vec_stems_imgs), sample_interval)):
        stem_img = vec_stems_imgs[i]
        path_img_src = f'{dir_img_src}/{stem_img}.png'  
        path_img_gt   = f'{dir_img_gt}/{stem_img[9:13]}.png' 
        # print(path_img_src, path_img_gt)
        psnr = calculate_psnr_nerf(path_img_src, path_img_gt)
        # print(f'PSNR: {psnr} {stem_img}')
        psnr_all.append(psnr)
    return np.array(psnr_all), vec_stems_imgs
        
