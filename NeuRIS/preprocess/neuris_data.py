
import os, glob
import logging, copy, pickle

import numpy as np
from cv2 import  cv2
from tqdm import tqdm

from preprocess.scannet_data import ScannetData
from utils.utils_geometry import read_cam_matrix, resize_cam_intrin
from utils.utils_image import cluster_normals_kmeans, find_labels_max_clusters, remove_small_isolated_areas
from utils.utils_io import add_file_name_suffix, checkExistence
from utils.utils_image import extract_superpixel, read_image, read_images, write_image, \
                                find_labels_max_clusters, read_image, read_images, write_image

import utils.utils_geometry as GeoUtils
import utils.utils_image  as ImageUtils
import utils.utils_io as IOUtils

from confs.path import path_tiltedsn_pth_pfpn, path_tiltedsn_pth_sr, dir_tiltedsn_code, \
                            path_snu_pth, dir_snu_code, \
                            lis_name_scenes_remove


def crop_images_neuris(dir_imgs, dir_imgs_crop, path_intrin, path_intrin_crop, crop_size):
    assert checkExistence(dir_imgs)
    crop_width_half, crop_height_half = ImageUtils.crop_images(dir_imgs, dir_imgs_crop, crop_size)
    GeoUtils.modify_intrinsics_of_cropped_images(path_intrin, path_intrin_crop,  crop_width_half, crop_height_half)
    
def extract_planes_from_normals(dir_normal, thres_uncertain = 70, folder_name_planes = 'pred_normal_planes', n_clusters = 12):
    vec_path_files = sorted(glob.glob(f'{dir_normal}/**.npz'))
    path_log = f'{dir_normal}/../log_extract_planes.txt'
    f_log = open(path_log, 'w')
    for i in tqdm(range(len(vec_path_files))):
        path = vec_path_files[i]
        msg_log = cluster_normals_kmeans(path, n_clusters= n_clusters, thres_uncertain = thres_uncertain, folder_name_planes = folder_name_planes)
        f_log.write(msg_log + '\n')
    f_log.close()

def segment_one_image_superpixels(path_img, folder_visual):
    # refer to: https://github.com/SJTU-ViSYS/StructDepth for more details
    MAGNITUDE_PLANE_LABEL = 1
    image, pred = extract_superpixel(path_img)
    shape = image.shape

    num_max_clusters= 12
    pred = pred.reshape(-1)
    sorted_max5, prop_planes, num_max_clusters = find_labels_max_clusters(pred, num_max_clusters=num_max_clusters)

    # visulize labels
    colors_planes =[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]  # r,g,b, yellow, Cyan, Magenta
    planes_rgb = np.zeros(shape).reshape(-1,3)
    img_labels = np.zeros([*shape[:2]]).reshape(-1,1)
    for i in range(num_max_clusters):
        curr_plane = (pred==sorted_max5[i])
        ratio = curr_plane.sum() / shape[0]*shape[1]
        if ratio < 0.05:
            continue

        img_labels[curr_plane] = (i+1) * MAGNITUDE_PLANE_LABEL
        if i < 6:
            planes_rgb[curr_plane] = colors_planes[i]
        else:
            planes_rgb[curr_plane] = np.random.randint(low=0, high=255, size=3)

        
    img_labels = img_labels.reshape(*shape[:2])
    write_image(IOUtils.add_file_name_prefix(path_img, f'../{folder_visual}/'), img_labels)

    planes_rgb = planes_rgb.reshape(shape)
    mask_planes = planes_rgb.sum(axis=-1 )>0
    mask_planes = np.stack([mask_planes]*3, axis=-1)
    curre_img_compose = copy.deepcopy(image )
    curre_img_compose[mask_planes==False] = 0
    planes_rgb_cat = np.concatenate([planes_rgb, curre_img_compose, image], axis=0)
    write_image(IOUtils.add_file_name_prefix(path_img, f'../{folder_visual}_visual/'), planes_rgb_cat)

    msg_log = f'{prop_planes} {np.sum(prop_planes[:6]):.04f} {path_img.split("/")[-1]}'

    return msg_log

def segment_images_superpixels(dir_images):
    '''Segment images in directory and save segmented images in the same parent folder
    Args:
        dir_images
    Return:
        None
    '''
    vec_path_imgs = IOUtils.get_files_path(dir_images, '.png')
    path_log = f'{dir_images}/log_seg_images.txt'
    f_log = open(path_log, 'w')
    for i in tqdm(range(len(vec_path_imgs))):
        path = vec_path_imgs[i]
        msg_log = segment_one_image_superpixels(path, folder_visual='image_planes')
        f_log.write(msg_log + '\n')
    f_log.close()

def compose_normal_img_planes(dir):
    MAGNITUDE_COMPOSED_PLANER_LABEL = 1
    PROPORTION_PLANES = 0.03  # 5%

    dir_img_planes = f'{dir}/image_planes'
    dir_normal_planes = f'{dir}/pred_normal_planes'
    
    dir_planes_compose = f'{dir}/pred_normal_subplanes'
    dir_planes_compose_visual = f'{dir}/pred_normal_subplanes_visual'
    IOUtils.ensure_dir_existence(dir_planes_compose)
    IOUtils.ensure_dir_existence(dir_planes_compose_visual)
    
    planes_normal_all, stem_normals = read_images(f'{dir}/pred_normal_planes')
    planes_color_all, stem_imgs = read_images( f'{dir}/image_planes', target_img_size=planes_normal_all[0].shape[::-1], interpolation=cv2.INTER_NEAREST)
    imgs_rgb,_ = read_images(f'{dir}/image', target_img_size=planes_normal_all[0].shape[::-1])

    # 0. all planes
    planes_compose_all = []
    planes_compose_rgb_all = []
    f_log = open(f'{dir}/log_num_subplanes.txt', 'w')
    for i in tqdm(range(len(planes_color_all))):
        curr_planes_normal = planes_normal_all[i]
        curr_planes_color = planes_color_all[i]
        
        # 1. nornal planes
        curr_planes_compose = np.zeros_like(curr_planes_normal)
        curr_planes_compose_rgb = np.zeros((curr_planes_normal.shape[0], curr_planes_normal.shape[1], 3))
        count_planes = 0
        for j in range(int(curr_planes_normal.max())):
            mask_curr_plane_normal = (curr_planes_normal==j+1)

            # make other planes' label be zero
            curr_planes_color_2 = copy.deepcopy(curr_planes_color)
            curr_planes_color_2[mask_curr_plane_normal==False] = 0

            sorted_max_planes, prop_planes, num_max_clusters2 = find_labels_max_clusters(curr_planes_color_2.reshape(-1), num_max_clusters=6)
            
            # 2. image planes
            ratios = np.zeros(6)
            for k in range(len(sorted_max_planes)):
                # check proportion and remove background
                if sorted_max_planes[k] == 0:
                    continue
                if prop_planes[k] < PROPORTION_PLANES:
                    continue
                ratios[k] = prop_planes[k]

                mask_curr_plane_img = (curr_planes_color_2 == sorted_max_planes[k])
                mask_curr_plane_img = remove_small_isolated_areas(mask_curr_plane_img, min_size=3000) > 0
                curr_planes_compose_rgb[mask_curr_plane_img] = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
                # curr_planes_compose_rgb = curr_planes_compose_rgb.astype(np.uint8)
                
                count_planes += 1
                curr_planes_compose[mask_curr_plane_img] = count_planes * MAGNITUDE_COMPOSED_PLANER_LABEL
        assert count_planes < 255 // MAGNITUDE_COMPOSED_PLANER_LABEL
        f_log.write(f'{stem_imgs[i]}: {count_planes} \n')
        write_image(f'{dir_planes_compose}/{stem_imgs[i]}.png', curr_planes_compose)

        mask_planes = curr_planes_compose_rgb.sum(axis=-1 )>0
        mask_planes = np.stack([mask_planes]*3, axis=-1)
        curre_img = copy.deepcopy(imgs_rgb[i])
        curre_img[mask_planes==False] = 0
        curr_planes_compose_rgb2 = np.concatenate([curr_planes_compose_rgb, curre_img, imgs_rgb[i]], axis=0)
        write_image(f'{dir_planes_compose_visual}/{stem_imgs[i]}.png', curr_planes_compose_rgb2)

        planes_compose_rgb_all.append(curr_planes_compose_rgb)
    f_log.close()

# scannet     
def prepare_neuris_data_from_scannet(dir_scan, dir_neus, sample_interval=6, 
                                    b_sample = True, 
                                    b_crop_images = True,
                                    b_generate_neus_data = True,
                                    b_pred_normal = True, normal_method = 'snu',
                                    b_detect_planes = False):
    '''Sample iamges (1296,968)
    '''   
    
    H,W, _ = read_image(f'{dir_scan}/rgb/0.jpg').shape
    path_intrin_color = f'{dir_scan}/../intrinsic_color.txt'
    
    if W == 1296: 
        path_intrin_color_crop_resize = f'{dir_scan}/../intrinsic_color_crop1248_resize640.txt'
        origin_size = (1296, 968)
        cropped_size = (1248, 936)
        reso_level = 1.95
    elif W == 640:
        path_intrin_color_640 = add_file_name_suffix(path_intrin_color, '_640')
        if not checkExistence(path_intrin_color_640):
            intrin_temp = read_cam_matrix(path_intrin_color)
            intrin_640 = resize_cam_intrin(intrin_temp, resolution_level=1296/640)
            np.savetxt(path_intrin_color_640, intrin_640, fmt='%f')
            
        path_intrin_color = path_intrin_color_640
        path_intrin_color_crop_resize = f'{dir_scan}/../intrinsic_color_crop640_resize640.txt'
        origin_size = (640, 480)
        cropped_size = (624, 468)
        reso_level = 0.975   
    else:
        raise NotImplementedError
    
    if not IOUtils.checkExistence(path_intrin_color_crop_resize):
        crop_width_half = (origin_size[0]-cropped_size[0])//2
        crop_height_half =(origin_size[1]-cropped_size[1]) //2
        
        
        intrin = np.loadtxt(path_intrin_color)
        intrin[0,2] -= crop_width_half
        intrin[1,2] -= crop_height_half
        intrin_resize = resize_cam_intrin(intrin, reso_level)
        np.savetxt(path_intrin_color_crop_resize, intrin_resize, fmt = '%f')
    
    if b_sample:
        start_id = 0
        num_images = len(glob.glob(f"{dir_scan}/rgb/**.jpg"))
        end_id = num_images*2
        ScannetData.select_data_by_range(dir_scan, dir_neus, start_id, end_id, sample_interval, 
                                            b_crop_images, cropped_size)
    # prepare neus data
    if b_generate_neus_data:
        dir_neus = dir_neus
        height, width =  480, 640

        msg = input('Remove pose (nan)...[y/n]') # observe pose file size
        if msg != 'y':
            exit()

        path_intrin_color = f'{dir_scan}/../../intrinsic_color.txt'
        if b_crop_images:
            path_intrin_color = path_intrin_color_crop_resize
        path_intrin_depth = f'{dir_scan}/../../intrinsic_depth.txt'
        dataset = ScannetData(dir_neus, height, width, 
                                    use_normal = False,
                                    path_intrin_color = path_intrin_color,
                                    path_intrin_depth = path_intrin_depth)
        dataset.generate_neus_data()

        if b_pred_normal:
            predict_normal(dir_neus, normal_method)

        # detect planes
        if b_detect_planes == True:
            extract_planes_from_normals(dir_neus + '/pred_normal', thres_uncertain=-1)
            segment_images_superpixels(dir_neus + '/image')
            compose_normal_img_planes(dir_neus)
        
# privivate data
def prepare_neuris_data_from_private_data(dir_neus, size_img = (6016, 4016),
                                            b_generate_neus_data = True,
                                            b_pred_normal = True, normal_method = 'snu',
                                            b_detect_planes = False):   
    '''Run sfm and prepare neus data
    '''
    W, H = size_img
    
    path_intrin_color = f'{dir_neus}/intrinsics.txt'
    if b_generate_neus_data:
        dataset = ScannetData(dir_neus, H, W, use_normal=False,
                                    path_intrin_color = path_intrin_color,
                                    path_intrin_depth = path_intrin_color, 
                                    path_cloud_sfm=f'{dir_neus}/point_cloud_openMVS.ply')
        dataset.generate_neus_data()
    
    if b_pred_normal:
        predict_normal(dir_neus, normal_method)

        
    # detect planes
    if b_detect_planes == True:
        extract_planes_from_normals(dir_neus + '/pred_normal', thres_uncertain=-1)
        segment_images_superpixels(dir_neus + '/image')
        compose_normal_img_planes(dir_neus)

def predict_normal(dir_neus, normal_method = 'snu'):
    # For scannet data, retraining of normal network is required to guarantee the test scenes are in the test set of normal network.
    # For your own data, the officially provided pretrained model of the normal network can be utilized directly.
    if normal_method == 'snu':
        # ICCV2021, https://github.com/baegwangbin/surface_normal_uncertainty
        logging.info('Predict normal')
        IOUtils.changeWorkingDir(dir_snu_code)
        os.system(f'python test.py --pretrained scannet --path_ckpt {path_snu_pth} --architecture BN --imgs_dir {dir_neus}/image/')

    # Tilted-SN
    if normal_method == 'tiltedsn':
        # ECCV2020, https://github.com/MARSLab-UMN/TiltedImageSurfaceNormal
        IOUtils.changeWorkingDir(dir_tiltedsn_code)
        os.system(f'python inference_surface_normal.py --checkpoint_path {path_tiltedsn_pth_pfpn} \
                                --sr_checkpoint_path {path_tiltedsn_pth_sr} \
                                --log_folder {dir_neus} \
                                --operation inference \
                                --batch_size 8 \
                                --net_architecture sr_dfpn \
                                --test_dataset {dir_neus}/image')
 
# normal prior
def update_pickle_TiltedSN_rectified_2dofa(path_pickle, path_update, scenes_remove):
    '''Remove part of training data
    '''
    data_split = pickle.load(open(path_pickle, 'rb'))
    keys_splits = ['train'] #, 'test'
    keys_subsplicts = ['e2', '-e2']

    masks_all = []
    remove_all = {}
    for key_split in keys_splits:
        for key_subsplit in keys_subsplicts:
            data_curr = data_split[key_split][key_subsplit]
            len_data_curr = len(data_curr)
            mask_curr = np.ones(len_data_curr).astype(bool)
            remove_all[key_split+'_'+key_subsplit] =  []
            for i in range(len_data_curr):
                path_i = data_curr[i].split('/')
                scene_i = path_i[0]
                if scene_i[:-3] in scenes_remove:
                    mask_curr[i] = False
                    if scene_i not in remove_all[key_split+'_'+key_subsplit]:
                        remove_all[key_split+'_'+key_subsplit].append(scene_i)
            
            remove_all[key_split+'_'+key_subsplit] = sorted(remove_all[key_split+'_'+key_subsplit])
            logging.info(f'{key_split} {key_subsplit}: {int(mask_curr.sum())}/{len_data_curr}. Ratio: {int(mask_curr.sum())/len_data_curr:.03f}')
            masks_all.append(mask_curr)
            # remove_all.append(remove_curr)

            # update original data
            data_split[key_split][key_subsplit] = np.array(data_split[key_split][key_subsplit])[mask_curr].tolist()
            len_update = len(data_split[key_split][key_subsplit])
            logging.info(f'[{key_split} {key_subsplit}]:{len_data_curr}->{len_update}')
            
    # save data
    with open(path_update, 'wb') as handle:
        pickle.dump(data_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # debug
    # data_split2 = pickle.load(open(path_update, 'rb'))
    # print('done')

def update_pickle_TiltedSN_full_2dofa(path_pickle, path_update, scenes_remove):
    '''Remove part of training data
    '''
    data_split = pickle.load(open(path_pickle, 'rb'))
    keys_splits = ['train']
    keys_subsplicts_l1 = ['with_ga', 'no_ga']
    keys_subsplicts_l2 = ['e2', '-e2']

    masks_all = []
    remove_all = {}
    for key_split in keys_splits:
        for key_l1 in keys_subsplicts_l1:   
            for key_l2 in keys_subsplicts_l2:
                if key_l1 == 'with_ga':
                    data_curr = data_split[key_split][key_l1][key_l2]
                elif key_l1 == 'no_ga':
                    data_curr = data_split[key_split][key_l1]
                else:
                    raise NotImplementedError
                
                len_data_curr = len(data_curr)
                mask_curr = np.ones(len_data_curr).astype(bool)
                remove_all[key_split + '_'+ key_l1 + '_'  + key_l2] =  []
                for i in range(len_data_curr):
                    path_i = data_curr[i].split('/')
                    scene_i = path_i[0]
                    if scene_i[:-3] in scenes_remove:
                        mask_curr[i] = False
                        if scene_i not in remove_all[key_split + '_'+ key_l1 + '_'  + key_l2]:
                            remove_all[key_split + '_'+ key_l1 + '_'  + key_l2].append(scene_i)
            
                logging.info(f'[{key_split} {key_l1} {key_l2}]: {int(mask_curr.sum())}/{len_data_curr}. Ratio: {int(mask_curr.sum())/len_data_curr:.03f}')
                masks_all.append(mask_curr)
                remove_all[key_split + '_'+ key_l1 + '_'  + key_l2] = sorted(remove_all[key_split + '_'+ key_l1 + '_'  + key_l2])
                # remove_all.append(remove_curr)

                # update original data
                if key_l1 == 'with_ga':
                    data_split[key_split][key_l1][key_l2] = np.array(data_split[key_split][key_l1][key_l2])[mask_curr].tolist()
                    len_update = len(data_split[key_split][key_l1][key_l2])
                elif key_l1 == 'no_ga':
                    data_split[key_split][key_l1] = np.array(data_split[key_split][key_l1])[mask_curr].tolist()
                    len_update = len(data_split[key_split][key_l1])
                else:
                    raise NotImplementedError

                logging.info(f'[{key_split} {key_l1} {key_l2}]:{len_data_curr}->{len_update}')
            
    # save data
    with open(path_update, 'wb') as handle:
        pickle.dump(data_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # debug
    # data_split2 = pickle.load(open(path_update, 'rb'))
    # print('done')

def update_pickle_file_TiltedSN():
    '''Remove part of training data
    '''
    path_pkl = '../TiltedImageSurfaceNormal/data/rectified_2dofa_framenet.pkl' 
    path_update = IOUtils.add_file_name_suffix(path_pkl, "_update")
    update_pickle_TiltedSN_rectified_2dofa(path_pkl, path_update, lis_name_scenes_remove)

    path_pkl = '../TiltedImageSurfaceNormal/data/full_2dofa_framenet.pkl'  
    path_update = IOUtils.add_file_name_suffix(path_pkl, "_update")
    update_pickle_TiltedSN_full_2dofa(path_pkl, path_update, lis_name_scenes_remove)

def update_pickle_framenet(path_pickle, path_update):
    '''Remove part of training data
    '''
    scenes_remove = lis_name_scenes_remove
    
    data_split = pickle.load(open(path_pickle, 'rb'))
    keys_splits = ['train'] #, 'test'
    keys_subsplicts = [0, 1, 2]

    masks_all = []
    remove_all = {}
    data_split_update = {'train':[],
                         'test':[]}
    scenes_all_used = []
    scenes_all = {}                     
    for key_split in keys_splits:
        for key_subsplit in keys_subsplicts:
            data_curr = data_split[key_split][key_subsplit]  
            key_subsplit2 = str(key_subsplit)          
            len_data_curr = len(data_curr)
            mask_curr = np.ones(len_data_curr).astype(bool)
            remove_all[key_split+'_'+key_subsplit2] =  []
            
            scenes_all[key_split]  = []
            for i in range(len_data_curr):
                path_i = data_curr[i].split('/')
                scene_i = path_i[4]
                if scene_i[:-3] in scenes_remove:
                    mask_curr[i] = False
                    if scene_i not in remove_all[key_split+'_'+key_subsplit2]:
                        remove_all[key_split+'_'+key_subsplit2].append(scene_i)
                # if scene_i[:-3] not in scenes_all[key_split]:
                scenes_all[key_split].append(scene_i)
            remove_all[key_split+'_'+key_subsplit2] = sorted(remove_all[key_split+'_'+key_subsplit2])
            logging.info(f'{key_split} {key_subsplit}: {int(mask_curr.sum())}/{len_data_curr}. Ratio: {int(mask_curr.sum())/len_data_curr:.03f}')
            masks_all.append(mask_curr)
            # remove_all.append(remove_curr)
            scenes_all[key_split] = np.unique(np.array(scenes_all[key_split]))

            # update original data
            data_update_ = np.array(data_split[key_split][key_subsplit])[mask_curr].tolist()
            data_split_update[key_split].append(data_update_)
            len_update = len(data_update_)
            logging.info(f'[{key_split} {key_subsplit}]:{len_data_curr}->{len_update}')
            
            num_remove = len(remove_all[key_split+'_'+key_subsplit2])
            logging.info(f'[{key_split}] scenes_all: {len(scenes_all[key_split])}, scenes-remove: {num_remove}')
            scenes_all_used.append(scenes_all) 
            
    # save data
    data_split['train'] = ( data_split_update['train'][0],data_split_update['train'][1], data_split_update['train'][2])
    with open(path_update, 'wb') as handle:
        pickle.dump(data_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # debug
    # data_split2 = pickle.load(open(path_update, 'rb'))
    # print('done')