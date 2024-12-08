import sys, os
import argparse

DIR_FILE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(DIR_FILE, '..'))

import utils.utils_io as IOUtils
import os, sys, glob, subprocess
from pathlib import Path
import numpy as np
import cv2
import re

# SfM and MVS paras
nNumThreads = 6 
nNumViews = 5
nMaxResolution = 7000
fDepthDiffThreshold = 0.005
fNormalDiffThreshold = 25
bRemoveDepthMaps = True
verbosity = 2
nRamdomIters = 4
nEstiIters = 2

from confs.path import DIR_MVS_BUILD, DIR_MVG_BUILD



def perform_sfm(dir_images, dir_output, nImageWidth):
    IOUtils.ensure_dir_existence(dir_output)
    
    dir_undistorted_images = dir_output + "/undistorted_images" 
    IOUtils.changeWorkingDir(dir_output) 

    fFocalLength_pixel = 1.2 * nImageWidth
    IOUtils.INFO_MSG("Use sequential pipeline")
    args_sfm = ["python3",  DIR_FILE + "/sfm_pipeline.py", \
                            dir_images, dir_output, str(fFocalLength_pixel), str(nNumThreads), DIR_MVG_BUILD] 
    IOUtils.run_subprocess(args_sfm)


    dir_input_sfm2mvs = dir_output + "/reconstruction_sequential/sfm_data.bin"
    args_sfm2mvs = [DIR_MVG_BUILD +"/Linux-x86_64-RELEASE/openMVG_main_openMVG2openMVS", 
                        "-i", dir_input_sfm2mvs, 
                        "-o", dir_output + "/scene.mvs",
                        "-d", dir_undistorted_images]
    IOUtils.run_subprocess(args_sfm2mvs)

def removeFiles(path_files, file_ext):
    """
    Remove files in "path_files" with extension "file_ext"
    """
    paths = Path(path_files).glob("**/*" + file_ext)
    for path in paths:
        path_str = str(path)  # because path is object not string

        if os.path.exists(path_str):	# Remove file
            print(f"Remove file {path_str}.")
            os.remove(path_str)
        # else:
        #     print(f"File {path_str} doesn't exist."

def perform_mvs(dir_output, nResolutionLevel, bRemoveDepthMaps=True):
    os.chdir(dir_output)
    if bRemoveDepthMaps:
        removeFiles(os.getcwd(), ".dmap")

    DENSE_RECONSTRUCTION = DIR_MVS_BUILD + "/bin/DensifyPointCloud"
    args_dense_reconstruction = [DENSE_RECONSTRUCTION, "scene.mvs",  
                                    "--resolution-level", str(nResolutionLevel), 
                                    "--max-resolution", str(nMaxResolution),
                                    "--depth-diff-threshold", str(fDepthDiffThreshold),
                                    "--normal-diff-threshold", str(fNormalDiffThreshold),
                                    "--random-iters", str(nRamdomIters),
                                    "--estimation-iters",str(nEstiIters),
                                    "--verbosity", str(verbosity),
                                    "--number-views", str(nNumViews)
                                    ]
    # reconstruction
    IOUtils.run_subprocess(args_dense_reconstruction)
    
    # remove depth maps
    if bRemoveDepthMaps:
        removeFiles(os.getcwd(), ".dmap")

def extract_intrinsics_from_KRC(path_KRC, path_intrin, path_imgs_cal):
    fKRC = open(path_KRC, 'r').readlines()
    lines_K = fKRC[3:6]
    intrin = np.identity(4)
    idx_row = 0
    
    # get calibrated images
    stems_img_cal = []
    count_lines = 0
    for line in fKRC:
        line = re.split('/|\n', line)
        # print(line)   
        if count_lines%13==1:
            stems_img_cal.append(Path(line[-2]).stem)    
        count_lines+=1
    
    IOUtils.write_list_to_txt(path_imgs_cal, stems_img_cal)
    # get camera instrisics
    for line in lines_K:
        line = re.split('[ \] \[ , ; \n]', line)
        idx_col = 0
        for elem in line:
            if len(elem) > 0:
                intrin[idx_row, idx_col] = float(elem)
                
                idx_col +=1
            
        idx_row +=1
    np.savetxt(path_intrin, intrin, fmt='%f')         
    return intrin, stems_img_cal

def export_cameras(dir_mvs_output):
    BIN_EXPORT_DATA = DIR_MVS_BUILD + "/bin/ExportData"
    IOUtils.changeWorkingDir(dir_mvs_output)

    # export cameras
    dir_export_cameras = dir_mvs_output + "/cameras"
    IOUtils.ensure_dir_existence(dir_export_cameras)
    pExportData = subprocess.Popen([BIN_EXPORT_DATA, "scene.mvs", "--export-path", dir_export_cameras])
    pExportData.wait()

def select_extrin_to_poses(dir_source, dir_target, lis_stem_sample,
                        target_img_size = None, 
                        ext_source = '.txt', ext_target = '.txt'):
    '''Convert image type in directory from ext_imgs to ext_target
    '''
    IOUtils.ensure_dir_existence(dir_target)
    stems = []
    for i in range(len(lis_stem_sample)):
        # id_img = int(stem)
        stem_curr = lis_stem_sample[i]        
        path_source = f'{dir_source}/{stem_curr}{ext_source}'       
        path_target = f'{dir_target}/{stem_curr}{ext_target}'       

        extrin = np.loadtxt(path_source)
        pose = np.linalg.inv(extrin)
        np.savetxt(path_target, pose)

def select_images(dir_source, dir_target, lis_stem_sample,
                        target_img_size = None, 
                        ext_source = '.png', ext_target = '.png'):
    '''Convert image type in directory from ext_imgs to ext_target
    '''
    IOUtils.ensure_dir_existence(dir_target)
    stems = []
    for i in range(len(lis_stem_sample)):
        # id_img = int(stem)
        stem_curr = lis_stem_sample[i]        
        path_source = f'{dir_source}/{stem_curr}{ext_source}'       
        path_target = f'{dir_target}/{stem_curr}{ext_target}'       
        if ext_source[-4:] == ext_target and (target_img_size is not None):
            IOUtils.copy_file(path_source, path_target)
        else:
            img = cv2.imread(path_source, cv2.IMREAD_UNCHANGED)
            if target_img_size is not None:
                img = cv2.resize(img, target_img_size)
            cv2.imwrite(path_target, img)

def select_depths(dir_source, dir_target, lis_stem_sample, size_image, 
                        target_img_size = None, 
                        ext_source = '.npy', ext_target = '.npy',
                        stem_prefix_src = 'depth'):
    '''Convert image type in directory from ext_imgs to ext_target
    '''
    IOUtils.ensure_dir_existence(dir_target)
    
    path_depth0 = f'{dir_source}/{stem_prefix_src}{lis_stem_sample[0]}{ext_source}'  
    depth0 = np.load(path_depth0).reshape(size_image[1], size_image[0])
    
    for i in range(len(lis_stem_sample)):
        # id_img = int(stem)
        stem_curr = lis_stem_sample[i] 
        path_source = f'{dir_source}/{stem_curr}{ext_source}'       
        if stem_prefix_src is not None:       
            path_source = f'{dir_source}/{stem_prefix_src}{stem_curr}{ext_source}'       
        path_target = f'{dir_target}/{stem_curr}{ext_target}'
        if not IOUtils.checkExistence(path_source):
            print(f'Depth not existed. Create one with all zeros... {path_target}')
            depth_tmp = np.zeros_like(depth0)
            np.save(path_target, depth_tmp)  
            continue
        
        if ext_source[-4:] == ext_target and (target_img_size is None):
            depth_curr = np.load(path_source).reshape(size_image[1], size_image[0])
            np.save(path_target, depth_curr)
            # IOUtils.copy_file(path_source, path_target)
        else:
            raise NotImplementedError
     
def prepare_neus_data(dir_sfm_output, dir_neus, imgs_cal_stem, args):
    IOUtils.ensure_dir_existence(dir_neus)
    
    # images
    dir_src = f'{dir_sfm_output}/undistorted_images'
    dir_target = f'{dir_sfm_output}/images_calibrated'
    select_images(dir_src, dir_target, imgs_cal_stem)

    dir_src = f'{dir_sfm_output}'
    dir_target = f'{dir_sfm_output}/depth_calibrated'
    assert args.reso_level == 1
    select_depths(dir_src, dir_target, imgs_cal_stem, (args.image_width, args.image_height))
    
    # cameras
    dir_src = f'{dir_sfm_output}/cameras/extrinsics'
    dir_target = f'{dir_neus}/pose'
    select_extrin_to_poses(dir_src, dir_target, imgs_cal_stem)

    # intrinsics
    path_src = f'{dir_sfm_output}/cameras/intrinsics.txt'
    path_target = f'{dir_sfm_output}/intrinsics.txt'
    IOUtils.copy_file(path_src, path_target)
    
    # point cloud
    path_src = f'{dir_sfm_output}/scene_dense.ply'
    scene_name = path_src.split('/')[-3]
    path_target = f'{dir_neus}/point_cloud_openMVS.ply'
    IOUtils.copy_file(path_src, path_target)
    
    # neighbors
    path_src = f'{dir_sfm_output}/neighbors.txt'
    path_target = f'{dir_neus}/neighbors.txt'
    IOUtils.copy_file(path_src, path_target)
    
    
def perform_sfm_mvs_pipeline(dir_mvs, args):
    dir_images = os.path.join(dir_mvs, 'images')
    dir_output = dir_mvs
    dir_neus = f'{dir_output}/..'
    
    # reconstruction
    perform_sfm(dir_images, dir_output, args.image_width)
    perform_mvs(dir_output, nResolutionLevel = args.reso_level)
    export_cameras(dir_output)
    
    # Prepare neus data: image, intrin, extrins
    dir_KRC = f'{dir_output}/cameras/KRC.txt'
    path_intrin = f'{dir_output}/cameras/intrinsics.txt'
    path_imgs_cal = f'{dir_output}/cameras/stems_calibrated_images.txt'
    intrin, stems_img_cal = extract_intrinsics_from_KRC(dir_KRC, path_intrin, path_imgs_cal)
    prepare_neus_data(dir_output, dir_neus, stems_img_cal, args)
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_mvs', type=str, default='/home/ethan/Desktop/test_sfm/tmp_sfm_mvs' )
    parser.add_argument('--image_width', type=int, default = 1024)
    parser.add_argument('--image_height', type=int, default = 768)
    parser.add_argument('--reso_level', type=int, default = 1)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()    
    perform_sfm_mvs_pipeline(args.dir_mvs, args)
