# borrowed from nerfingmvs and neuralreon

import os, cv2,logging
import numpy as np
import open3d as o3d

import submodules.NeuRIS.utils.utils_geometry as GeoUtils
import submodules.NeuRIS.utils.utils_io as IOUtils

def load_gt_depths(image_list, datadir, H=None, W=None):
    depths = []
    masks = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, '{:04d}.png'.format(int(frame_id)))
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000
        
        if H is not None:
            mask = (depth > 0).astype(np.uint8)
            depth_resize = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            depths.append(depth_resize)
            masks.append(mask_resize > 0.5)
        else:
            depths.append(depth)
            masks.append(depth > 0)
    return np.stack(depths), np.stack(masks)

def load_depths_npy(image_list, datadir, H=None, W=None):
    depths = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, '{}_depth.npy'.format(frame_id))
        if not os.path.exists(depth_path):
            depth_path = os.path.join(datadir, '{}.npy'.format(frame_id))
        depth = np.load(depth_path)
        
        if H is not None:
            depth_resize = cv2.resize(depth, (W, H))
            depths.append(depth_resize)
        else:
            depths.append(depth)

    return np.stack(depths)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def depth_evaluation(gt_depths, pred_depths, savedir=None, pred_masks=None, min_depth=0.1, max_depth=20, scale_depth = False):
    assert gt_depths.shape[0] == pred_depths.shape[0]

    gt_depths_valid = []
    pred_depths_valid = []
    errors = []
    num = gt_depths.shape[0]
    for i in range(num):
        gt_depth = gt_depths[i]
        mask = (gt_depth > min_depth) * (gt_depth < max_depth)
        gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = cv2.resize(pred_depths[i], (gt_width, gt_height))

        if pred_masks is not None:
            pred_mask = pred_masks[i]
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_width, gt_height)) > 0.5
            mask = mask * pred_mask

        if mask.sum() == 0:
            continue

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        pred_depths_valid.append(pred_depth)
        gt_depths_valid.append(gt_depth)

    ratio = 1.0
    if scale_depth:
        ratio = np.median(np.concatenate(gt_depths_valid)) / \
                    np.median(np.concatenate(pred_depths_valid))
    
    for i in range(len(pred_depths_valid)):
        gt_depth = gt_depths_valid[i]
        pred_depth = pred_depths_valid[i]

        pred_depth *= ratio
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)

    # print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()))
    # print("\n-> Done!")

    if savedir is not None:
        with open(os.path.join(savedir, 'depth_evaluation.txt'), 'a+') as f:
            if len(f.readlines()) == 0:
                f.writelines(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + '    scale_depth\n')
            f.writelines(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + f"    {scale_depth}   \\\\")
    
    return mean_errors
   
def save_evaluation_results(dir_log_eval, errors_mesh, name_exps, step_evaluation):
    # save evaluation results to latex format
    mean_errors_mesh = errors_mesh.mean(axis=0)     # 4*7

    names_log = ['err_gt_mesh', 'err_gt_mesh_scale', 'err_gt_depth', 'err_gt_depth_scale']
    dir_log_eval = f'{dir_log_eval}/{step_evaluation:08d}'
    IOUtils.ensure_dir_existence(dir_log_eval)
    for idx_errror_type in range(4):
        with open(f'{dir_log_eval}/{names_log[idx_errror_type]}.txt', 'w') as f_log:
            len_name = len(name_exps[0][0])
            f_log.writelines(('No.' + ' '*np.max([0, len_name-len('   scene id')]) + '     scene id     ' + "{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + '\n')
            for idx_scan in range(errors_mesh.shape[0]):
                f_log.writelines((f'[{idx_scan}] {name_exps[idx_scan][0]} ' + ("&{: 8.3f}  " * 7).format(*errors_mesh[idx_scan, idx_errror_type, :].tolist())) + f" \\\ {name_exps[idx_scan][1]}\n")
            
            f_log.writelines((' '*len_name + 'Mean' + " &{: 8.3f} " * 7).format(*mean_errors_mesh[idx_errror_type, :].tolist()) + " \\\ \n")
            
def evaluate_geometry_neucon(file_pred, file_trgt, threshold=.05, down_sample=.02):
    """ Borrowed from NeuralRecon
    Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points

    Returns:
        Dict of mesh metrics
    """

    def nn_correspondance(verts1, verts2):
        """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's

        Returns:
            ([indices], [distances])

        """

        indices = []
        distances = []
        if len(verts1) == 0 or len(verts2) == 0:
            return indices, distances

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts1)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        for vert in verts2:
            _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
            indices.append(inds[0])
            distances.append(np.sqrt(dist[0]))

        return indices, distances

    pcd_pred = GeoUtils.read_point_cloud(file_pred)
    pcd_trgt = GeoUtils.read_point_cloud(file_trgt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)  # para2->para1: dist1 is gt->pred
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {'dist1': np.mean(dist2),  # pred->gt
               'dist2': np.mean(dist1),  # gt -> pred
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               }
    # plot graph
    # if path_fscore_curve:
    #     EvalUtils.draw_figure_fscore(path_fscore_curve, threshold, dist2, dist1, plot_stretch=5)

    metrics = np.array([np.mean(dist2), np.mean(dist1), precision, recal, fscore])
    logging.info(f'{file_pred.split("/")[-1]}: {metrics}')
    return metrics

def evaluate_3D_mesh(path_mesh_pred, scene_name, dir_dataset = './dataset/indoor',
                            eval_threshold = 0.05, reso_level = 2.0, 
                            check_existence = True):
    '''Evaluate geometry quality of neus using Precison, Recall and F-score.
    '''
    path_intrin = f'{dir_dataset}/intrinsic_depth.txt'
    target_img_size = (640, 480)
    dir_scan = f'{dir_dataset}/{scene_name}'
    dir_poses = f'{dir_scan}/pose'
    # dir_images = f'{dir_scan}/image'
    
    path_mesh_gt = f'{dir_dataset}/{scene_name}/{scene_name}_vh_clean_2.ply'
    path_mesh_gt_clean = IOUtils.add_file_name_suffix(path_mesh_gt, '_clean')
    path_mesh_gt_2dmask = f'{dir_dataset}/{scene_name}/{scene_name}_vh_clean_2_2dmask.npz'
    
    # (1) clean GT mesh
    GeoUtils.clean_mesh_faces_outside_frustum(path_mesh_gt_clean, path_mesh_gt, 
                                                path_intrin, dir_poses, 
                                                target_img_size, reso_level=reso_level,
                                                check_existence = check_existence)
    GeoUtils.generate_mesh_2dmask(path_mesh_gt_2dmask, path_mesh_gt_clean, 
                                                path_intrin, dir_poses, 
                                                target_img_size, reso_level=reso_level,
                                                check_existence = check_existence)
    # for fair comparison
    GeoUtils.clean_mesh_faces_outside_frustum(path_mesh_gt_clean, path_mesh_gt, 
                                                path_intrin, dir_poses, 
                                                target_img_size, reso_level=reso_level,
                                                path_mask_npz=path_mesh_gt_2dmask,
                                                check_existence = check_existence)


    # (2) clean predicted mesh
    path_mesh_pred_clean_bbox = IOUtils.add_file_name_suffix(path_mesh_pred, '_clean_bbox')
    path_mesh_pred_clean_bbox_faces = IOUtils.add_file_name_suffix(path_mesh_pred, '_clean_bbox_faces')
    path_mesh_pred_clean_bbox_faces_mask = IOUtils.add_file_name_suffix(path_mesh_pred, '_clean_bbox_faces_mask')

    GeoUtils.clean_mesh_points_outside_bbox(path_mesh_pred_clean_bbox, path_mesh_pred, path_mesh_gt,
                                                scale_bbox=1.1,
                                                check_existence = check_existence)
    GeoUtils.clean_mesh_faces_outside_frustum(path_mesh_pred_clean_bbox_faces, path_mesh_pred_clean_bbox, 
                                                    path_intrin, dir_poses, 
                                                    target_img_size, reso_level=reso_level,
                                                    check_existence = check_existence)
    GeoUtils.clean_mesh_points_outside_frustum(path_mesh_pred_clean_bbox_faces_mask, path_mesh_pred_clean_bbox_faces, 
                                                    path_intrin, dir_poses, 
                                                    target_img_size, reso_level=reso_level,
                                                    path_mask_npz=path_mesh_gt_2dmask,
                                                    check_existence = check_existence)
    
    path_eval = path_mesh_pred_clean_bbox_faces_mask 
    metrices_eval = evaluate_geometry_neucon(path_eval, path_mesh_gt_clean, 
                                                        threshold=eval_threshold, down_sample=.02) #f'{dir_eval_fig}/{scene_name}_step{iter_step:06d}_thres{eval_threshold}.png')

    return metrices_eval

def save_evaluation_results_to_latex(path_log, 
                                        header = '                     Accu.      Comp.      Prec.     Recall     F-score \n', 
                                        results = None, 
                                        names_item = None, 
                                        save_mean = None, 
                                        mode = 'w',
                                        precision = 3):
    '''Save evaluation results to txt in latex mode
    Args:
        header:
            for F-score: '                     Accu.      Comp.      Prec.     Recall     F-score \n'
        results:
            narray, N*M, N lines with M metrics
        names_item:
            N*1, item name for each line
        save_mean: 
            whether calculate the mean value for each metric
        mode:
            write mode, default 'w'
    '''
    # save evaluation results to latex format
    with open(path_log, mode) as f_log:
        if header:
            f_log.writelines(header)
        if results is not None:
            num_lines, num_metrices = results.shape
            if names_item is None:
                names_item = np.arange(results.shape[0])
            for idx in range(num_lines):
                f_log.writelines((f'{names_item[idx]}    ' + ("&{: 8.3f}  " * num_metrices).format(*results[idx, :].tolist())) + " \\\ \n")
        if save_mean:
            mean_results = results.mean(axis=0)     # 4*7
            mean_results = np.round(mean_results, decimals=precision)
            f_log.writelines(( '       Mean    ' + " &{: 8.3f} " * num_metrices).format(*mean_results[:].tolist()) + " \\\ \n")
 


