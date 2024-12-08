import os

BIN_DIR = "/home/ethan/Research/NeuRIS"
DIR_MVG_BUILD = BIN_DIR + "/openMVG_Build"
DIR_MVS_BUILD = BIN_DIR + "/openMVS_build"

# normal path
# dir_snu_code = '/path/snucode' # directory of code
# path_snu_pth = 'path/scannet.pt'
dir_snu_code = '/home/xhd/xhd/surface_normal_uncertainty' # directory of code
path_snu_pth = '/home/xhd/xhd/surface_normal_uncertainty/checkpoints/scannet.pt'
assert os.path.exists(path_snu_pth)

# dir_tiltedsn_code = '/path/tiltedsn_code'
# dir_tiltedsn_ckpt = '/path/tiltedsn_ckpt' # directory of pretrained model
# path_tiltedsn_pth_pfpn = f"{dir_tiltedsn_ckpt}/PFPN_SR_full/model-best.ckpt"
# path_tiltedsn_pth_sr = f"{dir_tiltedsn_ckpt}/SR_only/model-latest.ckpt"
# assert os.path.exists(path_tiltedsn_pth_sr)

# used scenes
names_scenes_neuris = ['scene0009_01', 'scene0085_00', 'scene0114_02',
                        'scene0603_00', 'scene0617_00', 'scene0625_00',
                        'scene0721_00', 'scene0771_00']
names_scenes_manhattansdf = ['scene0050_00', 'scene0084_00', 
                                'scene0580_00', 'scene0616_00']
lis_name_scenes = names_scenes_neuris + names_scenes_manhattansdf

# lis_name_scenes = ['scene0085_00']
# update training/test split
names_scenes_neuris_remove = ['scene0009', 'scene0085', 'scene0114',
                        'scene0603', 'scene0617', 'scene0625',
                        'scene0721', 'scene0771']
names_scenes_manhattansdf_remove = ['scene0050', 'scene0084', 
                                'scene0580', 'scene0616']
lis_name_scenes_remove = names_scenes_neuris_remove + names_scenes_manhattansdf_remove
    