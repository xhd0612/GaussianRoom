import torch
import torch.nn.functional as F

def get_angles(normals_source, normals_target):
    '''Get angular error betwee predicted normals and ground truth normals
    Args:
        normals_source, normals_target: N*3
        mask: N*1 (optional, default: None)
    Return:
        angular_error: float
    '''
    inner = (normals_source * normals_target).sum(dim=-1,keepdim=True)
    norm_source =  torch.linalg.norm(normals_source, dim=-1, ord=2,keepdim=True)
    norm_target = torch.linalg.norm(normals_target, dim=-1, ord=2,keepdim=True)
    angles = torch.arccos(inner/((norm_source*norm_target) + 1e-6)) #.clip(-np.pi, np.pi)
    assert not torch.isnan(angles).any()
    return angles

def get_angular_error(normals_source, normals_target, mask = None, clip_angle_error = -1):
    '''Get angular error betwee predicted normals and ground truth normals
    Args:
        normals_source, normals_target: N*3
        mask: N*1 (optional, default: None)
    Return:
        angular_error: float
    '''
    inner = (normals_source * normals_target).sum(dim=-1,keepdim=True)
    norm_source =  torch.linalg.norm(normals_source, dim=-1, ord=2,keepdim=True)
    norm_target = torch.linalg.norm(normals_target, dim=-1, ord=2,keepdim=True)
    angles = torch.arccos(inner/((norm_source*norm_target) + 1e-6)) #.clip(-np.pi, np.pi)
    assert not torch.isnan(angles).any()
    if mask is None:
        mask = torch.ones_like(angles)
    if mask.ndim == 1:
        mask =  mask.unsqueeze(-1)
    assert angles.ndim == mask.ndim

    mask_keep_gt_normal = torch.ones_like(angles).bool()
    if clip_angle_error>0:
        mask_keep_gt_normal = angles < clip_angle_error
        # num_clip = mask_keep_gt_normal.sum()
    angular_error = F.l1_loss(angles*mask*mask_keep_gt_normal, torch.zeros_like(angles), reduction='sum') / (mask*mask_keep_gt_normal+1e-6).sum()
    return angular_error, mask_keep_gt_normal

# evaluation
def calculate_psnr(img_pred, img_gt, mask=None):
    psnr = 20.0 * torch.log10(1.0 / (((img_pred - img_gt)**2).mean()).sqrt())
    return psnr

def convert_to_homo(pts):
    pts_homo = torch.cat([pts, torch.ones(pts.shape[:-1] + tuple([1])) ], axis=-1)
    return pts_homo