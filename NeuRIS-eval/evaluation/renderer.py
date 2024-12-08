
# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

# Borrowed from Atlas

import numpy as np
import trimesh, pyrender

from tqdm import tqdm
from skimage import measure
from matplotlib.cm import get_cmap as colormap

import utils.utils_geometry as GeoUtils



class Renderer():
    """Borrowed from Atlas
    OpenGL mesh renderer 
    
    Used to render depthmaps from a mesh for 2d evaluation
    """
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        #self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)#, self.render_flags) 

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R =  np.array([[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose@axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


def render_depthmaps_pyrender(path_mesh, path_intrin, dir_poses,
                path_mask_npz = None):

    # gt depth data loader
    width, height = 640, 480

    # mesh renderer
    renderer = Renderer()
    mesh = trimesh.load(path_mesh, process=False)
    mesh_opengl = renderer.mesh_opengl(mesh)
    
    intrin = GeoUtils.read_cam_matrix(path_intrin)
    poses = GeoUtils.read_poses(dir_poses)
    num_images = len(poses)
    depths_all = []
    for i in tqdm(range(num_images)):
        pose_curr = poses[i]
        _, depth_pred = renderer(height, width, intrin, pose_curr, mesh_opengl)
        depths_all.append(depth_pred)

    depths_all = np.array(depths_all)
    return depths_all