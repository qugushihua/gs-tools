'''transform the params.npz format in SplaTAM to the view format in gaussian_spaltting_lightning'''
import numpy as np
from plyfile import PlyData, PlyElement

data = np.load('params.npz')

# view the array names contained in the file
print(data.files)

# load each key-value
means3D = data['means3D']
rgb_colors = data['rgb_colors']
unnorm_rotations = data['unnorm_rotations']
logit_opacities = data['logit_opacities']
log_scales = data['log_scales']
cam_unnorm_rots = data['cam_unnorm_rots']
cam_trans = data['cam_trans']
timestep = data['timestep']
intrinsics = data['intrinsics']
w2c = data['w2c']
org_width = data['org_width']
org_height = data['org_height']
gt_w2c_all_frames = data['gt_w2c_all_frames']
keyframe_time_indices = data['keyframe_time_indices']

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def generate_point_cloud(num_points, max_sh_degree):
    # generate random pcd
    # xyz = np.random.rand(num_points, 3)
    # opacities = np.random.rand(num_points, 1)
    # features_dc = np.random.rand(num_points, 3, 1)
    # num_extra_features = 3 * (max_sh_degree + 1) ** 2 - 3
    # features_extra = np.random.rand(num_points, num_extra_features)
    # scale_names = 3
    # scales = np.random.rand(num_points, scale_names)
    # rot_names = 4
    # rots = np.random.rand(num_points, rot_names)

    xyz = means3D

    opacities = logit_opacities

    fused_color = RGB2SH(rgb_colors)
    features = np.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2))
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0
    features_dc = features[:,:,0:1]
    num_extra_features = 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((num_points, num_extra_features))

    scales = log_scales

    rots = unnorm_rotations

    return xyz, opacities, features_dc, features_extra, scales, rots

def save_point_cloud(path, num_points, max_sh_degree):
    xyz, opacities, features_dc, features_extra, scales, rots = generate_point_cloud(num_points, max_sh_degree)

    # create pcd structure
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + \
                 [('opacity', 'f4')] + \
                 [('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')] + \
                 [(f'f_rest_{i}', 'f4') for i in range(features_extra.shape[1])] + \
                 [(f'scale_{i}', 'f4') for i in range(scales.shape[1])] + \
                 [(f'rot_{i}', 'f4') for i in range(rots.shape[1])]

    elements = np.empty(num_points, dtype=dtype_full)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['opacity'] = opacities[:, 0]
    elements['f_dc_0'] = features_dc[:, 0, 0]
    elements['f_dc_1'] = features_dc[:, 1, 0]
    elements['f_dc_2'] = features_dc[:, 2, 0]

    for idx in range(features_extra.shape[1]):
        elements[f'f_rest_{idx}'] = features_extra[:, idx]

    for idx in range(scales.shape[1]):
        elements[f'scale_{idx}'] = scales[:, idx]

    for idx in range(rots.shape[1]):
        elements[f'rot_{idx}'] = rots[:, idx]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

save_point_cloud('outputs/vanilla_3dgs/random/point_cloud/iteration_30000/point_cloud.ply', num_points=means3D.shape[0], max_sh_degree=3)