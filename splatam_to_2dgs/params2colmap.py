'''read camera intrinsic & extrinsic data from params.npz and generate cameras.txt & images.txt 
   note: the intrinsic need to be edited
'''
import numpy as np
import sys
import os
from pyquaternion import Quaternion

if __name__ == '__main__':
    # output_dir = sys.argv[1]
    output_dir = 'result'
    param_path = os.path.join(output_dir, 'params.npz')
    images_path = os.path.join(output_dir, 'images.txt')
    points3d_path = os.path.join(output_dir, 'points3D.txt')
    cameras_path = os.path.join(output_dir, 'cameras.txt')
    traj_file = open(images_path, 'w')
    points3d_file = open(points3d_path, 'w')
    cameras_file = open(cameras_path, 'w')
    param_dict = dict(np.load(param_path, allow_pickle=True))
    ts = param_dict['cam_trans'][0].transpose()
    rs = param_dict['cam_unnorm_rots'][0].transpose()
    cam_writer = '1 PINHOLE 1024 1024 512.0000000000001 512.0000000000001 511.0 511.0'
    cameras_file.write(cam_writer)
    for i, r in enumerate(rs):
        q = Quaternion(r/np.linalg.norm(r))
        R = q.rotation_matrix
        T = np.zeros((4,4))
        T[:3,:3] = R
        T[:3,3] = ts[i]
        T[3,3] = 1
        traj = np.squeeze(T.reshape(1,16))
        traj_line = str(i) + ' ' +str(q.w) + ' ' +str(q.x) + ' ' +str(q.y) + ' ' +str(q.z) + ' ' +str(ts[i][0]) + ' ' +str(ts[i][1]) + ' ' +str(ts[i][2]) + ' ' + str(1) +' '+ str(i).zfill(4)+'.png\n\n'
        # traj_line = str(i) + ' ' +str(q.w) + ' ' +str(q.x) + ' ' +str(q.y) + ' ' +str(q.z) + ' ' +str(ts[i][0]) + ' ' +str(ts[i][1]) + ' ' +str(ts[i][2]) + ' ' + str(1) +' '+ str(i)+'.png\n\n'
        traj_file.write(traj_line)
    traj_file.close()
    cameras_file.close()
    points3d_file.close()
