'''read points data from params.npz and generate points3D.txt'''
import numpy as np
data = np.load('params.npz')

means3D = data['means3D']
rgb_colors = data['rgb_colors']

import numpy as np
# keys = data.keys()

# for key in keys:
#     print("Key:", key)
# print(data['unnorm_rotations'].shape)

assert means3D.shape[0] == rgb_colors.shape[0], "both arrays must be the same length"

with open('points3D.txt', 'w') as file:
    for point_id, (xyz, rgb) in enumerate(zip(means3D, rgb_colors)):
        normalized_rgb = np.clip(rgb, 0, 1)
        scaled_rgb = normalized_rgb * 255
        line = f"{point_id}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\t{int(scaled_rgb[0])}\t{int(scaled_rgb[1])}\t{int(scaled_rgb[2])}\n"
        file.write(line)
print("points data have been written to points3D.txt")