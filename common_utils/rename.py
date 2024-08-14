''' move images from dir "rgb" to dir "rename_rgb" and change the name format '''

import os
rgb_folder = "rgb"
rename_folder = "rename_rgb"
if not os.path.exists(rename_folder):
    os.makedirs(rename_folder)

for filename in os.listdir(rgb_folder):
    if filename.endswith(".png"):
        file_number = os.path.splitext(filename)[0]
        new_filename = file_number.zfill(4) + ".png"
        os.rename(os.path.join(rgb_folder, filename), os.path.join(rename_folder, new_filename))