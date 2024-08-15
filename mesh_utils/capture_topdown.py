'''load mesh and capture topdown view png'''
import open3d as o3d
import numpy as np

# load scene
mesh = o3d.io.read_triangle_mesh("/home/user/Workspace/giftednav_ws/src/GiftedNav/habitat/goat-bench/data/scene_datasets/hm3d/val/00829-QaLdnwvtxbs/QaLdnwvtxbs.glb", True, True)

# create a visual window
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
vis.add_geometry(mesh)

# set camera parameters to get a top view
ctr = vis.get_view_control()
parameters = ctr.convert_to_pinhole_camera_parameters()

# adjust the camera position to enlarge the scene
parameters.extrinsic = np.array([[1, 0, 0, 0],
                                 [0, 0, -1, -5],  # move the camera position down
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]])
ctr.convert_from_pinhole_camera_parameters(parameters)

# render and save the image
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("top_down_view.png")
vis.destroy_window()