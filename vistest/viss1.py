import open3d
import json
import numpy as np
import copy
import os
from plyfile import PlyData
import pandas as pd

def one_load_view_point(pcd, filename):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1920, height=1080)
    vis.get_render_option().point_size = 2
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(filename)
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    T = np.linalg.inv(param.extrinsic)
    new_axis_pcd = copy.deepcopy(axis_pcd).transform(T)
    vis.add_geometry(pcd)
    vis.add_geometry(new_axis_pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()

def vistest_beforeransac(file_in_lidar):
    tmp = os.path.join(file_in_lidar)
    plydata = PlyData.read(tmp)
    ply = plydata['vertex']
    plys = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()
    plys = plys.tolist()

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(plys)
    pcd.paint_uniform_color([0, 1.0, 0])
    one_load_view_point(pcd, '../input/viewpoint.json')

def vistest_afterransac(file_in_res1):
    data_res1 = []
    with open(file_in_res1, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data_res1.append([eachdata[0],eachdata[1],eachdata[2]])

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(data_res1)
    pcd.paint_uniform_color([0, 1.0, 0])
    one_load_view_point(pcd, '../input/viewpoint.json')

vistest_beforeransac(file_in_lidar='../input/lidarcloud/000000.ply')
vistest_afterransac(file_in_res1='../result_new/res1/0000.json')