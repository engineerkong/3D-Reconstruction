import open3d
import json
import numpy as np
import copy

def two_load_view_point(pcd1, pcd2, filename):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1920, height=1080)
    vis.get_render_option().point_size = 2
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(filename)
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    T = np.linalg.inv(param.extrinsic)
    new_axis_pcd = copy.deepcopy(axis_pcd).transform(T)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.add_geometry(new_axis_pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()

def vistest_matching(file_in_res1, file_in_res4567):
    data_res1 = []
    with open(file_in_res1, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data_res1.append([eachdata[0],eachdata[1],eachdata[2]])

    data_res4567 = []
    with open (file_in_res4567, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data_res4567.append([eachdata[0],eachdata[1],eachdata[2]])

    objpoints = []
    envpoints = []
    for point in data_res1:
        if point in data_res4567:
            objpoints.append(point)
        else:
            envpoints.append(point)

    print(len(objpoints))
    envpcd = open3d.geometry.PointCloud()
    envpcd.points = open3d.utility.Vector3dVector(envpoints)
    envpcd.paint_uniform_color([0, 1.0, 0])
    objpcd = open3d.geometry.PointCloud()
    objpcd.points = open3d.utility.Vector3dVector(objpoints)
    objpcd.paint_uniform_color([0, 0, 1.0])
    two_load_view_point(envpcd, objpcd, '../input/viewpoint.json')

vistest_matching(file_in_res1='../result_new/res1/0000.json',file_in_res4567='../result_new/res7car/0000_0001.json')