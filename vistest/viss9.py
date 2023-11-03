import open3d
import numpy as np
import os
import json
from plyfile import PlyData
import pandas as pd
import copy
import re
import math


def loadMesh(fileName):
    data = []
    with open(fileName, 'r') as f:
        for line in f.readlines():
            tmp = line.split()
            if len(tmp) == 4:
                if tmp[0] == 'v':
                    vertices = np.array([float(tmp[1]), float(tmp[2]), float(tmp[3])])
                    vertices = 2 * vertices
                    data.append(vertices)
    return data

def onlypoints_pcd(points, colors):
    point_pcd = open3d.geometry.PointCloud()
    point_pcd.points = open3d.utility.Vector3dVector(points)
    point_pcd.paint_uniform_color(colors)
    return point_pcd

def vistest_transrotat(file_in_lidar, file_in_carkeypoints, file_in_carmesh, file_in_translation, file_in_rotation):
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=600, height=600, left=10, top=10, visible=True)
    vis.get_render_option().point_size = 10

    tmp = os.path.join(file_in_lidar)
    plydata = PlyData.read(tmp)
    ply = plydata['vertex']
    plys = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()

    with open(file_in_rotation, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
        R = RPYAnglesToRotationMatrix([0,0,eachdata])
    T = []
    with open(file_in_translation, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            T.append(eachdata)
        T = np.array(T)
    data = loadMesh(file_in_carmesh)
    data = np.array(data)
    data_mesh = np.zeros(np.shape(data))
    data_mesh[:, 0] = data[:, 2]
    data_mesh[:, 1] = -1 * data[:, 0]
    data_mesh[:, 2] = -1 * data[:, 1]
    points = []
    for eachdata in data_mesh:
        vertices = np.array(eachdata)
        vertices_obj = vertices.T
        vertices_world = np.dot(R, vertices_obj) + T
        vertices_app = [vertices_world[0], vertices_world[1], vertices_world[2]]
        points.append(vertices_app)
    points = np.array(points)
    carpoint_pcd = onlypoints_pcd(points, [0, 0, 1.0])
    carlidar = []
    with open(file_in_carkeypoints, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            carlidar.append(eachdata)
        carlidar = np.array(carlidar)
    carlidar_pcd = onlypoints_pcd(carlidar, [1.0, 0, 0])
    envpoint_pcd = onlypoints_pcd(plys, [0, 1.0, 0])
    key_to_callback = {}
    view = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.visualization.draw_geometries_with_key_callbacks([axis_pcd, envpoint_pcd, carlidar_pcd, carpoint_pcd], key_to_callback)

vistest_transrotat(file_in_lidar='../input/lidarcloud/000000.ply',file_in_carkeypoints='../result/res8car/0000_0002.json',file_in_carmesh='../input/carmesh/0001_0002.obj',
                   file_in_translation='../output/t/0002.json',file_in_rotation='../output/a/0002.json')