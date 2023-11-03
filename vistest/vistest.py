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


def save_view_point(pcd, filename):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1920, height=1080)
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def three_load_view_point(pcd1, pcd2, pcd3, filename):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1920, height=1080)
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(filename)
    axis_pcd = open3d.geometry.create_mesh_coordinate_frame(size=3, origin=[0, 0, 0])
    T = np.linalg.inv(param.extrinsic)
    new_axis_pcd = copy.deepcopy(axis_pcd).transform(T)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.add_geometry(pcd3)
    vis.add_geometry(new_axis_pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def two_load_view_point(pcd1, pcd2, filename):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1920, height=1080)
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(filename)
    axis_pcd = open3d.geometry.create_mesh_coordinate_frame(size=3, origin=[0, 0, 0])
    T = np.linalg.inv(param.extrinsic)
    new_axis_pcd = copy.deepcopy(axis_pcd).transform(T)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.add_geometry(new_axis_pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def one_load_view_point(pcd, filename):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1920, height=1080)
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(filename)
    axis_pcd = open3d.geometry.create_mesh_coordinate_frame(size=3, origin=[0, 0, 0])
    T = np.linalg.inv(param.extrinsic)
    new_axis_pcd = copy.deepcopy(axis_pcd).transform(T)
    vis.add_geometry(pcd)
    vis.add_geometry(new_axis_pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()

def RPYAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(np.dot(R_z, R_y), R_x)
    return R

#################################################################################################

def vistest_projection(file_in_lidar, file_in_res2):
    axispcd = open3d.geometry.create_mesh_coordinate_frame(size=1, origin=[0, 0, 0])
    vis = open3d.Visualizer()
    vis.create_window(window_name='Open3D', width=600, height=600, left=10, top=10, visible=True)
    vis.get_render_option().point_size = 10

    tmp = os.path.join(file_in_lidar)
    plydata = PlyData.read(tmp)
    ply = plydata['vertex']
    plys = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()
    plys = plys.tolist()

    data = []
    with open (file_in_res2, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data.append(eachdata[2])

    campoints = []
    envpoints = []
    index = 0
    for point in plys:
        if index in data:
            campoints.append(point)
        else:
            envpoints.append(point)
        index += 1

    envpcd = open3d.geometry.PointCloud()
    envpcd.points = open3d.Vector3dVector(envpoints)
    envpcd.paint_uniform_color([0, 1.0, 0])
    objpcd = open3d.geometry.PointCloud()
    objpcd.points = open3d.Vector3dVector(campoints)
    objpcd.paint_uniform_color([0, 0, 1.0])

    key_to_callback = {}
    view = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.visualization.draw_geometries_with_key_callbacks([axispcd, objpcd, envpcd], key_to_callback)

def vistest_aftermatching(file_in_lidar, file_in_res34):
    axispcd = open3d.geometry.create_mesh_coordinate_frame(size=1, origin=[0, 0, 0])
    vis = open3d.Visualizer()
    vis.create_window(window_name='Open3D', width=600, height=600, left=10, top=10, visible=True)
    vis.get_render_option().point_size = 10

    tmp = os.path.join(file_in_lidar)
    plydata = PlyData.read(tmp)
    ply = plydata['vertex']
    plys = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()
    plys = plys.tolist()

    data = []
    with open (file_in_res34, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data.append(eachdata[2])

    objpoints = []
    envpoints = []
    index = 0
    for point in plys:
        if index in data:
            objpoints.append(point)
        else:
            envpoints.append(point)
        index += 1

    envpcd = open3d.geometry.PointCloud()
    envpcd.points = open3d.Vector3dVector(envpoints)
    envpcd.paint_uniform_color([0, 1.0, 0])
    objpcd = open3d.geometry.PointCloud()
    objpcd.points = open3d.Vector3dVector(objpoints)
    objpcd.paint_uniform_color([0, 0, 1.0])

    key_to_callback = {}
    view = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.visualization.draw_geometries_with_key_callbacks([axispcd, objpcd, envpcd], key_to_callback)

def vistests_afterdeal(file_in_lidar, file_in_res5678):
    axispcd = open3d.geometry.create_mesh_coordinate_frame(size=1, origin=[0, 0, 0])
    vis = open3d.Visualizer()
    vis.create_window(window_name='Open3D', width=600, height=600, left=10, top=10, visible=True)
    vis.get_render_option().point_size = 10

    tmp = os.path.join(file_in_lidar)
    plydata = PlyData.read(tmp)
    ply = plydata['vertex']
    plys = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()
    plys = plys.tolist()

    data = []
    with open (file_in_res5678, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data.append([eachdata[0],eachdata[1],eachdata[2]])

    objpoints = []
    envpoints = []
    for point in plys:
        if point in data:
            objpoints.append(point)
        else:
            envpoints.append(point)

    envpcd = open3d.geometry.PointCloud()
    envpcd.points = open3d.Vector3dVector(envpoints)
    envpcd.paint_uniform_color([0, 1.0, 0])
    objpcd = open3d.geometry.PointCloud()
    objpcd.points = open3d.Vector3dVector(objpoints)
    objpcd.paint_uniform_color([0, 0, 1.0])

    key_to_callback = {}
    view = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.visualization.draw_geometries_with_key_callbacks([axispcd, objpcd, envpcd], key_to_callback)

def vistest_incamerasys(file_in_lidar,file_in_res34):
    tmp = os.path.join(file_in_lidar)
    plydata = PlyData.read(tmp)
    ply = plydata['vertex']
    plys = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()
    plys = plys.tolist()

    data = []
    with open (file_in_res34, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data.append(eachdata[2])

    objpoints = []
    envpoints = []
    index = 0
    for point in plys:
        if index in data:
            objpoints.append(point)
        else:
            envpoints.append(point)
        index += 1

    envpcd = open3d.geometry.PointCloud()
    envpcd.points = open3d.Vector3dVector(envpoints)
    envpcd.paint_uniform_color([0, 1.0, 0])
    objpcd = open3d.geometry.PointCloud()
    objpcd.points = open3d.Vector3dVector(objpoints)
    objpcd.paint_uniform_color([0, 0, 1.0])
    # save_view_point(pcd, "viewpoint.json")
    two_load_view_point(envpcd,objpcd, 'viewpoint.json')

def vistest_beforematching2d(file_in_res1,file_in_res2):
    axispcd = open3d.geometry.create_mesh_coordinate_frame(size=1, origin=[0, 0, 0])
    vis = open3d.Visualizer()
    vis.create_window(window_name='Open3D', width=600, height=600, left=10, top=10, visible=True)
    vis.get_render_option().point_size = 10

    data_res1 = []
    with open(file_in_res1, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data_res1.append([eachdata[0],eachdata[1],1])

    data_res3 = []
    with open(file_in_res2, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data_res3.append([eachdata[0],eachdata[1],1])

    print(len(data_res1))
    print(len(data_res3))

    res1pcd = open3d.geometry.PointCloud()
    res1pcd.points = open3d.Vector3dVector(data_res1)
    res1pcd.paint_uniform_color([0, 1.0, 0])
    res3pcd = open3d.geometry.PointCloud()
    res3pcd.points = open3d.Vector3dVector(data_res3)
    res3pcd.paint_uniform_color([0, 0, 1.0])

    key_to_callback = {}
    view = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.visualization.draw_geometries_with_key_callbacks([res1pcd,res3pcd], key_to_callback)

def vistest_aftermatching2d(file_in_res3,file_in_res2):
    axispcd = open3d.geometry.create_mesh_coordinate_frame(size=1, origin=[0, 0, 0])
    vis = open3d.Visualizer()
    vis.create_window(window_name='Open3D', width=600, height=600, left=10, top=10, visible=True)
    vis.get_render_option().point_size = 10

    index_res3 = []
    with open(file_in_res3, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            index_res3.append(int(eachdata[2]))

    all_res2 = []
    with open(file_in_res2, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            all_res2.append([eachdata[0],eachdata[1],eachdata[2]])

    data_res3 = []
    data_res2 = []
    for data in all_res2:
        if data[2] in index_res3:
            data_res3.append([data[0],data[1],1])
        else:
            data_res2.append([data[0],data[1],1])

    print(len(data_res3))
    print(len(data_res2))

    res4pcd = open3d.geometry.PointCloud()
    res4pcd.points = open3d.Vector3dVector(data_res4)
    res4pcd.paint_uniform_color([0, 1.0, 0])
    res3pcd = open3d.geometry.PointCloud()
    res3pcd.points = open3d.Vector3dVector(data_res3)
    res3pcd.paint_uniform_color([0, 0, 1.0])

    key_to_callback = {}
    view = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.visualization.draw_geometries_with_key_callbacks([res4pcd,res3pcd], key_to_callback)

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

# vistest_incamerasys(file_in_lidar='000000.ply',file_in_res34=)
# vistests_afterdeal(file_in_lidar='./testdaten13mar/000002.ply',file_in_res5678='./testdaten13mar/res8people0002_14.json')
vistest_transrotat(file_in_lidar='../input/lidarcloud/000000.ply',file_in_carkeypoints='../result/res8car/0000_0002.json',file_in_carmesh='../input/carmesh/0001_0002.obj',
                   file_in_translation='../output/t/0002.json',file_in_rotation='../output/a/0002.json')