import open3d
import json
import os
from plyfile import PlyData
import pandas as pd

def vistests_afterdeal(file_in_lidar, file_in_res678):
    axispcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=600, height=600, left=10, top=10, visible=True)
    vis.get_render_option().point_size = 10

    tmp = os.path.join(file_in_lidar)
    plydata = PlyData.read(tmp)
    ply = plydata['vertex']
    plys = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()
    plys = plys.tolist()

    data = []
    with open (file_in_res678, 'r') as f:
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
    envpcd.points = open3d.utility.Vector3dVector(envpoints)
    envpcd.paint_uniform_color([0, 1.0, 0])
    objpcd = open3d.geometry.PointCloud()
    objpcd.points = open3d.utility.Vector3dVector(objpoints)
    objpcd.paint_uniform_color([0, 0, 1.0])

    key_to_callback = {}
    view = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.visualization.draw_geometries_with_key_callbacks([axispcd, objpcd, envpcd], key_to_callback)

vistests_afterdeal(file_in_lidar='../input/lidarcloud/000532.ply',file_in_res678='../result_new/res7car/0532_5603.json')