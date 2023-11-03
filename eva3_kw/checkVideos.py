import json
import cv2
import glob
import numpy as np
from plyfile import PlyData
import pandas as pd
from tqdm import tqdm
import os
import open3d as o3d
import sys
import re
import csv
import warnings
import math

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
def loadCsv1(fileName):
    file = open(fileName)
    csvreader = csv.reader(file)
    # header = next(csvreader)
    # print(header)
    rows = []
    i = 0
    for row in csvreader:
        if i != 0:
            rows.append(row)
        i = 1
    file.close()
    return rows

def loadCsv2(fileName):
    file = open(fileName)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    return rows

def loadData(fileName):
    data = []
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data.append(eachdata)
    return data

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

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, heading_angle):

    h = size[2]
    w = size[1]
    l = size[0]

    R = rotz(1*heading_angle)
    l = l/2
    w = w/2
    h = h/2
    center[2] = center[2] - h
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)

def camerabbox(cap,t,ses):
    imgs = []
    frame = int(t * ses[0]['fps'])  # find frame depending of fps
    cap[0].set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ret, img = cap[0].read()
    imgs.append(img)
    return imgs

def lidarbmps(lidarid,vis,data,data_car,data_people,data_peopleskelett): #bbox model points skelett
    for eachdata in data:
        lidar_idx = json.loads(eachdata[1])
        if lidar_idx == lidarid:
            label = json.loads(eachdata[5])
            bbox_position = json.loads(eachdata[7])
            bbox_lwh = json.loads(eachdata[8])
            bbox_heading = json.loads(eachdata[9])
            print(bbox_heading)
            corners_3d = my_compute_box_3d(bbox_position, bbox_lwh, bbox_heading)
            bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
            if label == 1:
                colors = [[1, 0, 0] for _ in range(len(bbox_lines))]  # car - red
            else:
                colors = [[0, 0, 1] for _ in range(len(bbox_lines))]  # people - blue
            bbox = o3d.geometry.LineSet()
            bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
            bbox.colors = o3d.utility.Vector3dVector(colors)
            bbox.points = o3d.utility.Vector3dVector(corners_3d)
            vis.add_geometry(bbox)

            file_mesh = eachdata[12]
            if file_mesh != 'None':
                data_mesh = loadMesh(path_in_carmesh+file_mesh)
                data_mesh = np.array(data_mesh)
                data_carmesh = np.zeros(np.shape(data_mesh))
                data_carmesh[:, 0] = data_mesh[:, 2]
                data_carmesh[:, 1] = -1 * data_mesh[:, 0]
                data_carmesh[:, 2] = -1 * data_mesh[:, 1]
                T = np.array([bbox_position[0],bbox_position[1],bbox_position[2]]).T
                R = rotz(1 * bbox_heading)
                carmesh = []
                for eachdata in data_carmesh:
                    vertices = np.array(eachdata)
                    vertices_obj = vertices.T
                    vertices_world = np.dot(R, vertices_obj) + T
                    vertices_app = [vertices_world[0], vertices_world[1], vertices_world[2]]
                    carmesh.append(vertices_app)
                carmesh = np.array(carmesh)
                pointscarmesh = o3d.geometry.PointCloud()
                pointscarmesh.colors = o3d.utility.Vector3dVector([255, 0, 255] for _ in range(len(carmesh)))
                pointscarmesh.points = o3d.utility.Vector3dVector(carmesh)
                vis.add_geometry(pointscarmesh)

    if data_car != []:
        data_car = np.array(data_car)
        pointscar = o3d.geometry.PointCloud()
        pointscar.colors = o3d.utility.Vector3dVector([255, 0, 0] for _ in range(len(data_car)))
        pointscar.points = o3d.utility.Vector3dVector(data_car)
        vis.add_geometry(pointscar)

    if data_peopleskelett != []:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            # (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        num = int(len(data_peopleskelett)/17)
        for n in range(num):
            data_eachskelett = []
            for i in range(17):
                data_eachskelett.append(data_peopleskelett[17*n+i])
            data_eachskelett = np.array(data_eachskelett)
            skelett = o3d.geometry.LineSet()
            skelett.lines = o3d.utility.Vector2iVector(l_pair)
            skelett.colors = o3d.utility.Vector3dVector([0, 0, 0] for _ in range(len(data_eachskelett)))
            skelett.points = o3d.utility.Vector3dVector(data_eachskelett)
            vis.add_geometry(skelett)

    if data_people != []:
        data_people = np.array(data_people)
        pointspeople = o3d.geometry.PointCloud()
        pointspeople.colors = o3d.utility.Vector3dVector([0, 0, 255] for _ in range(len(data_people)))
        pointspeople.points = o3d.utility.Vector3dVector(data_people)
        vis.add_geometry(pointspeople)

    vis.get_view_control().convert_from_pinhole_camera_parameters(lidarView)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('tmpl.jpg')
    imgPC = cv2.imread('tmpl.jpg')
    return imgPC

if __name__ == '__main__':

    projectPath=sys.path[0]
    print(projectPath)
    with open("../input/newMeta.json", "r") as f:
        meta = json.load(f)
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("../output_new/video_result.mp4", fourcc, 2,
                             (960, 1080))
###3D Point CLoud viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 2
    ctr = vis.get_view_control()
    lidarView = o3d.io.read_pinhole_camera_parameters('../input/viewpoint.json')
    ctr.convert_from_pinhole_camera_parameters(lidarView)

    M = np.zeros((4, 4))
    M[:3, :3] = cv2.Rodrigues(np.asarray([0.3442905614582061, 2.466378657279374, -1.7432149451377668]))[0]
    M[:3, 3] = [-17.1982144948334, 2.6602985635296443, 47.41931873523568]
    M[3,3] = 1
    M_INV = np.linalg.inv(M)

    data = loadCsv1('../output_new/output.csv')
    path_in_carresult = '../result_new/res7car/'
    path_in_carmesh = '../input/carmesh/'
    path_in_peopleresult = '../result_new/res7people/'
    path_in_peopleskelett = '../output_new/people_skelett/'
    filelist_carresult = os.listdir(path_in_carresult)
    filelist_peopleresult = os.listdir(path_in_peopleresult)
    filelist_peopleskelett = os.listdir(path_in_peopleskelett)
    # data_gt = loadCsv2('../output/test.csv')

##Start reading data
    for exp in [10]:#meta["experiment"]:
        cap = []
        maxFPS = 0
        ses = []
        skip=0
        print("loading exp: "+str(exp)+"...")
        for dev in meta["device"]:
           print(meta["device"])
           if meta["device"][dev]["type"]=="lidar":
                continue
           id=meta["experiment"][str(exp)][str(dev)]
           #path to input videos
           name=os.path.join("../input/video_bbox.mp4")
           if not os.path.isfile(name):
                print(name +" don't exist")
                skip=1
                continue
           cap.append(cv2.VideoCapture(name))
           ses.append(meta["session"][str(id)])
           maxFPS=max(maxFPS,ses[-1]['fps'])##find fastest camera
        if skip==1:
            continue
        print(str(len(cap))+ " cameras found")

        ##Path to lidar data
        lidarPath=os.path.join("../input/lidarcloud/")
        pcFiles = glob.glob(os.path.join(lidarPath,"*.ply"))
        pcFiles.sort()
        print(str(len(pcFiles))+" lidar files found")

        colors=[[0,0,255],[255,0,0],[0,255,0],[255,255,0],[0,255,255]]
        for i in tqdm(range(611)):
            plydata = PlyData.read(pcFiles[i])
            lidarid = int(re.findall('\d+', pcFiles[i])[0])
            print(lidarid)
            ply = plydata['vertex']
            points = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()
            ##plot point cloud in gry
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            gray=np.ones((points.shape[0],3))*0.7
            pcd.colors=o3d.utility.Vector3dVector(gray)
            vis.clear_geometries()
            vis.add_geometry(pcd)

            ##find lidar id
            ids = ply["id"]
            uniqIds = np.unique(ids)
            times=ply["time"]*pow(10. ,-6)##time of point in micro seconds
            frames=np.floor(maxFPS*times)##generate maximum frame number
            uniqFrames=np.unique(frames)##getting frames per point cloud
            for j in range(len(uniqFrames)):
                f=uniqFrames[j]
                idx = np.where(frames == f)## active points at this frame
                if(len(idx[0])<1000):
                    continue
                tmpPoints=points[idx[0],:]
                tmpIds=ids[idx[0]]
                t=f/maxFPS

                data_car = []
                for file_car in filelist_carresult:
                    if str('%04d' % lidarid) + '_' in file_car:
                        with open(path_in_carresult+file_car, 'r') as f:
                            for jf in f:
                                eachdata = json.loads(jf)
                                data_car.append(eachdata)
                data_people = []
                for file_people in filelist_peopleresult:
                    if str('%04d' % lidarid) + '_' in file_people:
                        with open(path_in_peopleresult+file_people, 'r') as f:
                            for jf in f:
                                eachdata = json.loads(jf)
                                data_people.append(eachdata)
                data_peopleskelett = []
                for file_peopleskelett in filelist_peopleskelett:
                    if str('%04d' % lidarid) + '_' in file_peopleskelett:
                        with open(path_in_peopleskelett+file_peopleskelett, 'r') as f:
                            for jf in f:
                                eachdata = json.loads(jf)
                                data_peopleskelett.append(eachdata)

                imgPC = lidarbmps(lidarid,vis,data,data_car,data_people,data_peopleskelett)
                imgs = camerabbox(cap,t,ses)
                # imgPC=plot_3D(tmpPoints, tmpIds, uniqIds, colors, vis)##plotting lidar color active at this frame
                # imgs=project_points(tmpPoints, cap, t, ses, uniqIds, tmpIds, colors)##find image for each camera and draw colored points
                imgs.append(imgPC)
                img1 = cv2.resize(imgs[0], (int(1920 / 2), int(1080 / 2))) #resize error
                img2 = cv2.resize(imgs[1], (int(1920 / 2), int(1080 / 2)))
                globImg = np.zeros((int(1080), int(1920/2), 3)).astype(np.uint8)
                globImg[:int(1080 / 2), :int(1920/2), :] = img1
                globImg[int(1080 / 2):, :int(1920/2), :] = img2
                cv2.putText(globImg, "Experiment "+str(exp), (10, 1000), 1, 3, (255, 255, 255), 2)
                writer.write(globImg)
                cv2.imshow("test",globImg)
                cv2.waitKey(1)
    writer.release()
