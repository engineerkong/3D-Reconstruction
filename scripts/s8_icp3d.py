import os
import re
import json
import numpy as np
from kd_tree import *


def loadData(fileName):
    data = []
    with open(fileName, 'r') as f:
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


def outputData(fileName, data):
    with open(fileName, 'w') as f:
        i = 0
        for d in data:
            i += 1
            d = d.tolist()
            json.dump(d, f)
            if i != len(data):
                f.write('\n')


def outputInt(fileName, data):
    with open(fileName, 'w') as f:
        json.dump(data, f)


def find_optimal_transform(P, Q):
    meanP = np.mean(P, axis=0)
    meanQ = np.mean(Q, axis=0)
    P_ = P - meanP
    Q_ = Q - meanQ
    W = np.dot(Q_.T, P_)
    U, S, VT = np.linalg.svd(W)
    R = np.dot(U, VT)
    T = meanQ.T - np.dot(R, meanP.T)
    return R, T


def find_t_heading(P, Q):
    meanP = np.mean(P, axis=0)
    meanQ = np.mean(Q, axis=0)
    P_ = P - meanP
    Q_ = Q - meanQ
    cs = 0
    ss = 0
    for i in range(len(P_)):
        cs += (P_[i][0] * Q_[i][0] + P_[i][1] * Q_[i][1])
        ss += (-P_[i][0] * Q_[i][1] + P_[i][0] * Q_[i][0])
    a = np.arctan2(ss, cs)
    c = np.cos(a)
    s = np.sin(a)
    tx = meanP[0] - (c * meanQ[0] - s * meanQ[1])
    ty = meanP[1] - (s * meanQ[0] + c * meanQ[1])
    tz = meanP[2] - meanQ[2]
    t = [tx, ty, tz]
    return t, a


def icpfunction(src, dst, maxIteration=100, tolerance=0.01):
    P = np.array(src)
    R_target = np.eye(3)
    T_target = np.zeros((3, 1))
    RErr_old = 1
    TErr_old = 1
    print(np.shape(src))
    print(np.shape(dst))
    kdtree = KDTree(dst, 3)
    for i in range(maxIteration):
        kdtree_dis = []
        kdtree_points = []
        for t in P:
            kdtree_dis.append(kdtree.get_nearest(t)[0])
            kdtree_points.append(kdtree.get_nearest(t)[1])
        kdtree_dis = np.array(kdtree_dis)
        kdtree_points = np.array(kdtree_points)
        R, T = find_optimal_transform(P, kdtree_points)
        P = np.dot(R, P.T).T + np.array([T for j in range(P.shape[0])])
        meanErr = np.sum(kdtree_dis) / kdtree_dis.shape[0]
        RErr = np.linalg.det(R - R_target)
        TErr = np.linalg.norm(T - T_target)
        print("Iteration : " + str(i + 1) + " with Err : " + str(meanErr) + " RErr : " + str(RErr) + " TErr : " + str(
            TErr))
        if abs(RErr_old) < tolerance and abs(TErr_old) < tolerance and abs(RErr) < tolerance and abs(
                TErr) < tolerance:  # compare rotation and translation
            break
        RErr_old = RErr
        TErr_old = TErr

    t, a = find_t_heading(P, np.array(src))  # data1 to data2
    return t, a


# use icp 3d to find the rotation and translation between keypoints and meshpoints
# def icp3d_world(path_in_keypoints, path_in_model_world, path_out_t_error, path_out_a_error):
#     if not os.path.exists(path_out_t_error):
#         os.makedirs(path_out_t_error)
#     if not os.path.exists(path_out_a_error):
#         os.makedirs(path_out_a_error)
#     filelist_keypoints = os.listdir(path_in_keypoints)
#     filelist_model_world = os.listdir(path_in_model_world)
#     for file_keypoints in filelist_keypoints:
#         id_keypoints = re.findall('\d+', file_keypoints)
#         for file_model_world in filelist_model_world:
#             id_model = re.findall('\d+', file_model_world)
#             if int(id_model[1]) == int(id_keypoints[1]):
#                 print('id_model:' + str(int(id_model[1])))
#                 data_keypoints = loadData(path_in_keypoints + file_keypoints)
#                 data_model_world = loadData(path_in_model_world + file_model_world)
#                 data_model_world = np.array(data_model_world)
#                 t_error, a_error = icpfunction(data_model_world, data_keypoints, maxIteration=10, tolerance=0.1)
#                 outputData(path_out_t_error + id_keypoints[1] + '.json', t_error)
#                 outputInt(path_out_a_error + id_keypoints[1] + '.json', a_error)


def icp3d_obj(path_in_carkeypoints, path_in_model_obj, path_out_t, path_out_a):
    if not os.path.exists(path_out_t):
        os.makedirs(path_out_t)
    if not os.path.exists(path_out_a):
        os.makedirs(path_out_a)
    filelist_keypoints = os.listdir(path_in_carkeypoints)
    filelist_model_obj = os.listdir(path_in_model_obj)
    for file_keypoints in filelist_keypoints:
        id_keypoints = re.findall('\d+', file_keypoints)
        for file_model_obj in filelist_model_obj:
            id_model = re.findall('\d+', file_model_obj)
            if int(id_model[1]) == int(id_keypoints[1]):
                print('id_model:' + str(int(id_model[1])))
                data_keypoints = loadData(path_in_carkeypoints + file_keypoints)
                data_model_obj = loadMesh(path_in_model_obj + file_model_obj)
                data_model_obj = np.array(data_model_obj)
                # data_model_obj = np.zeros(np.shape(data_mesh))
                # data_model_obj[:, 0] = data_mesh[:, 2]
                # data_model_obj[:, 1] = -1 * data_mesh[:, 0]
                # data_model_obj[:, 2] = -1 * data_mesh[:, 1]
                t, a = icpfunction(data_model_obj, data_keypoints, maxIteration=10, tolerance=0.1)
                outputData(path_out_t + id_keypoints[1] + '.json', t)
                outputInt(path_out_a + id_keypoints[1] + '.json', a)