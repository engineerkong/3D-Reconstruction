import numpy as np
import json
import os
import re
import cv2
from kd_tree import *

def loadData(fileName):
    data = []
    with open (fileName, 'r') as f:
        for jf in f:
            eachdata = json.loads(jf)
            data.append(eachdata)
    return data

def writeData(fileName, data):
    with open(fileName, 'w') as f:
        i = 0
        for d in data:
            i += 1
            # d = d.tolist()
            json.dump(d, f)
            if i != len(data):
                f.write('\n')

def keypoints3dto2d(keypoints3d):
    K = [[846.2772801361476, 0.0, 988.4546133916215, 0.0], [0.0, 851.9068507035013, 533.3470761399655, 0.0], [0.0, 0.0, 1.0, 0.0]]
    M = np.zeros((4, 4))
    M[:3, :3] = cv2.Rodrigues(np.asarray([0.3442905614582061, 2.466378657279374, -1.7432149451377668]))[0]
    M[:3, 3] = [-17.1982144948334, 2.6602985635296443, 47.41931873523568]

    keypoints2d = []
    index = 0
    for points in keypoints3d:
        points3d = np.array([points[0],points[1],points[2],1]).T
        points3d = np.dot(M,points3d)
        points2d = np.dot(K,points3d)
        scale = points2d[2]
        points2d = [points2d[0]/scale,points2d[1]/scale,index]
        keypoints2d.append(points2d)
        index += 1

    return keypoints2d

def matching_function(src,dst,dst3info):
    A = np.array(src)
    B = np.array(dst3info)
    kdtree = KDTree(dst,2)
    kdtree_dis = []
    kdtree_points = []
    for t in A:
        kdtree_dis.append(kdtree.get_nearest(t)[0])
        kdtree_points.append(kdtree.get_nearest(t)[1])
    kdtree_dis = np.array(kdtree_dis)
    kdtree_points = np.array(kdtree_points)
    dis = np.mean(kdtree_dis)
    C = np.zeros((len(src),3))
    for i in range(len(src)):
        kdtree_point = kdtree_points[i,:]
        for point in B:
            if kdtree_point[0]==point[0] and kdtree_point[1]==point[1]:
                index = point[2]
        C[i,0] = A[i,0]
        C[i,1] = A[i,1]
        C[i,2] = index
    return C

def findskelett(path_in_people2d,path_in_people3d,ratio,path_out_people):
    if not os.path.exists(path_out_people):
        os.makedirs(path_out_people)
    with open(path_in_people2d) as f:
        people_list = json.load(f)
    filelist_people3d = os.listdir(path_in_people3d)

    for people_dict in people_list:
        imageid2d = people_dict['image_id']
        imageid2d = re.findall('\d+', imageid2d)
        imageid2d = int(imageid2d[0])
        imageid2d = imageid2d
        index2d = people_dict['idx']
        for file_people3d in filelist_people3d:
            peoplename = re.findall('\d+', file_people3d)
            imageid3d = int(peoplename[0])
            index3d = int(peoplename[1])
            if imageid2d == (ratio*imageid3d) and index2d == index3d:
                keypoints = people_dict['keypoints']
                data_people2d = []
                for i in range(17):  # 17 skeleton keypoints on one people
                    data_people2d.append([keypoints[i * 3], keypoints[i * 3 + 1]])
                data_people3d = loadData(path_in_people3d + file_people3d)
                data_people2dprojectionindex = keypoints3dto2d(data_people3d)
                data_people2dprojection = []
                for eachdata in data_people2dprojectionindex:
                    data_people2dprojection.append([eachdata[0],eachdata[1]])
                if data_people2dprojection != []:
                    # print('data_people2d: ' + str(data_people2d))
                    # print('data_people2dprojection: ' + str(data_people2dprojection))
                    data_matched = matching_function(data_people2d,data_people2dprojection,data_people2dprojectionindex)
                    # print('data_matched: ' + str(data_matched))
                else:
                    data_matched = []
                data_out = []
                for data_m in data_matched:
                    i = 0
                    for data_p in data_people3d:
                        if data_m[2] == i:
                            data_out.append(data_p)
                        i += 1
                writeData(path_out_people+file_people3d,data_out) # the order is 1 2 3 4 .. 17
